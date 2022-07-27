#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   Modified for ISLP by Jonathan Taylor 2021

from typing import Any, Dict, List, Tuple

import numpy as np

from sklearn.base import RegressorMixin
from sklearn.ensemble import BaseEnsemble

from .tree import Tree
from .utils import (SampleSplittingVariable,
                    #marginal_loglikelihood,
                    compute_prior_probability)
from .particle_tree import ParticleTree

class BART(BaseEnsemble, RegressorMixin):
    """
    Particle Gibbs BART sampling step

    Parameters
    ----------
    vars: list
        List of value variables for sampler
    num_particles : int
        Number of particles for the conditional SMC sampler. Defaults to 10
    max_stages : int
        Maximum number of iterations of the conditional SMC sampler. Defaults to 100.
    batch : int
        Number of trees fitted per step. Defaults to  "auto", which is the 10% of the `m` trees
        during tuning and 20% after tuning.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).

    Note
    ----
    This sampler is inspired by the [Lakshminarayanan2015] Particle Gibbs sampler, but introduces
    several changes. The changes will be properly documented soon.

    References
    ----------
    .. [Lakshminarayanan2015] Lakshminarayanan, B. and Roy, D.M. and Teh, Y. W., (2015),
        Particle Gibbs for Bayesian Additive Regression Trees.
        ArviX, `link <https://arxiv.org/abs/1502.04622>`__
    """

    def __init__(self,
                 num_particles=10,
                 max_stages=100,
                 batch="auto",
                 m=50,
                 alpha=0.25,
                 k=2,
                 split_prior=None):

        self.num_particles = num_particles
        self.max_stages = max_stages
        self.batch = batch
        self.m = m
        self.alpha = alpha
        self.k = k
        self.split_prior = split_prior

    def fit(self,
            X,
            Y,
            sample_weight=None):

        missing_data = np.any(np.isnan(X))
        init_mean = Y.mean()    

        # if data is binary
        Y_unique = np.unique(Y)
        if Y_unique.size == 2 and np.all(Y_unique == [0, 1]):
            self.mu_std_ = 6 / (self.k * self.m ** 0.5)
        # maybe we need to check for count data
        else:
            self.mu_std_ = Y.std() / (self.k * self.m ** 0.5)
        self.mu_mean_ = 0 # mean of prior for mu

        self.num_observations_ = X.shape[0]
        self.num_variates_ = X.shape[1]
        available_predictors = list(range(self.num_variates_))

        sum_trees_output = np.full_like(Y, init_mean)
        self.init_tree_ = Tree.init_tree(
            tree_id=0,
            leaf_node_value=init_mean / self.m,
            idx_data_points=np.arange(self.num_observations_, dtype="int32"),
        )

        self.tune = True
        self._idx = 0
        self.iter = 0
        self.sum_trees = []

        log_num_particles = np.log(self.num_particles)
        self.indices_ = list(range(1, self.num_particles))

        # here we need to get likelihood

        prior_prob_leaf_node = compute_prior_probability(self.alpha)
        split_prior = self.split_prior or np.ones(X.shape[1])
        ssv = SampleSplittingVariable(split_prior)

        self.sigma_ = np.std(Y)
        self.init_likelihood_ = 0.
        self.init_log_weight_ = self.init_likelihood_ - log_num_particles

        self.all_particles_ = []
        for i in range(self.m):
            self.init_tree_.tree_id = i
            p = ParticleTree(
                self.init_tree_.copy(),
                self.init_log_weight_,
                self.init_likelihood_,
                missing_data,
                prior_prob_leaf_node,
                ssv,
                available_predictors,
                self.m,
                self.sigma_,
                self.mu_mean_,
                self.mu_std_
            )
            self.all_particles_.append(p)

    # Private methods

    def _step(self,
              X,
              Y,
              init_log_weight,
              init_likelihood,
              sum_trees_output):

        variable_inclusion = np.zeros(self.num_variates_, dtype="int")

        if self._idx == self.m:
            self._idx = 0

        if self.batch == "auto":
            batch = max(1, int(self.m * 0.1))
        else:
            batch = self.batch

        for tree_id in range(self._idx, self._idx + batch):
            if tree_id >= self.m:
                break
            # Generate an initial set of SMC particles
            # at the end of the algorithm we return one of these particles as the new tree

            particles = self.init_particles(tree_id,
                                            init_log_weight,
                                            init_likelihood)

            # Compute the sum of trees without the tree we are attempting to replace

            sum_trees_output_noi = sum_trees_output - particles[0].tree.predict_output()
            self._idx += 1

            # The old tree is not growing so we update the weights only once.
            particles[0].update_weight(Y,
                                       sum_trees_output_noi)

            for t in range(self.max_stages):
                # sample each particle (try to grow each tree)
                for p in particles[1:]:
                    tree_grew = p.sample_tree_sequential(
                        X,
                        Y,
                        sum_trees_output,
                    )
                    if tree_grew:
                        p.update_weight(Y,
                                        sum_trees_output_noi)

                W_t, normalized_weights = _normalize(particles)

                # Resample all but first particle
                re_n_w = normalized_weights[1:] / normalized_weights[1:].sum()
                new_indices = np.random.choice(self.indices_, size=len(self.indices_), p=re_n_w)
                particles[1:] = particles[new_indices]

                # Set the new weights
                for p in particles:
                    p.log_weight = W_t

                # Check if particles can keep growing, otherwise stop iterating
                non_available_nodes_for_expansion = []
                for p in particles[1:]:
                    if p.expansion_nodes:
                        non_available_nodes_for_expansion.append(0)
                if all(non_available_nodes_for_expansion):
                    break

            # Get the new tree and update
            new_particle = np.random.choice(particles, p=normalized_weights)
            new_tree = new_particle.tree
            new_particle.log_weight = new_particle.old_likelihood_logp - np.log(len(particles))
            self.all_particles_[new_tree.tree_id] = new_particle
            sum_trees_output = sum_trees_output_noi + new_tree.predict_output()

            # if self.tune:
            #     for index in new_particle.used_variates:
            #         self.split_prior[index] += 1
            #         self.ssv = SampleSplittingVariable(self.split_prior)
            # else:
            self.iter += 1
            self.sum_trees.append(new_tree)
            if not self.iter % self.m:
                # XXX update the all_trees variable in BARTRV to be used in the rng_fn method
                # this fails for chains > 1 as the variable is not shared between proccesses
                self.all_trees.append(self.sum_trees)
                self.sum_trees = []
            for index in new_particle.used_variates:
                variable_inclusion[index] += 1

        stats = {"variable_inclusion": variable_inclusion}
        return sum_trees_output, [stats]

    def init_particles(self,
                       tree_id: int,
                       init_log_weight: float,
                       init_likelihood: float) -> np.ndarray:
        """
        Initialize particles
        """
        p = self.all_particles_[tree_id]
        p.log_weight = self.init_log_weight_
        p.old_likelihood_logp = self.init_likelihood_
        particles = [p]

        for _ in self.indices_:
            self.init_tree_.tree_id = tree_id
            particles.append(
                ParticleTree(
                    self.init_tree_.copy(),
                    init_log_weight,
                    init_likelihood,
                    p.missing_data,
                    p.prior_prob_leaf_node,
                    p.ssv,
                    p.available_predictors,
                    p.m,
                    p.sigma,
                    p.mu_mean,
                    p.mu_std
                )
            )

        return np.array(particles)


# Private functions

def _normalize(particles: List[ParticleTree]) -> Tuple[float, np.ndarray]:
    """
    Use logsumexp trick to get W_t and softmax to get normalized_weights
    """
    log_w = np.array([p.log_weight for p in particles])
    log_w_max = log_w.max()
    log_w_ = log_w - log_w_max
    w_ = np.exp(log_w_)
    w_sum = w_.sum()
    W_t = log_w_max + np.log(w_sum) - np.log(log_w.shape[0])
    normalized_weights = w_ / w_sum
    # stabilize weights to avoid assigning exactly zero probability to a particle
    normalized_weights += 1e-12

    return W_t, normalized_weights

