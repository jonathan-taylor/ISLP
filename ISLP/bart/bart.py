
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
from scipy.stats import invgamma

from sklearn.base import RegressorMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_is_fitted


from .tree import Tree
from .likelihood import marginal_loglikelihood
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
                 max_stages=5000,
                 batch="auto",
                 m=200,
                 alpha_split=0.95,
                 beta_split=2,
                 k=2,
                 split_prior=None,
                 ndraw=100,
                 burnin=100,
                 keep_every=1,
                 sigma_prior_A=3,
                 sigma_prior_q=0.9):

        self.num_particles = num_particles
        self.max_stages = max_stages
        self.batch = batch
        self.m = m
        self.alpha_split = alpha_split
        self.beta_split = beta_split
        self.k = k
        self.split_prior = split_prior
        self.ndraw = ndraw
        self.burnin = burnin
        self.keep_every = keep_every

        self.sigma_prior_A = sigma_prior_A
        self.sigma_prior_q = sigma_prior_q

    def fit(self,
            X,
            Y,
            sample_weight=None):

        X = np.asarray(X)
        Y = np.asarray(Y)
        
        self.variable_inclusion_ = []

        missing_data = np.any(np.isnan(X))

        # Chipman's defaults according to Lakshminarayanan
        self._Y_min, self._Y_max = np.nanmin(Y), np.nanmax(Y)
        self._forward = lambda y: ((y - self._Y_min) / (self._Y_max - self._Y_min) - 0.5)
        self._inverse = lambda out: (out + 0.5) * (self._Y_max - self._Y_min) + self._Y_min
        Y_shift = self._forward(Y)
        init_mean = Y_shift.mean()    
       
        self.mu_prior_std_ = 0.5 / (self.k * np.sqrt(self.m))
        self.mu_prior_mean_ = 0 # mean of prior for mu

        self.num_observations_ = X.shape[0]
        self.num_variates_ = X.shape[1]
        available_predictors = list(range(self.num_variates_))

        sum_trees_output = np.full_like(Y_shift, init_mean)

        self.base_tree_ = Tree.init_tree(
            tree_id=0,
            leaf_node_value=init_mean / self.m,
            idx_data_points=np.arange(self.num_observations_, dtype="int32"),
        )

        self.indices_ = list(range(1, self.num_particles))

        split_prior = self.split_prior or np.ones(X.shape[1])
        ssv = SampleSplittingVariable(split_prior)

        sigma = np.std(Y_shift)
        sigma_prior_B = invgamma(self.sigma_prior_A, 0, 1).ppf(self.sigma_prior_q) * sigma

        # instantiate the particles
        
        self.all_particles_ = []
        sum_trees_output = 0
        for i in range(self.m):
            new_tree = self.base_tree_.copy()
            new_tree.tree_id = i
            log_weight = marginal_loglikelihood(Y_shift - init_mean * (self.m - 1) / self.m,
                                                sigma,
                                                self.mu_prior_mean_,
                                                self.mu_prior_std_)
            p = ParticleTree(new_tree,
                             self.alpha_split,
                             self.beta_split,
                             log_weight,
                             missing_data,
                             ssv,
                             available_predictors,
                             self.m,
                             sigma,
                             self.mu_prior_mean_,
                             self.mu_prior_std_)

            self.all_particles_.append(p)
            sum_trees_output += p.tree.predict_output()
            
        counter = 0
        self.trees_sample_ = []
        while True:
            trees, sum_trees_output, stats = self._gibbs_step_tree_value(X,
                                                                         Y_shift,
                                                                         sigma,
                                                                         sum_trees_output)
            sigma = self._gibbs_step_sigma(Y_shift - sum_trees_output,
                                           sigma_prior_B)
            if counter >= self.burnin and ((counter - self.burnin) % self.keep_every == 0):
                self.trees_sample_.append(trees)
            if counter - self.burnin >= self.ndraw * self.keep_every:
                break
            counter += 1

            self.variable_inclusion_.append(stats['variable_inclusion'])

        self.variable_inclusion_ = np.array(self.variable_inclusion_)

    def predict(self,
                X):

        check_is_fitted(self)

        nsample = len(self.trees_sample_)
        output = np.zeros(X.shape[0], np.float)

        for trees in self.trees_sample_:
            for tree in trees:
                tree_fit = np.array([tree.predict_out_of_sample(x) for x in X])
                output += tree_fit
        output = output / nsample
        return self._inverse(output)

    # Private methods

    def _gibbs_step_sigma(self,
                          resid,
                          sigma_prior_B):

        n = resid.shape[0]
        A = self.sigma_prior_A + n / 2
        B = sigma_prior_B + (resid**2).sum() / 2

        return np.sqrt(invgamma(A, 0, B).rvs())
    
    def _gibbs_step_tree_value(self,
                               X,
                               Y,
                               sigma,
                               sum_trees_output):

        variable_inclusion = np.zeros(self.num_variates_, dtype="int")

        if self.batch == "auto":
            batch = max(1, int(self.m * 0.1))
        else:
            batch = self.batch

        for tree_id in range(self.m):
            # Generate an initial set of SMC particles
            # at the end of the algorithm we return one of these particles as the new tree

            # Compute the sum of trees without the tree we are attempting to replace

            cur_particle = self.all_particles_[tree_id]
            sum_trees_output_noi = sum_trees_output - cur_particle.tree.predict_output()
            resid_noi = Y - sum_trees_output_noi

            particles = self.init_particles(cur_particle,
                                            sigma,
                                            resid_noi)

            for t in range(self.max_stages):
                # sample each particle (try to grow each tree)
                for p in particles[1:]:
                    # this is log_likelihood_ratio for the split if there was one
                    # so if tree does not grow this is just 0
                    # line 9 of Algorithm 2 of Lakshminarayanan
                    tree_grew, left_node, right_node = p.sample_tree_sequential(
                        X,
                        resid_noi,
                    )
                    # line 12 of Algorithm 2 of Lakshminarayanan
                    if tree_grew:
                        p.log_weight += p.increment_loglikelihood(resid_noi,
                                                                  left_node,
                                                                  right_node)
                        
                # line 13 of Algorithm 2 of Lakshminarayanan
                W_t, normalized_weights = _normalize(particles)

                # line 14-15 of Algorithm 2 of Lakshminarayanan
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
            new_particle.sample_values(resid_noi)
            new_tree = new_particle.tree
            new_particle.log_weight = W_t - np.log(len(particles))
            # now sample the mean parameters within each leaf

            self.all_particles_[new_tree.tree_id] = new_particle
            sum_trees_output = sum_trees_output_noi + new_tree.predict_output()

            for index in new_particle.used_variates:
                variable_inclusion[index] += 1

        stats = {"variable_inclusion": variable_inclusion}
        return [p.tree.copy() for p in self.all_particles_], sum_trees_output, stats

    def init_particles(self,
                       base_particle: ParticleTree,
                       sigma: float,
                       resid: np.ndarray) -> np.ndarray:
        """
        Initialize particles
        """
        p = base_particle

        init_loglikelihood = p.marginal_loglikelihood(resid)
        p.log_weight = init_loglikelihood
        particles = [p]

        for _ in self.indices_:
            new_tree = self.base_tree_.copy()
            new_tree.tree_id = p.tree.tree_id
            particles.append(
                ParticleTree(
                    new_tree,
                    p.alpha_split,
                    p.beta_split,
                    init_loglikelihood,
                    p.missing_data,
                    p.ssv,
                    p.available_predictors,
                    p.m,
                    sigma,
                    p.mu_prior_mean,
                    p.mu_prior_std
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

class SampleSplittingVariable(object):

    def __init__(self, alpha_prior):
        """
        Sample splitting variables proportional to `alpha_prior`.

        This is equivalent as sampling weights from a Dirichlet distribution with `alpha_prior`
        parameter and then using those weights to sample from the available spliting variables.
        This enforce sparsity.
        """
        self.enu = list(enumerate(np.cumsum(alpha_prior / alpha_prior.sum())))

    def rvs(self):
        r = np.random.random()
        for i, v in self.enu:
            if r <= v:
                return i

