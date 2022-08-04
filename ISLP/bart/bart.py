
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

from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state

from joblib import Parallel, delayed

from sklearn.tree._tree import Tree
from .particle import (SequentialTreeBuilder,
                       root_tree)

class BART(BaseEnsemble, RegressorMixin):
    """
    Particle Gibbs BART sampling step

    Parameters
    ----------
    num_particles : int
        Number of particles for the conditional SMC sampler. Defaults to 10
    max_stages : int
        Maximum number of iterations of the conditional SMC sampler. Defaults to 100.

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
                 num_trees=200,
                 num_particles=10,
                 max_stages=5000,
                 split_prob=lambda depth: 0.95/(1+depth)**2,
                 std_scale=2,
                 split_prior=None,
                 ndraw=100,
                 burnin=100,
                 sigma_prior=(5, 0.9),
                 num_quantile=50,
                 random_state=None,
                 n_jobs=-1,
                 max_depth=10):

        self.num_particles = num_particles
        self.max_stages = max_stages
        self.num_trees = num_trees
        self.split_prob = split_prob
        self.std_scale = std_scale
        self.split_prior = split_prior
        self.ndraw = ndraw
        self.burnin = burnin

        self.sigma_prior = sigma_prior
        self.num_quantile = num_quantile
        
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Chipman's default for prior
        self.mu_prior_var_ = (0.5 / (self.std_scale * np.sqrt(self.num_trees)))**2
        self.mu_prior_mean_ = 0 
        self.max_depth = max_depth

    def predict(self,
                X):

        check_is_fitted(self)

        nsample = len(self.trees_sample_)
        output = np.zeros(X.shape[0], float)

        for trees in self.trees_sample_:
            for tree in trees:
                tree_fit = np.squeeze([tree.predict(X.astype(np.float32))])
                output += tree_fit
        output = output / nsample
        return self._inverse(output)

    def staged_predict(self,
                       X,
                       start_idx=0):

        check_is_fitted(self)

        trees_sample_ = self.trees_sample_[start_idx:]
        nsample = len(trees_sample_)
        output = np.zeros((nsample, X.shape[0]), np.float)

        for nstep, trees in enumerate(trees_sample_):
            for tree in trees:
                tree_fit = np.array([tree.predict_out_of_sample(x) for x in X])
                output[nstep] += tree_fit
                
        output = np.cumsum(output, 0) / (np.arange(nsample) + 1)[:,None]
        return self._inverse(output)

    def fit(self,
            X,
            Y,
            sample_weight=None):

        tree_sampler = TreeSampler(max_depth=self.max_depth,
                                   num_particles=self.num_particles,
                                   max_stages=self.max_stages,
                                   random_state=self.random_state,
                                   sigmasq=1, # will be set later
                                   mu_prior_mean=self.mu_prior_mean_,
                                   mu_prior_var=self.mu_prior_var_,
                                   split_prob=self.split_prob)
        X = np.asarray(X)
        Y = np.asarray(Y)

        random_state = check_random_state(self.random_state)
        n_jobs = self.n_jobs
        if self.n_jobs <= 0:
            n_jobs = 1

        random_idx = random_state.randint(0,2**32-1,size=(n_jobs,))

        parallel = Parallel(n_jobs=len(random_idx))

        qvals = np.linspace(0, 100, self.num_quantile+2)[1:-1]
        if self.num_quantile is not None:
            X_quantiles = np.percentile(X,
                                        qvals,
                                        axis=0)
        else:
            X_quantiles = None

        X_missing = np.any(np.isnan(X))

        # Chipman's defaults according to Lakshminarayanan
        # scale to range [-0.5,0.5] 
        _Y_min, _Y_max = np.nanmin(Y), np.nanmax(Y)
        _forward = lambda y: ((y - _Y_min) / (_Y_max - _Y_min) - 0.5)
        self._inverse = lambda out: (out + 0.5) * (_Y_max - _Y_min) + _Y_min
        Y_shift = _forward(Y)

        sigmasq = np.var(Y_shift)
        A, q = self.sigma_prior
        sigma_prior_B = invgamma(A, 0, 1).ppf(q) * sigmasq

        # args for each job
        args = (X,
                X_quantiles,
                X_missing,
                _forward,
                Y_shift,
                sigma_prior_B,
                tree_sampler)

        self.trees_sample_ = []
        self.variable_inclusion_ = []
        #self.depths_ = []
        self.num_leaves_ = []
        
        work = parallel(delayed(clone(self)._gibbs_sample)(*(args + (rs,)))
                        for rs in random_idx)

        for (atts,
             batch_trees,
             variable_inclusion) in work:
            self.trees_sample_.extend(batch_trees)
            self.variable_inclusion_.extend(variable_inclusion)

        for key, value in atts.items():
            setattr(self, key, value)

#        for particles in self.trees_sample_:
            #self.depths_.append([tree._max_depth for tree in particles])
            #self.num_leaves_.append([len(tree.idx_leaf_nodes) for tree in particles])

        self.variable_inclusion_ = np.array(self.variable_inclusion_)
#        self.depths_ = np.array(self.depths_)
#        self.num_leaves_ = np.array(self.num_leaves_)
        
        return self

    # Private methods

    def _gibbs_sample(self,
                      X,
                      X_quantiles,
                      X_missing,
                      _forward,
                      Y_shift,
                      sigma_prior_B,
                      tree_sampler,
                      random_state):

        random_state = check_random_state(random_state)

        variable_inclusion = []
        depths = []
        num_leaves = []
        
        num_observations_ = X.shape[0]
        self.num_features_ = X.shape[1]
        available_predictors = list(range(self.num_features_))

        init_mean = Y_shift.mean()    
        sum_trees_output = np.full_like(Y_shift, init_mean)

        init_value_ = init_mean / self.num_trees
        self.init_idx_ = np.arange(num_observations_, dtype=int)
        self.particle_indices_ = list(range(1, self.num_particles))

        split_prior = self.split_prior or np.ones(X.shape[1])
        ssv = SampleSplittingVariable(split_prior, random_state)

        sigmasq = np.var(Y_shift)

        # instantiate the particles
        
        self.all_trees_ = []
        self._all_fits_ = np.empty((self.num_trees, num_observations_))
        self._all_leaf_maps_ = np.zeros((self.num_trees, num_observations_), dtype=np.intp)

        sum_trees_output = np.full_like(Y_shift, init_mean)

        init_resid = Y_shift - init_mean * (self.num_trees - 1) / self.num_trees
        classes = np.array([1])
        for i in range(self.num_trees):
            new_tree = root_tree(self.num_features_, num_observations_)
            self.all_trees_.append(new_tree)
            self._all_fits_[i] = init_mean / self.num_trees
            
        counter = 0
        batch_trees = []
        while True:
            (particle_trees,
             sum_trees_output,
             stats) = self._gibbs_step_tree_value(X,
                                                  Y_shift,
                                                  sigmasq,
                                                  sum_trees_output,
                                                  random_state,
                                                  tree_sampler)
            sigmasq = self._gibbs_step_sigma(Y_shift - sum_trees_output,
                                             sigma_prior_B,
                                             random_state)
            if counter >= self.burnin:
                batch_trees.append(particle_trees)
                variable_inclusion.append(stats['variable_inclusion'])

            if len(batch_trees) == self.ndraw:
                break

            counter += 1

        variable_inclusion = np.array(variable_inclusion)

        atts = {'sigma_prior_B_': sigma_prior_B}

        return (atts,
                batch_trees,
                variable_inclusion)
                
    def _gibbs_step_sigma(self,
                          resid,
                          sigma_prior_B,
                          random_state):

        n = resid.shape[0]
        A = self.sigma_prior[0] + n / 2
        B = sigma_prior_B + (resid**2).sum() / 2
        
        return invgamma(A,
                        0,
                        B).rvs(random_state=random_state)
    
    def _gibbs_step_tree_value(self,
                               X,
                               Y,
                               sigmasq,
                               sum_trees_output,
                               random_state,
                               tree_sampler):

        # update the sigmasq parameter
        
        tree_sampler.sigmasq = sigmasq
        variable_inclusion = np.zeros(self.num_features_, int)

        total_stages = 0
        for tree_id in range(self.num_trees):
            # Generate an initial set of SMC particles
            # at the end of the algorithm we return one of these particles as the new tree

            # Compute the sum of trees without the tree we are attempting to replace

            cur_tree = self.all_trees_[tree_id]
            cur_leaf_map = self._all_leaf_maps_[tree_id]
            cur_fit = self._all_fits_[tree_id]

            sum_trees_output_noi = sum_trees_output - cur_fit
            resid_noi = Y - sum_trees_output_noi

            (new_tree,
             new_leaf_map,
             new_fit) = tree_sampler.sample(cur_tree,
                                            cur_leaf_map,
                                            X, # keep a copy of modified X if it has been modified
                                            resid_noi.astype(np.float32))

            self.all_trees_[tree_id] = new_tree
            self._all_leaf_maps_[tree_id] = new_leaf_map
            self._all_fits_[tree_id] = new_fit

            sum_trees_output += new_fit - cur_fit

        stats = {"variable_inclusion": variable_inclusion}
        return self.all_trees_, sum_trees_output, stats

# Private functions

class TreeSampler(SequentialTreeBuilder):

    pass

# todo: if we want different features with different frequency

class SampleSplittingVariable(object):

    def __init__(self,
                 alpha_prior,
                 random_state):
        """
        Sample splitting variables proportional to `alpha_prior`.

        This is equivalent as sampling weights from a Dirichlet distribution with `alpha_prior`
        parameter and then using those weights to sample from the available spliting variables.
        This enforce sparsity.
        """
        self.enu = list(enumerate(np.cumsum(alpha_prior / alpha_prior.sum())))
        self.random_state = random_state
        
    def rvs(self):
        r = self.random_state.random()
        for i, v in self.enu:
            if r <= v:
                return i

