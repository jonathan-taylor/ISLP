# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Lars Buitinck
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

# From https://github.com/scikit-learn/scikit-learn/blob/076442c815b194ce4d9238acb8e2285852196980/sklearn/tree/_tree.pyx
# From https://github.com/scikit-learn/scikit-learn/blob/076442c815b194ce4d9238acb8e2285852196980/sklearn/tree/_splitter.pyx
# Modified 2022 Jonathan Taylor

from libc.math cimport log as ln
from libc.math cimport sqrt

import numpy as np
cimport numpy as cnp
cnp.import_array()

from sklearn.tree._tree cimport TreeBuilder, Tree, SIZE_t, DOUBLE_t, DTYPE_t
from sklearn.utils import check_random_state
from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE
from queue import SimpleQueue
from collections import namedtuple

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED

cdef extern from "<stack>" namespace "std" nogil:
    cdef cppclass stack[T]:
        ctypedef T value_type
        stack() except +
        bint empty()
        void pop()
        void push(T&) except +  # Raise c++ exception for bad_alloc -> MemoryError
        T& top()
        
cdef struct StackRecord:
    SIZE_t depth
    SIZE_t node_id
    SIZE_t start
    SIZE_t end
    
Record = namedtuple('Record', 'node_id,depth,start,end')

cpdef root_tree(SIZE_t num_features,
                SIZE_t num_samples,
                cnp.ndarray classes = np.array([1])):
    tree = Tree(num_features, classes, 1)
    tree._add_node(_TREE_UNDEFINED,
                   1,
                   1,
                   _TREE_UNDEFINED,
                   _TREE_UNDEFINED,
                   0,
                   num_samples,
                   num_samples)
    return tree
                

cdef class SequentialTreeBuilder(TreeBuilder):
    """Build a decision tree sequentially for particle Gibbs BART

    Nodes are added randomly in a sequential fashion, keeping
    track of marginal likelihood.
    """
    cdef SIZE_t max_leaf_nodes

    def __cinit__(self, 
                  SIZE_t max_depth,
                  SIZE_t num_particles,
                  SIZE_t max_stages,
                  random_state,
                  float sigmasq,
                  float mu_prior_mean,
                  float mu_prior_var,
                  split_prob=lambda d:0.95/(1+d)**2):

        self.max_depth = max_depth
        self.num_particles = num_particles
        self.max_stages = max_stages
        self.random_ = check_random_state(random_state)
        self.sigmasq = sigmasq
        self.mu_prior_mean = mu_prior_mean
        self.mu_prior_var = mu_prior_var
        self.particle_indices = np.arange(1, num_particles, dtype=np.intp)
        self.split_prob = split_prob
        
    cpdef build(self, Tree tree, object X, cnp.ndarray y,
                cnp.ndarray sample_weight=None):
        """Build a decision tree from the training set (X, y)."""

        # check input
        if not hasattr(self, "X"):
            self.X, y, sample_weight = self._check_input(X, y, sample_weight)
        else:
            X = self.X
            
        cdef DOUBLE_t* sample_weight_ptr = NULL

        # structure to capture splits
        cdef DTYPE_t[::1] Xf = np.empty_like(X[:,0])
        cdef SIZE_t[::1] samples = np.arange(X.shape[0], dtype=np.intp)
        cdef SIZE_t[::1] leaf_map = -np.ones(X.shape[0], dtype=np.intp)

        if sample_weight is not None:
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Initial capacity
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # Parameters
        cdef SIZE_t max_depth = self.max_depth
        
        cdef SIZE_t start
        cdef SIZE_t split
        cdef SIZE_t end
        cdef SIZE_t depth
        cdef SIZE_t idx
        cdef SIZE_t parent
        cdef bint is_left
        cdef SIZE_t node_id
        
        cdef stack[StackRecord] expansion_nodes
        cdef StackRecord stack_record
        
        cdef double impurity = 0
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0

        # make a tree with a single leaf node
        # tree should be empty?

        root_node_id = tree._add_node(_TREE_UNDEFINED,
                                      0,
                                      True,
                                      _TREE_UNDEFINED,
                                      _TREE_UNDEFINED,
                                      0,
                                      X.shape[1],
                                      X.shape[1])
    
        expansion_nodes.push({'node_id':root_node_id,
                              'depth':0,
                              'start':0,
                              'end':X.shape[0]})

        cdef float loglikelihood = marginal_loglikelihood(y,
                                                          None,
                                                          1,
                                                          self.sigmasq,
                                                          self.mu_prior_mean,
                                                          self.mu_prior_var)
        # try to split

        while not expansion_nodes.empty():
            stack_record = expansion_nodes.top()
            expansion_nodes.pop()
            
            node_id = stack_record.node_id
            depth = stack_record.depth
            start = stack_record.start
            end = stack_record.end

            if ((start < end) and 
                (self.random_.random() < self.split_prob(depth))):
                feature = np.random.choice(X.shape[1])

                for idx in range(start, end):
                    Xf[idx] = X[samples[idx], feature]
                sort(&Xf[start], &samples[start], end - start)
                
                split = self.random_.randint(start, end)
                threshold = Xf[split]
                
                # setup the right node

                right_id = tree._add_node(node_id, 
                                          0,  # is_left
                                          1,  # is_leaf
                                          _TREE_UNDEFINED,
                                          _TREE_UNDEFINED, 
                                          impurity, 
                                          end-split,
                                          end-split)
                
                # add it to stack as a candidate to split
                
                expansion_nodes.push({'node_id':right_id,
                                      'depth':depth+1,
                                      'start':split+1,
                                      'end':end})

                # repeat for left node

                left_id = tree._add_node(node_id, 
                                         1,  # is_left
                                         1,  # is_leaf
                                         _TREE_UNDEFINED,
                                         _TREE_UNDEFINED, 
                                         impurity, 
                                         split-start,
                                         split-start)
                
                expansion_nodes.push({'node_id':left_id,
                                      'depth':depth+1,
                                      'start':start,
                                      'end':split+1})
                
                # update current leaf to a split node

                tree.nodes[node_id].feature = feature
                tree.nodes[node_id].threshold = threshold
                tree.nodes[node_id].left_child = left_id
                tree.nodes[node_id].right_child = right_id
           
                # increment the loglikelihood

                loglikelihood += incremental_loglikelihood(y,
                                                           samples,
                                                           start,
                                                           split,
                                                           end,
                                                           self.sigmasq,
                                                           self.mu_prior_mean,
                                                           self.mu_prior_var)[0]

                if depth > max_depth_seen:
                    max_depth_seen = depth
                                        
            else:
                # the node will not be split again
                # update the leaf_map
                # this is used to be able to efficiently
                # compute marginal likelihood when the response is changed
                
                for idx in range(start, end):
                    leaf_map[samples[idx]] = node_id
                    
            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen
        if rc == -1:
            raise MemoryError()

        return tree, loglikelihood, np.asarray(leaf_map).astype(np.intp)

    cpdef sample(self,
                 Tree cur_tree, # tree we will resample
                 SIZE_t[::1] cur_leaf_map,
                 object X,
                 DTYPE_t[::1] response):
        """Sample a tree sequentially at random according to a BART prior for X, returning a loglikelihood based on response."""

        # check input
        if not hasattr(self, "X"):
            self.X, _, _ = self._check_input(X, np.asarray(response), None)
        else:
            X = self.X
            
        # prior parameters

        cdef float sigmasq = self.sigmasq
        cdef float mu_prior_mean = self.mu_prior_mean
        cdef float mu_prior_var = self.mu_prior_var

        cdef SIZE_t num_particles = self.num_particles
        cdef SIZE_t num_features = X.shape[1]
        cdef SIZE_t num_samples = X.shape[0]

        # structure to capture splits
        Xf_all = np.empty((num_particles-1, X.shape[0]), dtype=self.X.dtype) # used to keep track of node_map of each particle tree
        samples_all = np.multiply.outer(np.ones(num_particles-1), np.arange(num_samples)).astype(np.intp)
        leaf_map_all = np.empty((num_particles-1, num_samples), dtype=np.intp)
        cdef SIZE_t[::1] samples
        cdef DTYPE_t[::1] Xf
        cdef SIZE_t[::1] leaf_map
        
        # Initial capacity
        cdef int init_capacity

        # Parameters
        cdef SIZE_t max_depth = self.max_depth
        
        cdef SIZE_t start
        cdef SIZE_t split
        cdef SIZE_t end
        cdef SIZE_t depth
        cdef SIZE_t idx
        cdef SIZE_t parent
        cdef bint is_left
        cdef SIZE_t node_id
        
        cdef stack[StackRecord] expansion_nodes
        cdef StackRecord stack_record
        
        cdef double impurity = 0
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0

        # make a tree with a single leaf node
        # tree should be empty?

        if cur_tree.max_depth <= 10:
            init_capacity = (2 ** (cur_tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        classes = np.array([1], dtype=np.intp)

        # the queues can be reused

        particles = []
        for i in range(num_particles-1):
            particles.append((Tree(num_features, classes, 1),
                              SimpleQueue(),
                              samples_all[i],
                              Xf_all[i],
                              leaf_map_all[i]))
        particles = np.array(particles, dtype=object)

        weights = np.empty(num_particles)
        
        set_values_tree(cur_tree,
                        cur_leaf_map,
                        response)

        weights[0] = marginal_loglikelihood_tree(cur_tree,
                                                 sigmasq,
                                                 mu_prior_mean,
                                                 mu_prior_var)

        # initialize the particles
        
        cdef SIZE_t[::1] particle_indices = self.particle_indices
        cdef SIZE_t particle_idx

        cdef Tree tree
        for particle_idx in particle_indices:

            (tree,
             queue,
             samples,
             Xf,
             leaf_map) = particles[particle_idx-1]

            tree._resize(init_capacity)
            
            node_id = tree._add_node(_TREE_UNDEFINED,
                                     0,
                                     True,
                                     _TREE_UNDEFINED,
                                     _TREE_UNDEFINED,
                                     0,
                                     num_samples,
                                     num_samples)
    
            # each particle has a stack of expansion nodes

            queue.put(Record(node_id, 0, 0, num_samples))

            # set the value in case this tree will just be a root
            # the Gibbs step for mu will later use this 
            tree.value[0] = np.mean(response)
            weights[particle_idx] = marginal_loglikelihood(response,
                                                           None,
                                                           1,
                                                           sigmasq,
                                                           mu_prior_mean,
                                                           mu_prior_var)
        cdef SIZE_t stage = 0
        
        any_expansion_nodes = True
        
        while any_expansion_nodes:

            for particle_idx in particle_indices:

                (tree,
                 queue,
                 samples,
                 Xf,
                 leaf_map) = particles[particle_idx-1]

                if queue.qsize() > 0:
                    record = queue.get_nowait()

                    node_id = record.node_id
                    depth = record.depth
                    start = record.start
                    end = record.end

                    # try to split

                    if ((start < end-1) and 
                        (self.random_.random() < self.split_prob(depth))):
                        feature = int(self.random_.random() * num_features)

                        for idx in range(start, end):
                            Xf[idx] = X[samples[idx], feature]
                        sort(&Xf[start], &samples[start], end - start)

                        split = self.random_.randint(start, end-1)
                        threshold = Xf[split]

                        # setup the right node

                        right_id = tree._add_node(node_id, 
                                                  0,  # is_left
                                                  1,  # is_leaf
                                                  _TREE_UNDEFINED,
                                                  _TREE_UNDEFINED, 
                                                  impurity, 
                                                  end-split,
                                                  end-split)

                        # add it to stack as a candidate to split

                        queue.put(Record(right_id, depth+1, split+1, end))

                        # repeat for left node

                        left_id = tree._add_node(node_id, 
                                                 1,  # is_left
                                                 1,  # is_leaf
                                                 _TREE_UNDEFINED,
                                                 _TREE_UNDEFINED, 
                                                 impurity, 
                                                 split-start,
                                                 split-start)

                        queue.put(Record(left_id, depth+1, start, split+1))

                        # update current leaf to a split node

                        tree.nodes[node_id].feature = feature
                        tree.nodes[node_id].threshold = threshold
                        tree.nodes[node_id].left_child = left_id
                        tree.nodes[node_id].right_child = right_id

                        # increment the loglikelihood

                        increment, mean_L, mean_R = incremental_loglikelihood(response,
                                                                              samples,
                                                                              start,
                                                                              split,
                                                                              end,
                                                                              self.sigmasq,
                                                                              self.mu_prior_mean,
                                                                              self.mu_prior_var)
                        weights[particle_idx] += increment

                        # set the values in the nodes to the response mean
                        # will be reset later in a Gibbs step
                        # which uses this value

                        tree.value[left_id] = mean_L
                        tree.value[right_id] = mean_R

                        if depth > max_depth_seen:
                            max_depth_seen = depth

                    else:
                        # the node will not be split again
                        # update the leaf_map
                        # this is used to be able to efficiently
                        # compute marginal likelihood when the response is changed

                        pass
                        # for idx in range(start, end):
                        #     leaf_map[samples[idx]] = node_id

                    if rc >= 0:
                        rc = tree._resize_c(tree.node_count)

                    if rc >= 0:
                        tree.max_depth = max_depth_seen

            # line 13 of Algorithm 2 of Lakshminarayanan
            W_t, probs = _normalize(weights)

            # line 14-15 of Algorithm 2 of Lakshminarayanan
            # Resample all but first particle
            re_n_w = probs[1:] / probs[1:].sum()
            new_indices = self.random_.choice(particle_indices,
                                              size=num_particles-1,
                                              p=re_n_w)
            particles[:] = particles[new_indices-1]

            # Set the new weights
            weights[:] = 0

            # keep track of how many stages this took
            stage += 1
            if stage >= self.max_stages:
                break

            any_expansion_nodes = np.any([p[1].qsize() > 0 for p in particles]) # p[1] is the queue

        if rc == -1:
            raise MemoryError()

        new_idx = self.random_.choice(num_particles, p=probs)
        if new_idx > 0:
            new_tree, _, _, _, new_leaf_map = particles[new_idx-1]
            new_leaf_map = new_tree.apply(self.X)
        else:
            new_tree, new_leaf_map = cur_tree, cur_leaf_map

        sample_values_tree(new_tree,
                           self.random_,
                           self.sigmasq,
                           self.mu_prior_mean,
                           self.mu_prior_var)
                           
        cdef DTYPE_t[::1] predict_train = np.empty(response.shape[0], dtype=self.X.dtype)

        fill_values_tree(new_tree,
                         new_leaf_map,
                         predict_train)

        return new_tree, new_leaf_map, predict_train


#### Likelihood calculations and sampling for Regression version

cdef set_values_tree(Tree tree,
                     SIZE_t[::1] leaf_map,
                     DTYPE_t[::1] response):

    cdef SIZE_t node_count = tree.node_count
    cdef SIZE_t leaf_id
    cdef SIZE_t resp_idx

    cdef DTYPE_t[::1] response_sum = np.zeros(node_count, dtype=np.float32)
    cdef SIZE_t[::1] n_node_samples = np.zeros(node_count, dtype=np.intp)
    
    for resp_idx in range(response.shape[0]):
        r = response[resp_idx]
        response_sum[leaf_map[resp_idx]] += r
        n_node_samples[leaf_map[resp_idx]] += 1                                      

    for leaf_id in range(node_count):
        if tree.nodes[leaf_id].left_child == _TREE_LEAF: # we are in a leaf
            n_node_ = n_node_samples[leaf_id]
            tree.value[leaf_id] = response_sum[leaf_id] / n_node_
            if n_node_ == 0:
                print(leaf_id, tree.n_node_samples)
                print(tree.feature)
                print(tree.threshold)
                print(response_sum[leaf_id])
                raise ValueError('empty leaf')

cdef fill_values_tree(Tree tree,
                      SIZE_t[::1] leaf_map,
                      DTYPE_t[::1] output):

    cdef SIZE_t node_count = tree.node_count
    cdef SIZE_t leaf_id
    cdef SIZE_t resp_idx

    cdef DTYPE_t[::1] output_sum = np.zeros(node_count, dtype=np.float32)

    for resp_idx in range(output.shape[0]):
        output[resp_idx] = tree.value[leaf_map[resp_idx]]

cdef sample_values_tree(Tree tree,
                        random_state,
                        float sigmasq,
                        float mu_prior_mean,
                        float mu_prior_var):

    # we only update the value in the nodes
    # assumes that the current values in the nodes are the
    # sums of responses in that node
    
    random_state = check_random_state(random_state)
    cdef float response_mean
    
    for leaf_id in range(tree.node_count):
        if tree.nodes[leaf_id].left_child == _TREE_LEAF: # we are in a leaf
            n_node_samples = tree.n_node_samples[leaf_id]
            if n_node_samples > 0:
                response_mean = float(tree.value[leaf_id])
                quad = n_node_samples / sigmasq + 1 / mu_prior_var
                linear = response_mean / sigmasq + mu_prior_mean / mu_prior_var

                mean = linear / quad
                std = 1. / sqrt(quad)
                tree.value[leaf_id] = random_state.normal() * std + mean
            else:
                tree.value[leaf_id] = random_state.normal() * sqrt(mu_prior_var) + mu_prior_mean

cpdef incremental_loglikelihood(DTYPE_t[::1] response,
                                SIZE_t[::1] samples,
                                SIZE_t start,
                                SIZE_t split, 
                                SIZE_t end,
                                float sigmasq,
                                float mu_prior_mean,
                                float mu_prior_var):

    cdef float sum_L = 0
    cdef float sum_R = 0
    cdef SIZE_t n_L = split + 1 - start
    cdef SIZE_t n_R = end - (split + 1)
    cdef SIZE_t idx

    # left end point of split is included in the left node -- note the 1 when using split
    for idx in range(start, split+1):
        sum_L += response[samples[idx]]

    for idx in range(split+1, end):
        sum_R += response[samples[idx]]

    cdef sum_f = sum_L + sum_R
    
    # for idx_L

    cdef float sigmasq_bar_L = 1 / (n_L / sigmasq + 1 / mu_prior_var)
    cdef float mu_bar_L = (sum_L / sigmasq + mu_prior_mean / mu_prior_var) * sigmasq_bar_L

    cdef float logL_L = (0.5 * np.log(sigmasq_bar_L / mu_prior_var) +
                         0.5 * (mu_bar_L**2 / sigmasq_bar_L))
    logL_L -= 0.5 * mu_prior_mean**2 / mu_prior_var
                
    # for idx_R

    cdef float sigmasq_bar_R = 1 / (n_R / sigmasq + 1 / mu_prior_var)
    cdef float mu_bar_R = (sum_R / sigmasq + mu_prior_mean / mu_prior_var) * sigmasq_bar_R

    cdef float logL_R = (0.5 * np.log(sigmasq_bar_R / mu_prior_var) +
                         0.5 * (mu_bar_R**2 / sigmasq_bar_R))
    logL_R -= 0.5 * mu_prior_mean**2 / mu_prior_var
                
    # for full: union of idx_L and idx_R

    cdef float sigmasq_bar_f = 1 / ((n_L + n_R) / sigmasq + 1 / mu_prior_var)
    cdef float mu_bar_f = (sum_f / sigmasq + mu_prior_mean / mu_prior_var) * sigmasq_bar_f

    cdef float logL_f = (0.5 * np.log(sigmasq_bar_f / mu_prior_var) +
            0.5 * (mu_bar_f**2 / sigmasq_bar_f))
    logL_f -= 0.5 * mu_prior_mean**2 / mu_prior_var

    return logL_L + logL_R - logL_f, sum_L / n_L, sum_R / n_R

cdef float marginal_loglikelihood(DTYPE_t[::1] response,
                                  SIZE_t[::1] leaf_map,
                                  int node_count,
                                  float sigmasq,
                                  float mu_prior_mean,
                                  float mu_prior_var,
                                  bint incremental=False):
    
    cdef SIZE_t node_idx
    cdef SIZE_t resp_idx

    cdef DTYPE_t[::1] response_sum = np.zeros(node_count, dtype=np.float32)
    cdef SIZE_t[::1] n_node_samples = np.zeros(node_count, dtype=np.intp)

    cdef float sigmasq_bar
    cdef float mu_bar
    cdef float responsesq_sum = 0

    if leaf_map is not None:
        for resp_idx in range(response.shape[0]):
            r = response[resp_idx]
            response_sum[leaf_map[resp_idx]] += r
            n_node_samples[leaf_map[resp_idx]] += 1
            if not incremental:
                responsesq_sum += r*r
    else:
        n_node_samples[0] = response.shape[0]
        response_sum[0] = np.sum(response)
        if not incremental:
            for resp_idx in range(response.shape[0]):
                r = response[resp_idx]
                responsesq_sum += r*r

    cdef float logL = 0
    for node_idx in range(node_count):
        if n_node_samples[node_idx] > 0:
            sigmasq_bar = 1 / (n_node_samples[node_idx] / sigmasq + 1 / mu_prior_var)
            mu_bar = (response_sum[node_idx] / sigmasq + mu_prior_mean / mu_prior_var) * sigmasq_bar

            logL += (0.5 * ln(sigmasq_bar / mu_prior_var) +
                     0.5 * (mu_bar**2 / sigmasq_bar))
            logL -= 0.5 * mu_prior_mean**2 / mu_prior_var

    if not incremental:
        logL -= response.shape[0] * 0.5 * ln(sigmasq)
        logL -= 0.5 * responsesq_sum / sigmasq
                
    return logL

cdef float marginal_loglikelihood_tree(Tree tree,
                                       float sigmasq,
                                       float mu_prior_mean,
                                       float mu_prior_var):

    # NOTE: this does not include the (response**2).sum() and log sigmasq terms!
    # assumes that nodes have value that is a given response summed over
    # the leaves
    
    cdef SIZE_t node_idx
    cdef SIZE_t resp_idx

    cdef float response_mean
    cdef SIZE_t n_node_samples
    cdef float sigmasq_bar
    cdef float mu_bar

    cdef float logL = 0
    for leaf_id in range(tree.node_count):
        if tree.nodes[leaf_id].left_child == _TREE_LEAF: # we are in a leaf
            response_mean = tree.value[leaf_id]
            n_node_samples = tree.n_node_samples[leaf_id]
            if n_node_samples > 0:
                sigmasq_bar = 1 / (n_node_samples / sigmasq + 1 / mu_prior_var)
                mu_bar = (response_mean * n_node_samples / sigmasq + mu_prior_mean / mu_prior_var) * sigmasq_bar

                logL += (0.5 * ln(sigmasq_bar / mu_prior_var) +
                         0.5 * (mu_bar**2 / sigmasq_bar))
                logL -= 0.5 * mu_prior_mean**2 / mu_prior_var

    return logL


cdef inline double log(double x) nogil:
    return ln(x) / ln(2.0)

# Sort n-element arrays pointed to by Xf and samples, simultaneously,
# by the values in Xf. Algorithm: Introsort (Musser, SP&E, 1997).
cdef inline void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    if n == 0:
      return
    cdef int maxd = 2 * <int>log(n)
    introsort(Xf, samples, n, maxd)


cdef inline void swap(DTYPE_t* Xf, SIZE_t* samples,
        SIZE_t i, SIZE_t j) nogil:
    # Helper for sort
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]


cdef inline DTYPE_t median3(DTYPE_t* Xf, SIZE_t n) nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef DTYPE_t a = Xf[0], b = Xf[n // 2], c = Xf[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


# Introsort with median of 3 pivot selection and 3-way partition function
# (robust to repeated elements, e.g. lots of zero features).
cdef void introsort(DTYPE_t* Xf, SIZE_t *samples,
                    SIZE_t n, int maxd) nogil:
    cdef DTYPE_t pivot
    cdef SIZE_t i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(Xf, samples, n)
            return
        maxd -= 1

        pivot = median3(Xf, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if Xf[i] < pivot:
                swap(Xf, samples, i, l)
                i += 1
                l += 1
            elif Xf[i] > pivot:
                r -= 1
                swap(Xf, samples, i, r)
            else:
                i += 1

        introsort(Xf, samples, l, maxd)
        Xf += r
        samples += r
        n -= r
        
cdef inline void sift_down(DTYPE_t* Xf, SIZE_t* samples,
                           SIZE_t start, SIZE_t end) nogil:
    # Restore heap order in Xf[start:end] by moving the max element to start.
    cdef SIZE_t child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and Xf[maxind] < Xf[child]:
            maxind = child
        if child + 1 < end and Xf[maxind] < Xf[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(Xf, samples, root, maxind)
            root = maxind


cdef void heapsort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef SIZE_t start, end

    # heapify
    start = (n - 2) // 2
    end = n
    while True:
        sift_down(Xf, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(Xf, samples, 0, end)
        sift_down(Xf, samples, 0, end)
        end = end - 1

#### Util functions

def _normalize(weights):
    """
    Use logsumexp trick to get W_t and softmax to get normalized_weights
    """

    log_w = weights
    log_w_max = log_w.max()
    log_w_ = log_w - log_w_max
    w_ = np.exp(log_w_)
    w_sum = w_.sum()
    W_t = log_w_max + ln(w_sum) - ln(log_w.shape[0])
    normalized_weights = w_ / w_sum
    # stabilize weights to avoid assigning exactly zero probability to a particle
    normalized_weights += 1e-12
    return W_t, normalized_weights

