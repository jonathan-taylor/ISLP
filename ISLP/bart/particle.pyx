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
import numpy as np
cimport numpy as cnp
cnp.import_array()

from sklearn.tree._tree cimport TreeBuilder, Tree, SIZE_t, DOUBLE_t, DTYPE_t
from sklearn.utils import check_random_state
from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

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

cdef class SequentialTreeBuilder(TreeBuilder):
    """Build a decision tree sequentially for particle Gibbs BART

    Nodes are added randomly in a sequential fashion, keeping
    track of marginal likelihood.
    """
    cdef SIZE_t max_leaf_nodes

    def __cinit__(self, 
                  SIZE_t max_depth,
                  int random_state,
                  float sigmasq,
                  float mu_prior_mean,
                  float mu_prior_var):

        self.max_depth = max_depth
        self.random_ = check_random_state(random_state)
        self.sigmasq = sigmasq
        self.mu_prior_mean = mu_prior_mean
        self.mu_prior_var = mu_prior_var
        
    cpdef build(self, Tree tree, object X, cnp.ndarray y,
                cnp.ndarray sample_weight=None):
        """Build a decision tree from the training set (X, y)."""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)
        
        cdef DOUBLE_t* sample_weight_ptr = NULL

        # structure to capture splits
        cdef DTYPE_t[::1] Xf = np.empty_like(X[:,0])
        cdef SIZE_t[::1] samples = np.arange(X.shape[0], dtype=np.intp)
        cdef SIZE_t[::1] _apply_train = -np.ones(X.shape[0], dtype=np.intp)
        cdef SIZE_t[::1] _apply_count = np.zeros(X.shape[0], dtype=np.intp)

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
                
                print('split', start, split, end)
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
                
                #print(f'node: {node_id}, right:{right_id}, left:{left_id}, start:{start}, end:{end}, feature:{feature}')
                
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
                                                           self.mu_prior_var)

                if depth > max_depth_seen:
                    max_depth_seen = depth
                                        
            else:
                # the node will not be split again
                # update the _apply_train
                # this is used to be able to efficiently
                # compute marginal likelihood when the response is changed
                
                for idx in range(start, end):
                    _apply_train[samples[idx]] = node_id
                    _apply_count[samples[idx]] += 1

                print('final node', node_id, np.array(samples[start:end]), start, end, tree.nodes[node_id].left_child, tree.nodes[node_id].right_child)
                print(np.array(_apply_train)[samples[start:end]], 'apply_train')
                print(tree.apply(X)[samples[start:end]], 'apply')
                
            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen
        if rc == -1:
            raise MemoryError()

        return tree, loglikelihood, np.asarray(_apply_train)

    cpdef split_prob(self, SIZE_t depth):
        return 0.95 / ((1 + depth) * (1 + depth))

cpdef incremental_loglikelihood(cnp.ndarray response,
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

    cdef float logL_L = (0.5 * ln(sigmasq_bar_L / mu_prior_var) +
                         0.5 * (mu_bar_L**2 / sigmasq_bar_L))
    logL_L -= 0.5 * mu_prior_mean**2 / mu_prior_var
                
    # for idx_R

    cdef float sigmasq_bar_R = 1 / (n_R / sigmasq + 1 / mu_prior_var)
    cdef float mu_bar_R = (sum_R / sigmasq + mu_prior_mean / mu_prior_var) * sigmasq_bar_R

    cdef float logL_R = (0.5 * ln(sigmasq_bar_R / mu_prior_var) +
                         0.5 * (mu_bar_R**2 / sigmasq_bar_R))
    logL_R -= 0.5 * mu_prior_mean**2 / mu_prior_var
                
    # for full: union of idx_L and idx_R

    cdef float sigmasq_bar_f = 1 / ((n_L + n_R) / sigmasq + 1 / mu_prior_var)
    cdef float mu_bar_f = (sum_f / sigmasq + mu_prior_mean / mu_prior_var) * sigmasq_bar_f

    cdef float logL_f = (0.5 * ln(sigmasq_bar_f / mu_prior_var) +
            0.5 * (mu_bar_f**2 / sigmasq_bar_f))
    logL_f -= 0.5 * mu_prior_mean**2 / mu_prior_var

    return logL_L + logL_R - logL_f

cpdef marginal_loglikelihood(cnp.ndarray response,
                             SIZE_t[::1] node_map,
                             int node_count,
                             float sigmasq,
                             float mu_prior_mean,
                             float mu_prior_var,
                             bint incremental=False):
    
    cdef SIZE_t node_idx
    cdef SIZE_t resp_idx
    cdef float logL = 0

    response_sum = np.zeros(node_count, float)
    cdef SIZE_t[::1] n_sum = np.zeros(node_count, dtype=np.intp)

    cdef float sigmasq_bar
    cdef float mu_bar
    cdef float responsesq_sum = 0

    if node_map is not None:
        for resp_idx in range(response.shape[0]):
            r = response[resp_idx]
            response_sum[node_map[resp_idx]] += r
            n_sum[node_map[resp_idx]] += 1
            if not incremental:
                responsesq_sum += r*r
    else:
        n_sum[0] = response.shape[0]
        response_sum[0] = response.sum()
        if not incremental:
            responsesq_sum = (response**2).sum()

    for node_idx in range(node_count):
        sigmasq_bar = 1 / (n_sum[node_idx] / sigmasq + 1 / mu_prior_var)
        mu_bar = (response_sum[node_idx] / sigmasq + mu_prior_mean / mu_prior_var) * sigmasq_bar

        logL += (0.5 * ln(sigmasq_bar / mu_prior_var) +
                 0.5 * (mu_bar**2 / sigmasq_bar))
        logL -= 0.5 * mu_prior_mean**2 / mu_prior_var

    if not incremental:
        logL -= response.shape[0] * 0.5 * ln(sigmasq)
        logL -= 0.5 * responsesq_sum / sigmasq
                
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

        
