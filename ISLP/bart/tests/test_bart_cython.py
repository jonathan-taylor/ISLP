import numpy as np
from sklearn.tree._tree import Tree

from ISLP.bart.particle import (SequentialTreeBuilder,
                                marginal_loglikelihood,
                                marginal_loglikelihood_tree,
                                sample_values_tree,
                                incremental_loglikelihood)
from ISLP.bart.likelihood import (marginal_loglikelihood as marginal_loglikelihood_py,
                                  incremental_loglikelihood as incremental_loglikelihood_py)

class MyBuilder(SequentialTreeBuilder):

    def split_prob(self, depth):
        return (depth <= 1)

def test_builder(n=100, p=20):
    # make sure that the _apply_train attribute is tracked correctly

    sigmasq, mu_prior_mean, mu_prior_var = 1.5, 0.2, 0.4

    X = np.random.standard_normal((n, p)).astype(np.float32)
    y = np.random.standard_normal(n).astype(np.float32)
    tree = Tree(p, np.array([1]), 1)
    builder = MyBuilder(max_depth=10,
                        num_particles=10,
                        max_stages=5000,
                        random_state=0,
                        sigmasq=sigmasq,
                        mu_prior_mean=mu_prior_mean,
                        mu_prior_var=mu_prior_var)
    _, logL1, leaves_train = builder.build(tree, X, y, np.ones_like(y))

    idx1 = tree.apply(X.astype(np.float32))
    idx2 = leaves_train
    np.testing.assert_allclose(idx1, idx2)

    # compute the loglikelihood a response

    y_new = np.random.standard_normal(y.shape)
    logL2 = marginal_loglikelihood(y,
                                   leaves_train,
                                   tree.node_count,
                                   sigmasq,
                                   mu_prior_mean,
                                   mu_prior_var)

    # the incremental should match the `marginal` approach
    
    assert np.allclose(logL1, logL2)

    for leaf in np.unique(leaves_train):
        sum1 = y[leaves_train == leaf].sum()
        sum2 = tree.value[leaf]

    logL3 = marginal_loglikelihood_tree(tree,
                                        sigmasq,
                                        mu_prior_mean,
                                        mu_prior_var)
    
    # this logL3 doesn't have global terms in it. let's add them

    logL3 -= n * 0.5 * np.log(sigmasq)
    logL3 -= 0.5 * (y**2).sum() / sigmasq

    assert np.fabs((logL1 - logL3) / logL3) < 0.01 

    for leaf in np.unique(leaves_train):
        print(leaf, tree.value[leaf])

    sample_values_tree(tree,
                       0,
                       sigmasq,
                       mu_prior_mean,
                       mu_prior_var)                  

    for leaf in np.unique(leaves_train):
        print(leaf, tree.value[leaf])
    return tree

def test_marginal_loglikelihood(n=40):
    # make sure that marginal_loglikelihood is correct in cython

    y = np.random.standard_normal(n)
    node_map = np.zeros(n, np.intp)

    sigmasq, mu_prior_mean, mu_prior_var = 1.5, 0.2, 0.2

    val_py = marginal_loglikelihood_py(y,
                                       sigmasq,
                                       mu_prior_mean,
                                       mu_prior_var)[0]

    val_cy = marginal_loglikelihood(y,
                                    node_map,
                                    np.unique(node_map).shape[0],
                                    sigmasq,
                                    mu_prior_mean,
                                    mu_prior_var,
                                    incremental=False)

    assert np.allclose(val_py, val_cy)

    val_py = marginal_loglikelihood_py(y,
                                       sigmasq,
                                       mu_prior_mean,
                                       mu_prior_var,
                                       incremental=True)[0]

    val_cy = marginal_loglikelihood(y,
                                    node_map,
                                    np.unique(node_map).shape[0],
                                    sigmasq,
                                    mu_prior_mean,
                                    mu_prior_var,
                                    incremental=True)

    assert np.allclose(val_py, val_cy)

def test_incremental_loglikelihood(n=40):
    # make sure that marginal_loglikelihood is correct in cython

    y = np.random.standard_normal(n)
    node_map = np.zeros(n, np.intp)

    sigmasq, mu_prior_mean, mu_prior_var = 1, 0, 0.2

    samples = np.arange(y.shape[0], dtype=np.intp)
    np.random.shuffle(samples)
    idx_L = [3,4,5]
    idx_R = [6,7,8,9,10]
    start = 3
    split = 5
    end = 11

    val_py = incremental_loglikelihood_py(y,
                                          samples[idx_L],
                                          samples[idx_R],
                                          sigmasq,
                                          mu_prior_mean,
                                          mu_prior_var)[0]

    val_cy = incremental_loglikelihood(y,
                                       samples,
                                       start,
                                       split,
                                       end,
                                       sigmasq,
                                       mu_prior_mean,
                                       mu_prior_var)[0]

    assert np.allclose(val_py, val_cy)

   

