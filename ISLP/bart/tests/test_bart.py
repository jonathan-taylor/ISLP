import numpy as np

from ISLP.bart import BART
from sklearn.base import clone

def test_bart():
    # a smoke test
    
    n, p = 100, 20

    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)

    B = BART()
    B.fit(X, Y)
    sum_trees_output = np.zeros(n)
    sum_trees_output, V = B._step(X,
                                   Y,
                                   B.init_log_weight_,
                                   B.init_likelihood_,
                                   sum_trees_output)
    for _ in range(5):
        B._step(X,
                Y,
                B.init_log_weight_,
                B.init_likelihood_,
                sum_trees_output)

    print(V)

    clone(B)

    return B

if __name__ == "__main__":

    test_bart()
