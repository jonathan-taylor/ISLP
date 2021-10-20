import numpy as np

from ISLP.bart import PGBART

def test_bart():

    n, p = 100, 20

    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)

    B = PGBART(X, Y)
    sum_trees_output = np.zeros(n)
    sum_trees_output, V = B.astep(sum_trees_output)
    print(V)

    return B

