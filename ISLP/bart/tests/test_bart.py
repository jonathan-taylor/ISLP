import numpy as np

from ISLP.bart import BART
from sklearn.base import clone

def test_bart(n=200, p=20):
    # a smoke test
    
    np.random.seed(0)
    X = np.random.standard_normal((n,p))
    X2 = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)*5

    B = BART(ndraw=10, burnin=10)
    B.fit(X, Y.astype(np.float32))
    print(B.predict(X2))

    clone(B)

    return B

if __name__ == "__main__":

    test_bart()
