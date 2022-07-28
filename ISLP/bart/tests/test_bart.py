import numpy as np

from ISLP.bart import BART
from sklearn.base import clone

def test_bart():
    # a smoke test
    
    n, p = 1000, 50

    X = np.random.standard_normal((n,p))
    X2 = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)*5

    B = BART()
    B.fit(X, Y)
    print(B.predict(X2))

    clone(B)

    return B

if __name__ == "__main__":

    test_bart()
