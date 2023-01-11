"""

Functions helpful for LASSO models. 
In particular, a function for default $\lambda$ values.

"""


import numpy as np

def lam_values(X,
               Y,
               proportion=1e-4,
               nstep=100,
               scaler=None):
    '''
    Compute a grid of lambda values for LASSO path.

    Parameters
    ----------

    X : array-like of shape (n_samples, n_features)
        Design matrix for LASSO problem.

    Y : array-like of shape (n_samples,)
        Response for LASSO problem.

    proportion : float
        Smallest multiple of lambda_max in the sequence.

    nstep : int
        Number of steps on logarithmic scale for the sequence.

    Note
    ----

    Objective here is *sum* of log-likelihood terms not *mean*
    of log-likelihood terms. Values may have to be divided
    by `n_samples`.

    '''

    if scaler is not None:
        X = scaler.transform(X)
    lam_max = np.fabs(X.T.dot(Y - Y.mean())).max()
    return np.exp(np.linspace(0,
                              np.log(proportion),
                              nstep))
