import numpy as np
from scipy.optimize import root

def sim_time(linpred,
             cum_hazard,
             rng):
    """
    Simulate a survival time for a 
    cumulative hazard function $H$ with actual hazard

    $$
    H_l(t) = e^l H(t)
    $$
    
    with `l` the linear predictor `linpred` as in a
    Cox proportional hazards model.

    Compute a grid of lambda values for LASSO path.

    Parameters
    ----------

    lin_pred : float
        Linear predictor value.

    cum_hazard : callable
        Cumulative hazard function, takes a single non-negative argument.

    rng : numpy random number generator
    """

    U = rng.uniform()
    B = - np.log(U) /  np.exp(linpred)
    lower, upper = 1, 2

    while True:
        if cum_hazard(lower) > B:
            lower /= 2
        if cum_hazard(upper) < B:
            upper *= 2
        if ((cum_hazard(lower) < B) and
            (cum_hazard(upper) > B)):
            break
    T = root(lambda t: cum_hazard(t) - B,
             lower).x

    return T
