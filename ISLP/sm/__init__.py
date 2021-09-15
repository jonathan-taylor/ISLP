import numpy as np, pandas as pd
from ..sklearn import Poly

def summarize(results):
    """
    Take a fit statsmodels and summarize it
    by returning the usual coefficient estimates,
    their standard errors, the usual test
    statistics and P-values as well as 95%
    confidence intervals

    Based on:

    https://stackoverflow.com/questions/51734180/converting-statsmodels-summary-object-to-pandas-dataframe
    """
    tab = results.summary().tables[1]
    return pd.read_html(tab.as_html(),
                        index_col=0,
                        header=0)[0]

def poly(X, degree):
    """  
    Create columns of design matrix
    for orthogonal polynomial for a given series X
    """

    result = Poly(degree=degree).fit_transform(X)

from .sklearn import sklearn_sm
