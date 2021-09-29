import numpy as np, pandas as pd
from ..transforms import Poly
from .model_matrix import ModelMatrix, Column

def summarize(results,
              conf_int=False):
    """
    Take a fit statsmodels and summarize it
    by returning the usual coefficient estimates,
    their standard errors, the usual test
    statistics and P-values as well as 
    (optionally) 95% confidence intervals.

    Based on:

    https://stackoverflow.com/questions/51734180/converting-statsmodels-summary-object-to-pandas-dataframe

    Parameters
    ----------

    results : a results object

    conf_int : bool (optional)
        Include 95% confidence intervals?

    """
    tab = results.summary().tables[1]
    results_table = pd.read_html(tab.as_html(),
                                 index_col=0,
                                 header=0)[0]
    if not conf_int:
        columns = ['coef',
                   'std err',
                   't',
                   'P>|t|']
        return results_table[results_table.columns[:-2]]
    return results_table

def poly(X, degree):
    """  
    Create columns of design matrix
    for orthogonal polynomial for a given series X
    """

    result = Poly(degree=degree).fit_transform(X)

from .sklearn import sklearn_sm
