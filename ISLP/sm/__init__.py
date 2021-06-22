import numpy as np, pandas as pd

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

    result = np.zeros((X.shape[0], degree))
    powX = np.power.outer(X.values, np.arange(1, degree+1))
    powX -= powX.mean(0)
    result[:,0] = powX[:,0] / np.linalg.norm(powX[:,0])

    for i in range(1, degree):
        result[:,i] = powX[:,i]
        for j in range(i):
            result[:,i] -= (result[:,i] * result[:,j]).sum() * result[:,j]
        result[:,i] /= np.linalg.norm(result[:,i])
    df = pd.DataFrame(result, columns=['poly(%s, %d)' % (X.name, degree) 
                                       for degree in range(1, degree+1)])
    return df
