
from os.path import join as pjoin
import pandas as pd, numpy as np
from pkg_resources import resource_filename

# data originally saved via: [sm.datasets.get_rdataset(n, 'ISLR').data.to_csv('../ISLP/data/%s.csv' % n, index=False) for n in ['Carseats', 'College', 'Credit', 'Default', 'Hitters', 'Auto', 'OJ', 'Portfolio', 'Smarket', 'Wage', 'Weekly', 'Caravan']]

def load_data(dataset):
    if dataset == 'NCI60':
        features = resource_filename('ISLP', pjoin('data', 'NCI60data.npy'))
        X = np.load(features)
        labels = resource_filename('ISLP', pjoin('data', 'NCI60labs.csv'))
        Y = pd.read_csv(labels)
        return {'data':X, 'labels':Y}
    elif dataset == 'Khan':
        xtest = pd.read_csv(resource_filename('ISLP', pjoin('data', 'Khan_xtest.csv')))
        xtest = xtest.rename(columns=dict([('V%d' % d, 'X%d' % d) for d in range(1, len(xtest.columns))]))
        ytest = pd.read_csv(resource_filename('ISLP', pjoin('data', 'Khan_ytest.csv')))
        ytest = ytest.rename(columns={'x':'Y'})

        xtrain = pd.read_csv(resource_filename('ISLP', pjoin('data', 'Khan_xtrain.csv')))
        xtrain = xtrain.rename(columns=dict([('V%d' % d, 'X%d' % d) for d in range(1, len(xtest.columns))]))
        ytrain = pd.read_csv(resource_filename('ISLP', pjoin('data', 'Khan_ytrain.csv')))
        ytrain = ytrain.rename(columns={'x':'Y'})

        return {'xtest':xtest,
                'xtrain':xtrain,
                'ytest':ytest,
                'ytrain':ytrain}
    elif dataset == 'BrainCancer':
        df = pd.read_csv(resource_filename('ISLP', pjoin('data', 'BrainCancer.csv')))
        return df
    else:
        filename = resource_filename('ISLP', pjoin('data', '%s.csv' % dataset))
        return pd.read_csv(filename)

def smarize(results):
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

def smpoly(X, degree):
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
