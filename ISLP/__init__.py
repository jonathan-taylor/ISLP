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
