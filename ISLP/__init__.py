from os.path import join as pjoin
import pandas as pd
from pkg_resources import resource_filename

# data originally saved via: [sm.datasets.get_rdataset(n, 'ISLR').data.to_csv('../ISLP/data/%s.csv' % n, index=False) for n in ['Carseats', 'College', 'Credit', 'Default', 'Hitters', 'Auto', 'OJ', 'Portfolio', 'Smarket', 'Wage', 'Weekly', 'Caravan']]

def load_data(dataset):
    if dataset not in ['NCI60', 'Khan']:
        filename = resource_filename('ISLP', pjoin('data', '%s.csv' % dataset))
        return pd.read_csv(filename)
    elif dataset == 'NCI60':
        features = resource_filename('ISLP', pjoin('data', 'NCI60data.npy'))
        X = np.load(features)
        labels = resource_filename('ISLP', pjoin('data', 'NCI60labels.csv'))
        Y = pd.read_csv(labels)
        return {'data':X, 'labels':labels}
    else:
        xtest = pd.read_csv(resource_filename('ISLP', pjoin('data', 'Khan_xtest.csv')))
        ytest = pd.read_csv(resource_filename('ISLP', pjoin('data', 'Khan_ytest.csv')))
        xtrain = pd.read_csv(resource_filename('ISLP', pjoin('data', 'Khan_xtrain.csv')))
        ytrain = pd.read_csv(resource_filename('ISLP', pjoin('data', 'Khan_ytrain.csv')))
        return {'xtest':xtest,
                'xtrain':xtrain,
                'ytest':ytest,
                'ytrain':ytrain}
