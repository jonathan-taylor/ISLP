
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
        ytest = ytest['Y']
        
        xtrain = pd.read_csv(resource_filename('ISLP', pjoin('data', 'Khan_xtrain.csv')))
        xtrain = xtrain.rename(columns=dict([('V%d' % d, 'X%d' % d) for d in range(1, len(xtest.columns))]))
        ytrain = pd.read_csv(resource_filename('ISLP', pjoin('data', 'Khan_ytrain.csv')))
        ytrain = ytrain.rename(columns={'x':'Y'})
        ytrain = ytrain['Y']

        return {'xtest':xtest,
                'xtrain':xtrain,
                'ytest':ytest,
                'ytrain':ytrain}
    elif dataset == 'Hitters':
        filename = resource_filename('ISLP', pjoin('data', '%s.csv' % dataset))
        df = pd.read_csv(filename)
        for col in ['League', 'Division', 'NewLeague']:
            df[col] = pd.Categorical(df[col])
        return df
    elif dataset == 'Carseats':
        filename = resource_filename('ISLP', pjoin('data', '%s.csv' % dataset))
        df = pd.read_csv(filename)
        for col in ['ShelveLoc', 'Urban', 'US']:
            df[col] = pd.Categorical(df[col])
        return df
    elif dataset == 'NYSE':
        filename = resource_filename('ISLP', pjoin('data', '%s.csv' % dataset))
        df = pd.read_csv(filename).set_index('date')
        return df
    elif dataset == 'Publication':
        df = pd.read_csv(resource_filename('ISLP', pjoin('data', 'Publication.csv')))
        for col in ['mech']:
            df[col] = pd.Categorical(df[col])
        return df
    elif dataset == 'BrainCancer':
        df = pd.read_csv(resource_filename('ISLP', pjoin('data', 'BrainCancer.csv')))
        for col in ['sex', 'diagnosis', 'loc', 'stereo']:
            df[col] = pd.Categorical(df[col])
        return df
    elif dataset == 'Bikeshare':
        filename = resource_filename('ISLP', pjoin('data', '%s.csv' % dataset))
        df = pd.read_csv(filename)
        df['weathersit'] = pd.Categorical(df['weathersit'], ordered=False)
        # setting order to avoid alphabetical
        df['mnth'] = pd.Categorical(df['mnth'],
                                    ordered=False,
                                    categories=['Jan', 'Feb',
                                                'March', 'April',
                                                'May', 'June',
                                                'July', 'Aug',
                                                'Sept', 'Oct',
                                                'Nov', 'Dec'])
        df['hr'] = pd.Categorical(df['hr'],
                                  ordered=False,
                                  categories=range(24))
        return df
    elif dataset == 'Wage':
        df = pd.read_csv(resource_filename('ISLP', pjoin('data', 'Wage.csv')))
        df['education'] = pd.Categorical(df['education'], ordered=True)
        return df
    else:
        filename = resource_filename('ISLP', pjoin('data', '%s.csv' % dataset))
        return pd.read_csv(filename)

from sklearn.metrics import confusion_matrix as _confusion_matrix

def confusion_table(predicted_labels,
                    true_labels):
    """
    Return a data frame version of confusion 
    matrix with rows given by predicted label
    and columns the truth.
    """

    labels = sorted(np.unique(list(true_labels) +
                              list(predicted_labels)))
    C = _confusion_matrix(true_labels,
                          predicted_labels,
                          labels=labels)
    df = pd.DataFrame(C.T, columns=labels) # swap rows and columns
    df.index = pd.Index(labels, name='Predicted')
    df.columns.name = 'Truth'
    return df
        

from . import _version
__version__ = _version.get_versions()['version']
