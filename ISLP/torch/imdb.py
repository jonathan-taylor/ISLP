"""

Objects helpful in analysis of IMDB data from `keras`. Constructs
a lookup table that is usable directly with `keras.datasets.imdb` data.

"""

from os.path import join as pjoin
import numpy as np
import torch
from torch.utils.data import TensorDataset
from scipy.sparse import load_npz
from pkg_resources import resource_filename
from pickle import load as load_pickle

lookup_file = resource_filename('ISLP', pjoin('data', 'IMDB_word_index.pkl'))
lookup = load_pickle(open(lookup_file, 'rb'))

def load_sparse(validation=2000,
                random_state=0):
    """
    Load IMDB features from ISLP package in `scipy.sparse.coo_matrix` format.
    """
    X_test = load_npz(resource_filename('ISLP', pjoin('data', 'IMDB_X_test.npz')))
    X_train = load_npz(resource_filename('ISLP', pjoin('data', 'IMDB_X_train.npz')))

    Y_test = np.load(resource_filename('ISLP', pjoin('data', 'IMDB_Y_test.npy')))
    Y_train = np.load(resource_filename('ISLP', pjoin('data', 'IMDB_Y_train.npy')))

    np.random.seed(random_state)
    mask = np.zeros(X_train.shape[0], bool)
    mask[:-int(validation)] = 1
    np.random.shuffle(mask)
    
    X_train_tmp, Y_train_tmp = X_train, Y_train
    X_train, Y_train = X_train_tmp[mask], Y_train_tmp[mask]
    X_valid, Y_valid = X_train_tmp[~mask], Y_train_tmp[~mask]

    return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)

def load_tensor():
    """
    Load IMDB features from ISLP package in `torch` sparse tensor format.
    """

    test_idx = torch.load(resource_filename('ISLP', pjoin('data', 'IMDB_X_test_idx.tensor')))
    test_vals = torch.load(resource_filename('ISLP', pjoin('data', 'IMDB_X_test_values.tensor')))
    test_size = torch.load(resource_filename('ISLP', pjoin('data', 'IMDB_X_test_size.tensor')))

    X_test = torch.sparse_coo_tensor(test_idx, test_vals, test_size).coalesce()
    
    train_idx = torch.load(resource_filename('ISLP', pjoin('data', 'IMDB_X_train_idx.tensor')))
    train_vals = torch.load(resource_filename('ISLP', pjoin('data', 'IMDB_X_train_values.tensor')))
    train_size = torch.load(resource_filename('ISLP', pjoin('data', 'IMDB_X_train_size.tensor')))

    X_train = torch.sparse_coo_tensor(train_idx, train_vals, train_size).coalesce()

    Y_test = np.load(resource_filename('ISLP', pjoin('data', 'IMDB_Y_test.npy')))
    Y_train = np.load(resource_filename('ISLP', pjoin('data', 'IMDB_Y_train.npy')))

    Y_test = torch.tensor(Y_test)
    Y_train = torch.tensor(Y_train)
    
    return (TensorDataset(X_train, Y_train),
            TensorDataset(X_test, Y_test))

def load_sequential():
    """
    Load IMDB features from ISLP package in `torch` sparse tensor format.
    """

    S_train = torch.load(resource_filename('ISLP', pjoin('data', 'IMDB_S_train.tensor')))
    S_test = torch.load(resource_filename('ISLP', pjoin('data', 'IMDB_S_test.tensor')))
    
    Y_train = np.load(resource_filename('ISLP', pjoin('data', 'IMDB_Y_train.npy')))
    Y_test = np.load(resource_filename('ISLP', pjoin('data', 'IMDB_Y_test.npy')))

    Y_train = torch.tensor(Y_train)
    Y_test = torch.tensor(Y_test)
    
    return (TensorDataset(S_train, Y_train),
            TensorDataset(S_test, Y_test))
