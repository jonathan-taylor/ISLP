"""

Objects helpful in analysis of IMDB data from `keras`. Constructs
a lookup table that is usable directly with `keras.datasets.imdb` data.

"""

from os.path import join as pjoin
import numpy as np
import torch
from scipy.sparse import load_npz
from pkg_resources import resource_filename
from pickle import load as load_pickle

lookup_file = resource_filename('ISLP', pjoin('data', 'IMDB_word_index.pkl'))
lookup = load_pickle(open(lookup_file, 'rb'))

def load_sparse():
    """
    Load IMDB features from ISLP package in `scipy.sparse.coo_matrix` format.
    """
    X_test = load_npz(resource_filename('ISLP', pjoin('data', 'IMDB_X_test.npz')))
    X_valid = load_npz(resource_filename('ISLP', pjoin('data', 'IMDB_X_valid.npz')))
    X_train = load_npz(resource_filename('ISLP', pjoin('data', 'IMDB_X_train.npz')))

    Y_test = np.load(resource_filename('ISLP', pjoin('data', 'IMDB_Y_test.npy')))
    Y_valid = np.load(resource_filename('ISLP', pjoin('data', 'IMDB_Y_valid.npy')))
    Y_train = np.load(resource_filename('ISLP', pjoin('data', 'IMDB_Y_train.npy')))

    return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)

def load_tensor():
    """
    Load IMDB features from ISLP package in `torch` sparse tensor format.
    """

    test_idx = torch.load(resource_filename('ISLP', pjoin('data', 'IMDB_X_test_idx.tensor')))
    test_vals = torch.load(resource_filename('ISLP', pjoin('data', 'IMDB_X_test_values.tensor')))
    test_size = torch.load(resource_filename('ISLP', pjoin('data', 'IMDB_X_test_size.tensor')))

    X_test = torch.sparse_coo_tensor(test_idx, test_vals, test_size)
    
    train_idx = torch.load(resource_filename('ISLP', pjoin('data', 'IMDB_X_train_idx.tensor')))
    train_vals = torch.load(resource_filename('ISLP', pjoin('data', 'IMDB_X_train_values.tensor')))
    train_size = torch.load(resource_filename('ISLP', pjoin('data', 'IMDB_X_train_size.tensor')))

    X_train = torch.sparse_coo_tensor(train_idx, train_vals, train_size)

    valid_idx = torch.load(resource_filename('ISLP', pjoin('data', 'IMDB_X_valid_idx.tensor')))
    valid_vals = torch.load(resource_filename('ISLP', pjoin('data', 'IMDB_X_valid_values.tensor')))
    valid_size = torch.load(resource_filename('ISLP', pjoin('data', 'IMDB_X_valid_size.tensor')))

    X_valid = torch.sparse_coo_tensor(valid_idx, valid_vals, valid_size)
    
    Y_test = np.load(resource_filename('ISLP', pjoin('data', 'IMDB_Y_test.npy')))
    Y_valid = np.load(resource_filename('ISLP', pjoin('data', 'IMDB_Y_valid.npy')))
    Y_train = np.load(resource_filename('ISLP', pjoin('data', 'IMDB_Y_train.npy')))

    Y_test = torch.tensor(Y_test)
    Y_valid = torch.tensor(Y_valid)
    Y_train = torch.tensor(Y_train)
    
    return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)
