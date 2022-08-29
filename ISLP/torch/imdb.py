"""

Objects helpful in analysis of IMDB data from `keras`. Constructs
a lookup table that is usable directly with `keras.datasets.imdb` data.

"""
import os
from os.path import join as pjoin
from hashlib import md5

import numpy as np
import torch
from torch.utils.data import TensorDataset
from scipy.sparse import load_npz
from pkg_resources import resource_filename
from pickle import load as load_pickle
import urllib

urlbase = 'https://hastie.su.domains/ISLP/Data/IMDB/'

md5sums = {'IMDB_X_train_size.tensor': '859a90e4c8617925fdf58d67c15d063b',
           'IMDB_X_test_idx.tensor': '196829061c910afab8979822f2bdad5e',
           'IMDB_Y_test.npy': 'bedbed694970384ebf48088dfee80d51',
           'IMDB_Y_train.npy': '66bbcf3b4d43d2ddbafc03c2c5fbaab5',
           'IMDB_X_train_values.tensor': '67da83ef8c8a53f7da2d28ef5b2ecc9f',
           'IMDB_X_test_values.tensor': '4abfa952fe9466abaaf3ab7a8b140765',
           'IMDB_S_test.tensor': '02669d946d63aeede9b71abf3b8e811d',
           'IMDB_X_test_size.tensor': '859a90e4c8617925fdf58d67c15d063b',
           'IMDB_X_test.npz': 'd914c62cc0a3862067eea3cce955ea2b',
           'IMDB_S_train.tensor': '0c00ebbb1c050141bbfe90b19cf173dd',
           'IMDB_word_index.pkl': '5fa514f2ee6e3ea50a07e84711c42bbd',
           'IMDB_X_train.npz': '9d19e42410ca9264bd2c549122a842fa',
           'IMDB_X_train_idx.tensor': '47368c12728797527d6a6e8217f3da64'}


def _get_imdb(imdb_file,
              outdir,
              urlbase='https://hastie.su.domains/ISLP/Data/IMDB/'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not os.path.exists(pjoin(outdir, imdb_file)):
        # try to retrieve file
        if imdb_file not in md5sums.keys():
            raise ValueError(f'file "{imdb_file}" not part of IMDB dataset')
        with urllib.request.urlopen(pjoin(urlbase, imdb_file)) as response:
            print(f'Retrieving "{imdb_file}" from "{urlbase}".')
            raw_data = response.read()

        open(pjoin(outdir, imdb_file), 'wb').write(raw_data)

        if not _check_md5sum(imdb_file,
                             pjoin(outdir, imdb_file)):
            raise ValueError('md5 sum does not match, file possibly corrupted')

    return pjoin(outdir, imdb_file)
    
def _check_md5sum(imdb_file,
                  filename):
    
    with open(filename, 'rb') as reader:
        raw_data = reader.read()
        # check md5 sum
        md5_hash = md5()
        md5_hash.update(raw_data)
    return md5_hash.hexdigest() == md5sums[imdb_file]

def load_lookup(root='.'):
    lookup_file = _get_imdb('IMDB_word_index.pkl',
                            root)
    return load_pickle(open(lookup_file, 'rb'))

def load_sparse(root='.',
                validation=2000,
                random_state=0):
    """
    Load IMDB features from ISLP package in `scipy.sparse.coo_matrix` format.
    """

    # retrieve necessary files

    X_test, X_train = [load_npz(_get_imdb(f'IMDB_{r}', root))
                       for r in ['X_test.npz',
                                 'X_train.npz']]
    Y_test, Y_train = [np.load(_get_imdb(f'IMDB_{r}', root))
                       for r in ['Y_test.npy',
                                 'Y_train.npy']]
    
    rng = np.random.default_rng(random_state)
    mask = np.zeros(X_train.shape[0], bool)
    mask[:-int(validation)] = 1
    rng.shuffle(mask)
    
    X_train_tmp, Y_train_tmp = X_train, Y_train
    X_train, Y_train = X_train_tmp[mask], Y_train_tmp[mask]
    X_valid, Y_valid = X_train_tmp[~mask], Y_train_tmp[~mask]

    return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)

def load_tensor(root='.'):
    """
    Load IMDB features from ISLP package in `torch` sparse tensor format.
    """

    (test_idx,
     test_vals,
     test_size) = [torch.load(_get_imdb(f'IMDB_{r}', root))
                   for r in ['X_test_idx.tensor',
                             'X_test_values.tensor',
                             'X_test_size.tensor']]

    X_test = torch.sparse_coo_tensor(test_idx, test_vals, test_size).coalesce()
    
    (train_idx,
     train_vals,
     train_size) = [torch.load(_get_imdb(f'IMDB_{r}', root))
                   for r in ['X_train_idx.tensor',
                             'X_train_values.tensor',
                             'X_train_size.tensor']]

    X_train = torch.sparse_coo_tensor(train_idx, train_vals, train_size).coalesce()

    Y_test, Y_train = [np.load(_get_imdb(f'IMDB_{r}', root))
                       for r in ['Y_test.npy',
                                 'Y_train.npy']]
    Y_test = torch.tensor(Y_test)
    Y_train = torch.tensor(Y_train)
    
    return (TensorDataset(X_train, Y_train),
            TensorDataset(X_test, Y_test))

def load_sequential(root='.'):
    """
    Load IMDB features from ISLP package in `torch` sparse tensor format.
    """

    (S_train,
     S_test) = [torch.load(_get_imdb(f'IMDB_{r}', root))
                   for r in ['S_train.tensor',
                             'S_test.tensor',]]
    
    Y_test, Y_train = [np.load(_get_imdb(f'IMDB_{r}', root))
                       for r in ['Y_test.npy',
                                 'Y_train.npy']]
    Y_train = torch.tensor(Y_train)
    Y_test = torch.tensor(Y_test)
    
    return (TensorDataset(S_train, Y_train),
            TensorDataset(S_test, Y_test))
