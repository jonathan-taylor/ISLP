---
jupytext:
  cell_metadata_filter: -all
  formats: notebooks///ipynb,source///md:myst
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: islp_test
  language: python
  name: islp_test
---

# Creating a clean IMDB dataset

Running this example requires `keras`. Use `pip install keras` to install if necessary.

```{code-cell}
import pickle
```

```{code-cell}
import numpy as np
from scipy.sparse import coo_matrix, save_npz
import torch
```

```{code-cell}
from keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

```{code-cell}
# the 3 is for three terms: <START> <UNK> <UNUSED> 
num_words = 10000+3
((S_train, Y_train), 
 (S_test, Y_test)) = imdb.load_data(num_words=num_words)
```

```{code-cell}
Y_train = Y_train.astype(np.float32)
Y_test = Y_test.astype(np.float32)
```

```{code-cell}
def one_hot(sequences, ncol):
    idx, vals = [], []
    for i, s in enumerate(sequences):
        idx.extend({(i,v):1 for v in s}.keys())
    idx = np.array(idx).T
    vals = np.ones(idx.shape[1], dtype=np.float32)
    tens = torch.sparse_coo_tensor(indices=idx,
                                   values=vals,
                                   size=(len(sequences), ncol))
    return tens.coalesce()
```

```{code-cell}
X_train, L_train = one_hot(S_train, num_words), Y_train
X_test = one_hot(S_test, num_words)
```

```{code-cell}
def convert_sparse_tensor(X):
    idx = np.asarray(X.indices())
    vals = np.asarray(X.values())
    return coo_matrix((vals,
                      (idx[0],
                       idx[1])),
                      shape=X.shape).tocsr()
```

```{code-cell}
X_train_s = convert_sparse_tensor(X_train)
X_test_s = convert_sparse_tensor(X_test)
```

```{code-cell}
X_train_d = torch.tensor(X_train_s.todense())
X_test_d = torch.tensor(X_test_s.todense())
```

```{code-cell}
torch.save(X_train_d, 'IMDB_X_train.tensor')
torch.save(X_test_d, 'IMDB_X_test.tensor')
```

save the sparse matrices

```{code-cell}
save_npz('IMDB_X_test.npz', X_test_s)
save_npz('IMDB_X_train.npz', X_train_s)
```

```{code-cell}
np.save('IMDB_Y_test.npy', Y_test)
np.save('IMDB_Y_train.npy', L_train)
```

save and pickle the word index

```{code-cell}
word_index = imdb.get_word_index()
lookup = {(i+3):w for w, i in word_index.items()}
lookup[0] = "<PAD>"
lookup[1] = "<START>"
lookup[2] = "<UNK>"
lookup[4] = "<UNUSED>"
```

```{code-cell}
pickle.dump(lookup, open('IMDB_word_index.pkl', 'bw'))
```

create the padded representations

```{code-cell}
(S_train,
 S_test) = [torch.tensor(pad_sequences(S, maxlen=500, value=0))
            for S in [S_train,
                      S_test]]
```

```{code-cell}
torch.save(S_train, 'IMDB_S_train.tensor')
torch.save(S_test, 'IMDB_S_test.tensor')
```
