---
jupytext:
  cell_metadata_filter: -all
  formats: source///ipynb,jupyterbook///md:myst,jupyterbook///ipynb
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: islp_test
  language: python
  name: islp_test
---

# Creating IMDB dataset from `keras` version

This script details how the `IMDB` data in `ISLP` was constructed.

Running this example requires `keras`. Use `pip install keras` to install if necessary.

```{code-cell} ipython3
import pickle
import numpy as np
from scipy.sparse import coo_matrix, save_npz
import torch
from keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

We first load the data using `keras`, limiting focus to the 10000 most commmon words.

```{code-cell} ipython3
# the 3 is for three terms: <START> <UNK> <UNUSED> 
num_words = 10000+3
((S_train, L_train), 
 (S_test, L_test)) = imdb.load_data(num_words=num_words)
```

The object `S_train` is effectively a list in which each document has been encoded into a sequence of
values from 0 to 10002.

```{code-cell} ipython3
S_train[0][:10]
```

We'll use `np.float32` as that is the common precision used in `torch`.

```{code-cell} ipython3
L_train = L_train.astype(np.float32)
L_test = L_test.astype(np.float32)
```

We will use a one-hot encoding that captures whether or not a given word appears in a given review.

```{code-cell} ipython3
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

```{code-cell} ipython3
X_train = one_hot(S_train, num_words)
X_test = one_hot(S_test, num_words)
```

## Store as sparse tensors

We see later in the lab that the dense representation is faster. Nevertheless,
let's store the one-hot representation as sparse `torch` tensors 
as well as sparse `scipy` matrices.

```{code-cell} ipython3
def convert_sparse_tensor(X):
    idx = np.asarray(X.indices())
    vals = np.asarray(X.values())
    return coo_matrix((vals,
                      (idx[0],
                       idx[1])),
                      shape=X.shape).tocsr()
```

```{code-cell} ipython3
X_train_s = convert_sparse_tensor(X_train)
X_test_s = convert_sparse_tensor(X_test)
```

```{code-cell} ipython3
X_train_d = torch.tensor(X_train_s.todense())
X_test_d = torch.tensor(X_test_s.todense())
```

```{code-cell} ipython3
torch.save(X_train_d, 'IMDB_X_train.tensor')
torch.save(X_test_d, 'IMDB_X_test.tensor')
```

### Save as sparse `scipy` matrices

```{code-cell} ipython3
save_npz('IMDB_X_test.npz', X_test_s)
save_npz('IMDB_X_train.npz', X_train_s)
```

```{code-cell} ipython3
np.save('IMDB_Y_test.npy', L_test)
np.save('IMDB_Y_train.npy', L_train)
```

## Save and pickle the word index

We'll also want to store a lookup table to convert representations such as `S_train[0]` into words

```{code-cell} ipython3
word_index = imdb.get_word_index()
lookup = {(i+3):w for w, i in word_index.items()}
lookup[0] = "<PAD>"
lookup[1] = "<START>"
lookup[2] = "<UNK>"
lookup[4] = "<UNUSED>"
```

Let's look at our first training document:

```{code-cell} ipython3
' '.join([lookup[i] for i in S_train[0][:20]])
```

We save this lookup table so it can be loaded later 

```{code-cell} ipython3
pickle.dump(lookup, open('IMDB_word_index.pkl', 'bw'))
```

## Padded representations

For some of the recurrent models, we'll need sequences of common lengths, padded if necessary.
Here, we pad up to a maximum length of 500, filling the remaining entries with 0.

```{code-cell} ipython3
(S_train,
 S_test) = [torch.tensor(pad_sequences(S, maxlen=500, value=0))
            for S in [S_train,
                      S_test]]
```

Finally, we save these for later use in the deep learning lab.

```{code-cell} ipython3
torch.save(S_train, 'IMDB_S_train.tensor')
torch.save(S_test, 'IMDB_S_test.tensor')
```
