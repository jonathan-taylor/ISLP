"""

Objects helpful in analysis of IMDB data from `keras`. Constructs
a lookup table that is usable directly with `keras.datasets.imdb` data.

"""

from keras.datasets import imdb

word_index = imdb.get_word_index()
lookup = {(i+3):w for w, i in word_index.items()}
lookup[0] = "<PAD>"
lookup[1] = "<START>"
lookup[2] = "<UNK>"
lookup[4] = "<UNUSED>"

