---
jupytext:
  cell_metadata_filter: -all
  formats: notebooks/helpers///ipynb,source/helpers///md:myst
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

# Support Vector Machines

This module contains a single function used to help visualize the decision rule of an SVM.

```{code-cell}
import numpy as np
from sklearn.svm import SVC
from ISLP.svm import plot_svm
```

## Make a toy dataset

```{code-cell}
rng = np.random.default_rng(1)
X = rng.normal(size=(100, 5))
X[:40,4] += 5
Y = np.zeros(X.shape[0])
Y[:40] = 1
```

## Fit an SVM classifier

```{code-cell}
svm = SVC(kernel='linear')
svm.fit(X, Y)
```

```{code-cell}
plot_svm(X, Y, svm)
```

## Slicing through different features

When we generated our data, the real difference was in the 5th coordinate.
We can see this by taking a cross-section through the data that includes this coordinate as
one of the axes in the plot.

```{code-cell}
plot_svm(X, Y, svm, features=(2, 4))
```
