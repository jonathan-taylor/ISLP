---
jupytext:
  cell_metadata_filter: -all
  formats: source/helpers///ipynb,jupyterbook/helpers///md:myst,jupyterbook/helpers///ipynb
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: python3
  language: python
  name: python3
---

# Support Vector Machines

This module contains a single function used to help visualize the decision rule of an SVM.

```{code-cell}
import numpy as np
from sklearn.svm import SVC
from ISLP.svm import plot
```

## Make a toy dataset

```{code-cell}
rng = np.random.default_rng(1)
X = rng.normal(size=(100, 5))
X[:40][:,3:5] += 2
Y = np.zeros(X.shape[0])
Y[:40] = 1
```

## Fit an SVM classifier

```{code-cell}
svm = SVC(kernel='linear')
svm.fit(X, Y)
```

```{code-cell}
plot(X, Y, svm)
```

## Slicing through different features

When we generated our data, the real differences ware in the 4th and 5th coordinates.
We can see this by taking a cross-section through the data that includes this coordinate as
one of the axes in the plot.

```{code-cell}
plot(X, Y, svm, features=(3, 4))
```
