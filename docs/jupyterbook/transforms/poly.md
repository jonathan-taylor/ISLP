---
jupytext:
  formats: source/transforms///ipynb,jupyterbook/transforms///md:myst,jupyterbook/transforms///ipynb
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

# Polynomial features

The modelling tools included in `ISLP` allow for
construction of orthogonal polynomials of features.

Force rebuild

```{code-cell} ipython3
import numpy as np
from ISLP import load_data
from ISLP.models import ModelSpec, poly
```

```{code-cell} ipython3
Carseats = load_data('Carseats')
Carseats.columns
```

Let's make a term representing a quartic effect for `Population`.

```{code-cell} ipython3
quartic = poly('Population', 4)
```

The object `quartic` does not refer to any data yet, it must be included in a `ModelSpec` object
and fit using the `fit` method.

```{code-cell} ipython3
design = ModelSpec([quartic], intercept=False)
ISLP_features = design.fit_transform(Carseats)
ISLP_features.columns
```

## Compare to `R`

We can compare our polynomials to a similar function in `R`

```{code-cell} ipython3
%load_ext rpy2.ipython
```

We'll recompute these features using `poly` in `R`.

```{code-cell} ipython3
%%R -i Carseats -o R_features
R_features = poly(Carseats$Population, 4)
```

```{code-cell} ipython3
np.linalg.norm(ISLP_features - R_features)
```

## Underlying model

If we look at `quartic`, we see it is a `Variable`, i.e. it can be used to produce a set of columns
in a design matrix when it is a term used in creating the `ModelSpec`.

Its encoder is `Poly(degree=4)`. This is a special `sklearn` transform that expects a single column
in its `fit()` method and constructs a matrix of corresponding orthogonal polynomials.

The spline helpers `ns` and `bs` as well as `pca` follow a similar structure.

```{code-cell} ipython3
quartic
```

## Raw polynomials

One can compute raw polynomials (which results in a less well-conditioned design matrix) of course.

```{code-cell} ipython3
quartic_raw = poly('Population', degree=4, raw=True)
```

Let's compare the features again.

```{code-cell} ipython3
design = ModelSpec([quartic_raw], intercept=False)
raw_features = design.fit_transform(Carseats)
```

```{code-cell} ipython3
%%R -i Carseats -o R_features
R_features = poly(Carseats$Population, 4, raw=TRUE)
```

```{code-cell} ipython3
np.linalg.norm(raw_features - R_features)
```

## Intercept

Looking at `py_features` we see it contains columns: `[Population**i for i in range(1, 4)]`. That is, 
it doesn't contain an intercept, the order 0 term. This can be include with `intercept=True`

```{code-cell} ipython3
quartic_int = poly('Population', degree=4, raw=True, intercept=True)
design = ModelSpec([quartic_int], intercept=False)
intercept_features = design.fit_transform(Carseats)
```

```{code-cell} ipython3
np.linalg.norm(intercept_features.iloc[:,1:] - R_features)
```
