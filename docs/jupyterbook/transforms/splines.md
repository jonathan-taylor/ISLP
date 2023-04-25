---
jupytext:
  formats: source/transforms///ipynb,jupyterbook/transforms///md:myst,jupyterbook/transforms///ipynb
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

# Spline features

The modelling tools included in `ISLP` allow for
construction of spline functions of features.

Force rebuild

```{code-cell} ipython3
import numpy as np
from ISLP import load_data
from ISLP.models import ModelSpec, ns, bs
```

```{code-cell} ipython3
Carseats = load_data('Carseats')
Carseats.columns
```

Let's make a term representing a cubic spline for `Population`. We'll use knots based on the 
deciles.

```{code-cell} ipython3
knots = np.percentile(Carseats['Population'], np.linspace(10, 90, 9))
knots
```

```{code-cell} ipython3
bs_pop = bs('Population', internal_knots=knots, degree=3)
```

The object `bs_pop` does not refer to any data yet, it must be included in a `ModelSpec` object
and fit using the `fit` method.

```{code-cell} ipython3
design = ModelSpec([bs_pop], intercept=False)
py_features = np.asarray(design.fit_transform(Carseats))
```

## Compare to `R`

We can compare our polynomials to a similar function in `R`

```{code-cell} ipython3
%load_ext rpy2.ipython
```

We'll recompute these features using `bs` in `R`. The default knot selection of the
`ISLP` and `R` version are slightly different so we just fix the set of internal knots.

```{code-cell} ipython3
%%R -i Carseats,knots -o R_features
library(splines)
R_features = bs(Carseats$Population, knots=knots, degree=3)
```

```{code-cell} ipython3
np.linalg.norm(py_features - R_features)
```

## Underlying model

As for `poly`, the computation of the B-splines is done by a special `sklearn` transformer.

```{code-cell} ipython3
bs_pop
```

## Natural splines 

Natural cubic splines are also implemented.

```{code-cell} ipython3
ns_pop = ns('Population', internal_knots=knots)
design = ModelSpec([ns_pop], intercept=False)
py_features = np.asarray(design.fit_transform(Carseats))
```

```{code-cell} ipython3
%%R -o R_features
library(splines)
R_features = ns(Carseats$Population, knots=knots)
```

```{code-cell} ipython3
np.linalg.norm(py_features - R_features)
```

## Intercept

Looking at `py_features` we see it contains columns: `[Population**i for i in range(1, 4)]`. That is, 
it doesn't contain an intercept, the order 0 term. This can be include with `intercept=True`. This means that the
column space includes an intercept, though there is no specific column labeled as intercept.

```{code-cell} ipython3
bs_int = ns('Population', internal_knots=knots, intercept=True)
design = ModelSpec([bs_int], intercept=False)
py_int_features = np.asarray(design.fit_transform(Carseats))
```

```{code-cell} ipython3
py_int_features.shape, py_features.shape
```
