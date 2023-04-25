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

# Derived features: using PCA on a subset of columns

The modelling tools included in `ISLP` allow for
construction of transformers applied to features.

```{code-cell} ipython3
import numpy as np
from ISLP import load_data
from ISLP.models import ModelSpec, pca, Variable, derived_variable
from sklearn.decomposition import PCA
```

```{code-cell} ipython3
Carseats = load_data('Carseats')
Carseats.columns
```

Let's create a `ModelSpec` that is aware of all of the relevant columns.

```{code-cell} ipython3
design = ModelSpec(Carseats.columns.drop(['Sales'])).fit(Carseats)
```

Suppose we want to make a `Variable` representing the first 3 principal components of the
 features `['CompPrice', 'Income', 'Advertising', 'Population', 'Price']`.

+++

We first make a `Variable` that represents these five features columns, then `pca`
can be used to compute a new `Variable` that returns the first three principal components.

```{code-cell} ipython3
grouped = Variable(('CompPrice', 'Income', 'Advertising', 'Population', 'Price'), name='grouped', encoder=None)
sklearn_pca = PCA(n_components=3, whiten=True)
```

We can now fit `sklearn_pca` and create our new variable.

```{code-cell} ipython3
sklearn_pca.fit(design.build_columns(Carseats, grouped)[0]) 
pca_var = derived_variable(['CompPrice', 'Income', 'Advertising', 'Population', 'Price'],
                           name='pca(grouped)', encoder=sklearn_pca)
derived_features, _ = design.build_columns(Carseats, pca_var)
```

```{code-cell} ipython3
design.build_columns(Carseats, grouped)[0]
```

## Helper function

The function `pca` encompasses these steps into a single function for convenience.

```{code-cell} ipython3
group_pca = pca(['CompPrice', 'Income', 'Advertising', 'Population', 'Price'], 
                n_components=3, 
                whiten=True, 
                name='grouped')
```

```{code-cell} ipython3
pca_design = ModelSpec([group_pca], intercept=False)
ISLP_features = pca_design.fit_transform(Carseats)
ISLP_features.columns
```

## Direct comparison

```{code-cell} ipython3
X = np.asarray(Carseats[['CompPrice', 'Income', 'Advertising', 'Population', 'Price']])
sklearn_features = sklearn_pca.fit_transform(X)
```

```{code-cell} ipython3
np.linalg.norm(ISLP_features - sklearn_features), np.linalg.norm(ISLP_features - np.asarray(derived_features))
```
