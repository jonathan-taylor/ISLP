---
jupytext:
  cell_metadata_filter: -all
  formats: source/datasets///ipynb,jupyterbook/datasets///md:myst,jupyterbook/datasets///ipynb
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

# Portfolio Data

A simple simulated data set containing 100 returns for each of two
assets, X and Y. The data is used to estimate the optimal fraction
to invest in each asset to minimize investment risk of the
combined portfolio. One can then use the Bootstrap to estimate the
standard error of this estimate.

- `X`: Returns for Asset X

- `Y`: Returns for Asset Y

```{code-cell}
from ISLP import load_data
Portfolio = load_data('Portfolio')
Portfolio.columns
```

```{code-cell}
Portfolio.shape
```

```{code-cell}
Portfolio.columns
```

```{code-cell}
Portfolio.describe()
```
