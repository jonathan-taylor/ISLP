---
jupytext:
  cell_metadata_filter: -all
  formats: notebooks/datasets///ipynb,source/datasets///md:myst
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

# Fund Manager Data

A simulated data set containing the returns for 2,000 hedge fund
managers.

```{code-cell}
from ISLP import load_data
Fund = load_data('Fund')
Fund.columns
```

```{code-cell}
Fund.shape
```

```{code-cell}
Fund.columns
```
