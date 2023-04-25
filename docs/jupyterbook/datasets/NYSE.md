---
jupytext:
  cell_metadata_filter: -all
  formats: source/datasets///ipynb,jupyterbook/datasets///md:myst,jupyterbook/datasets///ipynb
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

# New York Stock Exchange Data

Data consisting of the Dow Jones returns, log trading volume, and
log volatility for the New York Stock Exchange over a 20 year
period

- `date`: Date

- `day_of_week`: Day of the week

- `DJ_return`: Return for Dow Jones Industrial Average

- `log_volume`: Log of trading volume

- `log_volatility`: Log of volatility

- `train`: For the first 4,281 observations, this is set to `True`

## Source

- B. LeBaron and A. Weigend (1998), IEEE Transactions on Neural
Networks 9(1): 213-220.

```{code-cell}
from ISLP import load_data
NYSE = load_data('NYSE')
NYSE.columns
```

```{code-cell}
NYSE.shape
```

```{code-cell}
NYSE.columns
```

```{code-cell}
NYSE.describe()
```
