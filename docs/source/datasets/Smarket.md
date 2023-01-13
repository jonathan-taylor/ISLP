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

# S&P Stock Market Data

Daily percentage returns for the S&P 500 stock index between 2001
and 2005.

- `Year`: The year that the observation was recorded

- `Lag1`: Percentage return for previous day

- `Lag2`: Percentage return for 2 days previous

- `Lag3`: Percentage return for 3 days previous

- `Lag4`: Percentage return for 4 days previous

- `Lag5`: Percentage return for 5 days previous

- `Volume`: Volume of shares traded (number of daily shares traded in
          billions)

- `Today`: Percentage return for today

- `Direction`: A factor with levels 'Down' and 'Up' indicating
 whether the market had a positive or negative return on a
 given day

## Source

Raw values of the S&P 500 were obtained from Yahoo Finance and
then converted to percentages and lagged.

```{code-cell}
from ISLP import load_data
Smarket = load_data('Smarket')
Smarket.columns
```

```{code-cell}
Smarket.shape
```

```{code-cell}
Smarket.columns
```

```{code-cell}
Smarket.describe().iloc[:,-4:]
```
