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

# Credit Card Balance Data

A simulated data set containing information on 400 customers.

- `Income`: Income in $1,000’s

- `Limit`: Credit limit

- `Rating`: Credit rating

- `Cards`: Number of credit cards

- `Age`: Age in years

- `Education`: Education in years

- `Own`: A factor with levels No and Yes indicating whether the individual owns a home

- `Student`: A factor with levels No and Yes indicating whether the individual is a student

- `Married`: A factor with levels No and Yes indicating whether the individual is married

- `Region`: A factor with levels East, South, and West indicating the individual’s geographical location

- `Balance`: Average credit card balance in $


## Source

Simulated data. Many thanks to Albert Kim for helpful suggestions,
and for supplying a draft of the man documentation page on Oct 19,
2017.

```{code-cell}
from ISLP import load_data
Credit = load_data('Credit')
Credit.columns
```

```{code-cell}
Credit.shape
```

```{code-cell}
Credit.columns
```

```{code-cell}
Credit.describe().iloc[:,:4]
```
