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

# Credit Card Default Data

A simulated data set containing information on ten thousand
customers. The aim here is to predict which customers will default
on their credit card debt.
     
- `default`: A factor with levels ‘No’ and ‘Yes’ indicating whether
          the customer defaulted on their debt

- `student`: A factor with levels ‘No’ and ‘Yes’ indicating whether
          the customer is a student

- `balance`: The average balance that the customer has remaining on
          their credit card after making their monthly payment

- `income`: Income of customer

```{code-cell}
from ISLP import load_data
Default = load_data('Default')
Default.columns
```

```{code-cell}
Default.shape
```

```{code-cell}
Default.columns
```

```{code-cell}
Default.describe()
```

```{code-cell}
Default['student'].value_counts()
```
