---
jupytext:
  cell_metadata_filter: -all
  formats: notebooks///ipynb,source/datasets///md:myst
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

# Sales of Child Car Seats

A simulated data set containing sales of child car seats at 400
different stores.

- `Sales`: Unit sales (in thousands) at each location

- `CompPrice`: Price charged by competitor at each location

- `Income`: Community income level (in thousands of dollars)

- `Advertising`: Local advertising budget for company at each location (in thousands of dollars)

- `Population`: Population size in region (in thousands)

- `Price`: Price company charges for car seats at each site

- `ShelveLoc`: A factor with levels Bad, Good and Medium indicating the quality of the shelving location for the car seats at each site

- `Age`: Average age of the local population

- `Education`: Education level at each location

- `Urban`: A factor with levels No and Yes to indicate whether the store is in an urban or rural location

- `US`: A factor with levels No and Yes to indicate whether the store is in the US or not

```{code-cell}
from ISLP import load_data
Carseats = load_data('Carseats')
Carseats.columns
```

```{code-cell}
Carseats.shape
```

```{code-cell}
Carseats.columns
```

```{code-cell}
Carseats.describe().iloc[:,:4]
```
