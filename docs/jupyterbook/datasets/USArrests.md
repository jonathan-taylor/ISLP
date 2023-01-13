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

# Violent Crime Rates by US State

This data set contains statistics, in arrests per 100,000
residents for assault, murder, and rape in each of the 50 US
states in 1973.  Also given is the percent of the population
living in urban areas.


- `Murder`: Murder arrests (per 100,000)  

- `Assault`: Assault arrests (per 100,000) 

- `UrbanPop`: Percent urban population      

- `Rape`: Rape arrests (per 100,000)    

## Notes

From the `R` base package. See help with command `?USArrests` (in `R`)

```{code-cell} ipython3
from statsmodels.datasets import get_rdataset
USArrests = get_rdataset('USArrests').data
```

```{code-cell} ipython3
USArrests.shape
```

```{code-cell} ipython3
USArrests.columns
```

```{code-cell} ipython3
USArrests.describe()
```
