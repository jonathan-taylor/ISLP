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

# Mid-Atlantic Wage Data

Wage and other data for a group of 3000 male workers in the
Mid-Atlantic region.

- `year`: Year that wage information was recorded

- `age`: Age of worker

- `maritl`: A factor with levels '1. Never Married', '2. Married', '3.
 '3. Widowed', '4. Divorced' and '5. Separated' indicating marital
 status

- `race`: A factor with levels '1. White', '2. Black', '3. Asian' and
 '4. Other' indicating race

- `education`: A factor with levels '1. < HS Grad', '2. HS Grad', 
 '3. Some College', '4. College Grad' and '5. Advanced Degree'
 indicating education level

- `region`: Region of the country (mid-atlantic only)

- `jobclass`: A factor with levels '1. Industrial' and '2.
 Information' indicating type of job

- `health`: A factor with levels '1. <=Good' and '2. >=Very Good'
 indicating health level of worker

- `health_ins`: A factor with levels '1. Yes' and '2. No' indicating
 whether worker has health insurance

- `logwage`: Log of workers wage

- `wage`: Workers raw wage

## Source

Data was manually assembled by Steve Miller, of Inquidia
Consulting (formerly Open BI). From the March 2011 Supplement to
Current Population Survey data.

See also: [re3data.org/repository/r3d100011860](https://www.re3data.org/repository/r3d100011860)

```{code-cell}
from ISLP import load_data
Wage = load_data('Wage')
Wage.columns
```

```{code-cell}
Wage.shape
```

```{code-cell}
Wage.columns
```

```{code-cell}
Wage.describe()
```
