---
jupytext:
  formats: notebooks/datasets///ipynb,source/datasets///md:myst
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

# Auto Data

Gas mileage, horsepower, and other information for 392 vehicles.

A data frame with 392 observations on the following 9 variables.

- `mpg`: miles per gallon

- `cylinders`: Number of cylinders between 4 and 8

- `displacement`: Engine displacement (cu. inches)

- `horsepower`: Engine horsepower

- `weight`: Vehicle weight (lbs.)

- `acceleration`: Time to accelerate from 0 to 60 mph (sec.)

- `year`: Model year (modulo 100)

- `origin`: Origin of car (1. American, 2. European, 3. Japanese)

- `name`: Vehicle name


## Notes

This dataset was taken from the StatLib library which is maintained at
Carnegie Mellon University. The dataset was used in the 1983
American Statistical Association Exposition. The original dataset
has 397 observations, of which 5 have missing values for the
variable `horsepower`. These rows are removed here. The original
dataset is available at [the book's website](https://www.statlearning.com).

```{code-cell}
from ISLP import load_data
Auto = load_data('Auto')
Auto.columns
```

```{code-cell}
Auto.shape
```

```{code-cell}
Auto.columns
```

```{code-cell}
Auto.describe()
```
