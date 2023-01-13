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

# Bike sharing data


This data set contains the hourly and daily count of rental bikes
between years 2011 and 2012 in Capital bikeshare system, along
with weather and seasonal information.
     
- `season`: Season of the year, coded as Winter=1, Spring=2,
          Summer=3, Fall=4.

- `mnth`: Month of the year, coded as a factor.

- `day`: Day of the year, from 1 to 365

- `hr`: Hour of the day, coded as a factor from 0 to 23.

- `holiday`: Is it a holiday? Yes=1, No=0.

- `weekday`: Day of the week, coded from 0 to 6, where Sunday=0,
          Monday=1, Tuesday=2, etc.

- `workingday`: Is it a work day? Yes=1, No=0.

- `weathersit`: Weather, coded as a factor.

- `temp`: Normalized temperature in Celsius. The values are derived
          via `(t-t_min)/(t_max-t_min)`, `t_min`=-8, `t_max`=+39.

- `atemp`: Normalized feeling temperature in Celsius. The values are
          derived via `(t-t_min)/(t_max-t_min)`, `t_min`=-16, `t_max`=+50.

- `hum`: Normalized humidity. The values are divided to 100 (max).

- `windspeed`: Normalized wind speed. The values are divided by 67
          (max).

- `casual`: Number of casual bikers.

- `registered`: Number of registered bikers.

- `bikers`: Total number of bikers.

## Source

The [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset).

```{code-cell}
from ISLP import load_data
Bikeshare = load_data('Bikeshare')
Bikeshare.columns
```

```{code-cell}
Bikeshare.shape
```

```{code-cell}
Bikeshare.columns
```

```{code-cell}
Bikeshare.describe().iloc[:,:4]
```
