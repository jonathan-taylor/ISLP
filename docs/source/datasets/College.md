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

# U.S. News and World Report's College Data

Statistics for a large number of US Colleges from the 1995 issue
of US News and World Report.

- `Private`: A factor with levels No and Yes indicating private or public university

- `Apps`: Number of applications received

- `Accept`: Number of applications accepted

- `Enroll`: Number of new students enrolled

- `Top10perc`: Pct. new students from top 10% of H.S. class

- `Top25perc`: Pct. new students from top 25% of H.S. class

- `F.Undergrad`: Number of fulltime undergraduates

- `P.Undergrad`: Number of parttime undergraduates

- `Outstate`: Out-of-state tuition

- `Room.Board`: Room and board costs

- `Books`: Estimated book costs

- `Personal`: Estimated personal spending

- `PhD`: Pct. of faculty with Ph.D.’s

- `Terminal`: Pct. of faculty with terminal degree

- `S.F.Ratio`: Student/faculty ratio

- `perc.alumni`: Pct. alumni who donate

- `Expend`: Instructional expenditure per student

- `Grad.Rate`: Graduation rate

## Source

This dataset was taken from the StatLib library which is
maintained at Carnegie Mellon University. The dataset was used in
the ASA Statistical Graphics Section's 1995 Data Analysis
Exposition.

```{code-cell}
from ISLP import load_data
College = load_data('College')
College.columns
```

```{code-cell}
College.shape
```

```{code-cell}
College.columns
```

```{code-cell}
College.describe().iloc[:,:4]
```
