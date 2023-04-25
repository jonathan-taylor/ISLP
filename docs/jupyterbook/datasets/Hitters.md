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

# Baseball Data

Major League Baseball Data from the 1986 and 1987 seasons.

- `AtBat`: Number of times at bat in 1986

- `Hits`: Number of hits in 1986

- `HmRun`: Number of home runs in 1986

- `Runs`: Number of runs in 1986

- `RBI`: Number of runs batted in in 1986

- `Walks`: Number of walks in 1986

- `Years`: Number of years in the major leagues

- `CAtBat`: Number of times at bat during his career

- `CHits`: Number of hits during his career

- `CHmRun`: Number of home runs during his career

- `CRuns`: Number of runs during his career

- `CRBI`: Number of runs batted in during his career

- `CWalks`: Number of walks during his career

- `League`: A factor with levels A and N indicating player’s league at the end of 1986

- `Division`: A factor with levels E and W indicating player’s division at the end of 1986

- `PutOuts`: Number of put outs in 1986

- `Assists`: Number of assists in 1986

- `Errors`: Number of errors in 1986

- `Salary`: 1987 annual salary on opening day in thousands of dollars

- `NewLeague`: A factor with levels A and N indicating player’s league at the beginning of 1987

## Notes

This dataset was taken from the StatLib library which is maintained at
Carnegie Mellon University. This is part of the data that was
used in the 1988 ASA Graphics Section Poster Session. The salary
data were originally from Sports Illustrated, April 20, 1987. The
1986 and career statistics were obtained from The 1987 Baseball
Encyclopedia Update published by Collier Books, Macmillan
Publishing Company, New York.

```{code-cell}
from ISLP import load_data
Hitters = load_data('Hitters')
Hitters.columns
```

```{code-cell}
Hitters.shape
```

```{code-cell}
Hitters.columns
```

```{code-cell}
Hitters.describe().iloc[:,:4]
```
