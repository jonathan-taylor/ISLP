---
jupytext:
  cell_metadata_filter: -all
  formats: source/helpers///ipynb,jupyterbook/helpers///md:myst,jupyterbook/helpers///ipynb
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

# Survival Analysis

This module has a single function, used to simulate data with a given
cumulative survival function.

```{code-cell}
import numpy as np
from lifelines import KaplanMeierFitter
from ISLP.survival import sim_time
```

## Define a cumulative hazard

For simplicity we'll use the the baseline $H(t)=t$ which defines the exponential distribution.

We'll take as our linear predictor $l=\log(2)$. This means we will observe draws from $H_l(t)=2t$ which
corresponds to an exponential distribution with mean 0.5.

```{code-cell}
cum_haz = lambda t: t
rng = np.random.default_rng(1)
```

```{code-cell}
T = np.array([sim_time(np.log(2), cum_haz, rng) for _ in range(500)])
```

## Plot survival function

```{code-cell}
kmf = KaplanMeierFitter(label="Simulated data")
kmf.fit(T, np.ones_like(T))
ax = kmf.plot()
Tval = np.linspace(0, T.max(), 500)
ax.plot(Tval, 
        np.exp(-2*Tval),
        'r--',
        linewidth=4,
        label='Truth')
ax.legend();
```

```{code-cell}

```

```{code-cell}

```
