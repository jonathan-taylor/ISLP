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

# Brain Cancer Data

A data set consisting of survival times for patients diagnosed
with brain cancer.
     
- `sex`: Factor with levels "Female" and "Male"

- `diagnosis`: Factor with levels "Meningioma", "LG glioma", "HG glioma", and "Other".

- `loc`: Location factor with levels "Infratentorial" and "Supratentorial".

- `ki`: Karnofsky index.

- `gtv`: Gross tumor volume, in cubic centimeters.

- `stereo`: Stereotactic method factor with levels "SRS" and "SRT".

- `status`: Whether the patient is still alive at the end of the study: 0=Yes, 1=No.

- `time`: Survival time, in months

## Source

- I. Selingerova, H. Dolezelova, I. Horova, S. Katina, and J.
     Zelinka. Survival of patients with primary brain tumors:
     Comparison of two statistical approaches. PLoS One,
     11(2):e0148733, 2016.
     [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4749663/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4749663/)

```{code-cell}
from ISLP import load_data
BrainCancer = load_data('BrainCancer')
BrainCancer.columns
```

```{code-cell}
BrainCancer.shape
```

```{code-cell}
BrainCancer.columns
```

```{code-cell}
BrainCancer.describe()
```

```{code-cell}
BrainCancer['diagnosis'].value_counts()
```
