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

# Time-to-Publication Data

Publication times for 244 clinical trials funded by the National
Heart, Lung, and Blood Institute.
     

- `posres`: Did the trial produce a positive (significant) result?
  1=Yes, 0=No.

- `multi`: Did the trial involve multiple centers? 1=Yes, 0=No.

- `clinend`: Did the trial focus on a clinical endpoint? 1=Yes, 0=No.

- `mech`: Funding mechanism within National Institute of Health: a
  qualitative variable.

- `sampsize`: Sample size for the trial.

- `budget`: Budget of the trial, in millions of dollars.

- `impact`: Impact of the trial; this is related to the number of
  publications.

- `time`: Time to publication, in months.

- `status`: Whether or not the trial was published at `time`:
  1=Published, 0=Not yet published.

## Source

- Gordon, Taddei-Peters, Mascette, Antman, Kaufmann, and Lauer.
Publication of trials funded by the National Heart, Lung, and
Blood Institute.  New England Journal of Medicine,
369(20):1926-1934, 2013.

```{code-cell}
from ISLP import load_data
Publication = load_data('Publication')
Publication.columns
```

```{code-cell}
Publication.shape
```

```{code-cell}
Publication.columns
```

```{code-cell}
Publication.describe().iloc[:,:4]
```
