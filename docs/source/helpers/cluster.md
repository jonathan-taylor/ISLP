---
jupytext:
  cell_metadata_filter: -all
  formats: ipynb,md:myst
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

# Clustering

This module has a single function, used to help visualize a dendrogram from a
hierarchical clustering.

```{code-cell} ipython3
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from ISLP.cluster import compute_linkage
```

## Make a toy dataset

```{code-cell} ipython3
rng = np.random.default_rng(1)
X = rng.normal(size=(30, 5))
X[:10] += 1
```

## Cluster it

```{code-cell} ipython3
clust = AgglomerativeClustering(distance_threshold=0,
                                n_clusters=None,
                                linkage='complete')
```

```{code-cell} ipython3
clust.fit(X)
```

## Plot the dendrogram

```{code-cell} ipython3
linkage = compute_linkage(clust)
dendrogram(linkage);
```
