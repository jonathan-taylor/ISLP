---
jupytext:
  cell_metadata_filter: -all
  formats: source/helpers///ipynb,jupyterbook/helpers///md:myst,jupyterbook/helpers///ipynb
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Clustering

This module has a single function, used to help visualize a dendrogram from a
hierarchical clustering. The function is based on this example from [sklearn.cluster](https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html).

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
