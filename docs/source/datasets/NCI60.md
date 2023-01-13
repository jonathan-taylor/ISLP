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
---

NCI 60 Data

Description:

     NCI microarray data. The data contains expression levels on 6830
     genes from 64 cancer cell lines. Cancer type is also recorded.

Usage:

     NCI60
     
Format:

     The format is a list containing two elements: ‘data’ and ‘labs’.

     ‘data’ is a 64 by 6830 matrix of the expression values while
     ‘labs’ is a vector listing the cancer types for the 64 cell lines.

Source:

     The data come from Ross et al. (Nat Genet., 2000). More
     information can be obtained at
     http://genome-www.stanford.edu/nci60/
