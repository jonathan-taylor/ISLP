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

Portfolio Data

Description:

     A simple simulated data set containing 100 returns for each of two
     assets, X and Y. The data is used to estimate the optimal fraction
     to invest in each asset to minimize investment risk of the
     combined portfolio. One can then use the Bootstrap to estimate the
     standard error of this estimate.

Usage:

     Portfolio
     
Format:

     A data frame with 100 observations on the following 2 variables.

     ‘X’ Returns for Asset X

     ‘Y’ Returns for Asset Y

Source:

     Simulated data
