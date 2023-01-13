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

New York Stock Exchange Data

Description:

     Data consisting of the Dow Jones returns, log trading volume, and
     log volatility for the New York Stock Exchange over a 20 year
     period

Usage:

     Portfolio
     
Format:

     A data frame with 6,051 observations and 6 variables:

     ‘date’ Date

     ‘day_of_week’ Day of the week

     ‘DJ_return’ Return for Dow Jones Industrial Average

     ‘log_volume’ Log of trading volume

     ‘log_volatility’ Log of volatility

     ‘train’ For the first 4,281 observations, this is set to TRUE

Source:

     B. LeBaron and A. Weigend (1998), IEEE Transactions on Neural
     Networks 9(1): 213-220.
