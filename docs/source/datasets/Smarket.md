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

S&P Stock Market Data

Description:

     Daily percentage returns for the S&P 500 stock index between 2001
     and 2005.

Usage:

     Smarket
     
Format:

     A data frame with 1250 observations on the following 9 variables.

     ‘Year’ The year that the observation was recorded

     ‘Lag1’ Percentage return for previous day

     ‘Lag2’ Percentage return for 2 days previous

     ‘Lag3’ Percentage return for 3 days previous

     ‘Lag4’ Percentage return for 4 days previous

     ‘Lag5’ Percentage return for 5 days previous

     ‘Volume’ Volume of shares traded (number of daily shares traded in
          billions)

     ‘Today’ Percentage return for today

     ‘Direction’ A factor with levels ‘Down’ and ‘Up’ indicating
          whether the market had a positive or negative return on a
          given day

Source:

     Raw values of the S&P 500 were obtained from Yahoo Finance and
     then converted to percentages and lagged.
