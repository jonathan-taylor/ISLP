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

# Caravan

The data contains 5822 real customer records. Each record consists of 86 variables, containing
sociodemographic data (variables 1-43) and product ownership (variables 44-86). The sociodemographic data is derived from zip codes. All customers living in areas with the same zip code have
the same sociodemographic attributes. Variable 86 (Purchase) indicates whether the customer purchased a caravan insurance policy. Further information on the individual variables can be obtained
at [http://www.liacs.nl/~putten/library/cc2000/data.html](http://www.liacs.nl/~putten/library/cc2000/data.html)

## References

- P. van der Putten and M. van Someren (eds) . CoIL Challenge
  2000: The Insurance Company Case.  Published by Sentient Machine
  Research, Amsterdam. Also a Leiden Institute of Advanced Computer
  Science Technical Report 2000-09. June 22, 2000. See
  [http://www.liacs.nl/~putten/library/cc2000/](http://www.liacs.nl/~putten/library/cc2000/)

-  P. van der Putten and M. van Someren. A Bias-Variance Analysis of a Real World Learning Problem: The CoIL Challenge 2000. Machine Learning, October 2004, vol. 57, iss. 1-2, pp. 177-195, Kluwer Academic Publishers

```{code-cell} ipython3
from ISLP import load_data
Caravan = load_data('Caravan')
Caravan.columns
```

```{code-cell} ipython3
Caravan.shape
```

```{code-cell} ipython3
Caravan.columns[:20]
```
