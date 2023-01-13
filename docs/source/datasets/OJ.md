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

# Orange Juice Data

The data contains 1070 purchases where the customer either
purchased Citrus Hill or Minute Maid Orange Juice. A number of
characteristics of the customer and product are recorded.

     
- `Purchase`: A factor with levels 'CH' and 'MM' indicating whether
the customer purchased Citrus Hill or Minute Maid Orange
Juice

- `WeekofPurchase`: Week of purchase

- `StoreID`: Store ID

- `PriceCH`: Price charged for CH

- `PriceMM`: Price charged for MM

- `DiscCH`: Discount offered for CH

- `DiscMM`: Discount offered for MM

- `SpecialCH`: Indicator of special on CH

- `SpecialMM`: Indicator of special on MM

- `LoyalCH`: Customer brand loyalty for CH

- `SalePriceMM`: Sale price for MM

- `SalePriceCH`: Sale price for CH

- `PriceDiff`: Sale price of MM less sale price of CH

- `Store7`: A factor with levels 'No' and 'Yes' indicating whether
the sale is at Store 7

- `PctDiscMM`: Percentage discount for MM

- `PctDiscCH`: Percentage discount for CH

- `ListPriceDiff`: List price of MM less list price of CH

- `STORE`: Which of 5 possible stores the sale occured at

## Source

Stine, Robert A., Foster, Dean P., Waterman, Richard P. Business
Analysis Using Regression (1998). Published by Springer.

```{code-cell}
from ISLP import load_data
OJ = load_data('OJ')
OJ.columns
```

```{code-cell}
OJ.shape
```

```{code-cell}
OJ.columns
```

```{code-cell}
OJ.describe().iloc[:,:4]
```
