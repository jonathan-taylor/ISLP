---
jupytext:
  formats: source/models///ipynb,jupyterbook/models///md:myst,jupyterbook/models///ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: python3
  language: python
  name: python3
---

# Model selection using `ModelSpec`

```{code-cell} ipython3
import numpy as np, pandas as pd
%load_ext rpy2.ipython

from ISLP import load_data
from ISLP.models import ModelSpec

import statsmodels.api as sm
```

```{code-cell} ipython3
Carseats = load_data('Carseats')
%R -i Carseats
Carseats.columns
```

## Let's break up income into groups

```{code-cell} ipython3
Carseats['OIncome'] = pd.cut(Carseats['Income'], 
                             [0,50,90,200], 
                             labels=['L','M','H'])
Carseats['OIncome']
```

Let's also create an unordered version

```{code-cell} ipython3
Carseats['UIncome'] = pd.cut(Carseats['Income'], 
                             [0,50,90,200], 
                             labels=['L','M','H'],
                             ordered=False)
Carseats['UIncome']
```

## A simple model

```{code-cell} ipython3
design = ModelSpec(['Price', 'Income'])
X = design.fit_transform(Carseats)
X.columns
```

```{code-cell} ipython3
Y = Carseats['Sales']
M = sm.OLS(Y, X).fit()
M.params
```

## Basic procedure

The design matrix is built by cobbling together a set of columns and possibly transforming them.
A `pd.DataFrame` is essentially a list of columns. One of the first tasks done  in `ModelSpec.fit`
is to inspect a dataframe for column info. The column `ShelveLoc` is categorical:

```{code-cell} ipython3
Carseats['ShelveLoc']
```

This is recognized by `ModelSpec` in the form of `Column` objects which are just named tuples with two methods
`get_columns` and `fit_encoder`.

```{code-cell} ipython3
design.column_info_['ShelveLoc']
```

It recognized ordinal columns as well.

```{code-cell} ipython3
design.column_info_['OIncome']
```

```{code-cell} ipython3
income = design.column_info_['Income']
cols, names = income.get_columns(Carseats)
(cols[:4], names)
```

## Encoding a column

In building a design matrix we must extract columns from our dataframe (or `np.ndarray`). Categorical
variables usually are encoded by several columns, typically one less than the number of categories.
This task is handled by the `encoder` of the `Column`. The encoder must satisfy the `sklearn` transform
model, i.e. `fit` on some array and `transform` on future arrays. The `fit_encoder` method of `Column` fits
its encoder the first time data is passed to it.

```{code-cell} ipython3
shelve = design.column_info_['ShelveLoc']
cols, names = shelve.get_columns(Carseats)
(cols[:4], names)
```

```{code-cell} ipython3
oincome = design.column_info_['OIncome']
oincome.get_columns(Carseats)[0][:4]
```

## The terms

The design matrix consists of several sets of columns. This is managed by the `ModelSpec` through
the `terms` argument which should be a sequence. The elements of `terms` are often
going to be strings (or tuples of strings for interactions, see below) but are converted to a
`Variable` object and stored in the `terms_` of the fitted `ModelSpec`. A `Variable` is just a named tuple.

```{code-cell} ipython3
design.terms
```

```{code-cell} ipython3
design.terms_
```

While each `Column` can itself extract data, they are all promoted to `Variable` to be of a uniform type.  A
`Variable` can also create columns through the `build_columns` method of `ModelSpec`

```{code-cell} ipython3
price = design.terms_[0]
design.build_columns(Carseats, price)
```

Note that `Variable` objects have a tuple of `variables` as well as an `encoder` attribute. The
tuple of `variables` first creates a concatenated dataframe from all corresponding variables and then
is run through `encoder.transform`. The `encoder.fit` method of each `Variable` is run once during 
the call to `ModelSpec.fit`.

```{code-cell} ipython3
from ISLP.models.model_spec import Variable

new_var = Variable(('Price', 'Income', 'UIncome'), name='mynewvar', encoder=None)
design.build_columns(Carseats, new_var)
```

Let's now transform these columns with an encoder. Within `ModelSpec` we will first build the
arrays above and then call `pca.fit` and finally `pca.transform` within `design.build_columns`.

```{code-cell} ipython3
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(design.build_columns(Carseats, new_var)[0]) # this is done within `ModelSpec.fit`
pca_var = Variable(('Price', 'Income', 'UIncome'), name='mynewvar', encoder=pca)
design.build_columns(Carseats, pca_var)
```

The elements of the `variables` attribute may be column identifiers ( `"Price"`), `Column` instances (`price`)
or `Variable` instances (`pca_var`).

```{code-cell} ipython3
fancy_var = Variable(('Price', price, pca_var), name='fancy', encoder=None)
design.build_columns(Carseats, fancy_var)
```

We can of course run PCA again on these features (if we wanted).

```{code-cell} ipython3
pca2 = PCA(n_components=2)
pca2.fit(design.build_columns(Carseats, fancy_var)[0]) # this is done within `ModelSpec.fit`
pca2_var = Variable(('Price', price, pca_var), name='fancy_pca', encoder=pca2)
design.build_columns(Carseats, pca2_var)
```

## Building the design matrix

With these notions in mind, the final design is essentially then

```{code-cell} ipython3
X_hand = np.column_stack([design.build_columns(Carseats, v)[0] for v in design.terms_])[:4]
```

An intercept column is added if `design.intercept` is `True` and if the original argument to `transform` is
a dataframe the index is adjusted accordingly.

```{code-cell} ipython3
design.intercept
```

```{code-cell} ipython3
design.transform(Carseats)[:4]
```

## Predicting

Constructing the design matrix at any values is carried out by the `transform` method.

```{code-cell} ipython3
new_data = pd.DataFrame({'Price':[10,20], 'Income':[40, 50]})
new_X = design.transform(new_data)
M.get_prediction(new_X).predicted_mean
```

```{code-cell} ipython3
%%R -i new_data,Carseats
predict(lm(Sales ~ Price + Income, data=Carseats), new_data)
```

### Difference between using `pd.DataFrame` and `np.ndarray`

If the `terms` only refer to a few columns of the data frame, the `transform` method only needs a dataframe with those columns.

If we had used an `np.ndarray`, the column identifiers would be integers identifying specific columns so,
in order to work correctly, `transform` would need another `np.ndarray` where the columns have the same meaning.

```{code-cell} ipython3
Carseats_np = np.asarray(Carseats[['Price', 'ShelveLoc', 'US', 'Income']])
design_np = ModelSpec([0,3]).fit(Carseats_np)
design_np.transform(Carseats_np)[:4]
```

The following will fail for hopefully obvious reasons

```{code-cell} ipython3
try:
    new_D = np.zeros((2,2))
    new_D[:,0] = [10,20]
    new_D[:,1] = [40,50]
    M.get_prediction(new_D).predicted_mean
except ValueError as e:
    print(e)
```

Ultimately, `M` expects 3 columns for new predictions because it was fit
with a matrix having 3 columns (the first representing an intercept).

We might be tempted to try as with the `pd.DataFrame` and produce
an `np.ndarray` with only the necessary variables.

```{code-cell} ipython3
try:
    new_X = np.zeros((2,2))
    new_X[:,0] = [10,20]
    new_X[:,1] = [40,50]
    new_D = design_np.transform(new_X)
    M.get_prediction(new_D).predicted_mean
except IndexError as e:
    print(e)
```

This fails because `design_np` is looking for column `3` from its `terms`:

```{code-cell} ipython3
design_np.terms_
```

However, if we have an `np.ndarray` in which the first column indeed represents `Price` and the fourth indeed
represents `Income` then we can arrive at the correct answer by supplying such the array to `design_np.transform`:

```{code-cell} ipython3
new_X = np.zeros((2,4))
new_X[:,0] = [10,20]
new_X[:,3] = [40,50]
new_D = design_np.transform(new_X)
M.get_prediction(new_D).predicted_mean
```

Given this subtlety about needing to supply arrays with identical column structure to `transform` when
using `np.ndarray` we presume that using a `pd.DataFrame` will be the more popular use case.

+++

## A model with some categorical variables

Categorical variables become `Column` instances with encoders.

```{code-cell} ipython3
design = ModelSpec(['Population', 'Price', 'UIncome', 'ShelveLoc']).fit(Carseats)
design.column_info_['UIncome']
```

```{code-cell} ipython3
X = design.fit_transform(Carseats)
X.columns
```

```{code-cell} ipython3
sm.OLS(Y, X).fit().params
```

```{code-cell} ipython3
%%R
lm(Sales ~ Population + Price + UIncome + ShelveLoc, data=Carseats)$coef
```

## Getting the encoding you want

By default the level dropped by `ModelSpec` will be the first of the `categories_` values from 
`sklearn.preprocessing.OneHotEncoder()`. We might wish to change this. It seems
as if the correct way to do this would be something like `Variable(('UIncome',), 'mynewencoding', new_encoder)`
where `new_encoder` would somehow drop the column we want dropped. 

However, when using the convenient identifier `UIncome` in the `variables` argument, this maps to the `Column` associated to `UIncome` within `design.column_info_`:

```{code-cell} ipython3
design.column_info_['UIncome']
```

This column already has an encoder and `Column` instances are immutable as named tuples. Further, there are times when 
we may want to encode `UIncome` differently within the same model. In the model below the main effect of `UIncome` is encoded with two columns while in the interaction `UIncome` (see below) has three columns. This is a design of interest
and we need a way to allow different encodings of the same column of `Carseats`

```{code-cell} ipython3
%%R
lm(Sales ~ UIncome:ShelveLoc + UIncome, data=Carseats)
```

 We can create a new 
`Column` with the encoder we want. For categorical variables, there is a convenience function to do so.

```{code-cell} ipython3
from ISLP.models.model_spec import contrast
pref_encoding = contrast('UIncome', 'drop', 'L')
```

```{code-cell} ipython3
design.build_columns(Carseats, pref_encoding)
```

```{code-cell} ipython3
design = ModelSpec(['Population', 'Price', pref_encoding, 'ShelveLoc']).fit(Carseats)
X = design.fit_transform(Carseats)
X.columns
```

```{code-cell} ipython3
sm.OLS(Y, X).fit().params
```

```{code-cell} ipython3
%%R
lm(Sales ~ Population + Price + UIncome + ShelveLoc, data=Carseats)$coef
```

## Interactions

We've referred to interactions above. These are specified (by convenience) as tuples in the `terms` argument
to `ModelSpec`.

```{code-cell} ipython3
design = ModelSpec([('UIncome', 'ShelveLoc'), 'UIncome'])
X = design.fit_transform(Carseats)
sm.OLS(Y, X).fit().params
```

The tuples in `terms` are converted to `Variable` in the formalized `terms_` attribute by creating a `Variable` with
`variables` set to the tuple and the encoder an `Interaction` encoder which (unsurprisingly) creates the interaction columns from the concatenated data frames of `UIncome` and `ShelveLoc`.

```{code-cell} ipython3
design.terms_[0]
```

Comparing this to the previous `R` model.

```{code-cell} ipython3
%%R
lm(Sales ~ UIncome:ShelveLoc + UIncome, data=Carseats)
```

We note a few important things:

1. `R` has reorganized the columns of the design from the formula: although we wrote `UIncome:ShelveLoc` first these
columns have been built later. **`ModelSpec` builds columns in the order determined by `terms`!**

2. As noted above, `R` has encoded `UIncome` differently in the main effect and in the interaction. For `ModelSpec`, the reference to `UIncome` always refers to the column in `design.column_info_` and will always build only the columns for `L` and `M`. **`ModelSpec` does no inspection of terms to decide how to encode categorical variables.**

A few notes:

- **Why not try to inspect the terms?** For any nontrivial formula in `R` with several categorical variables and interactions, predicting what columns will be produced from a given formula is not simple. **`ModelSpec` errs on the side of being explicit.**

- **Is it impossible to build the design as `R` has?** No. An advanced user who *knows* they want the columns built as `R` has can do so (fairly) easily.

```{code-cell} ipython3
full_encoding = contrast('UIncome', None)
design.build_columns(Carseats, full_encoding)
```

```{code-cell} ipython3
design = ModelSpec([pref_encoding, (full_encoding, 'ShelveLoc')])
X = design.fit_transform(Carseats)
sm.OLS(Y, X).fit().params
```

## Special encodings

For flexible models, we may want to consider transformations of features, i.e. polynomial
or spline transformations. Given transforms that follow the `fit/transform` paradigm
we can of course achieve this with a `Column` and an `encoder`. The `ISLP.transforms`
package includes a `Poly` transform

```{code-cell} ipython3
from ISLP.models.model_spec import poly
poly('Income', 3)
```

```{code-cell} ipython3
design = ModelSpec([poly('Income', 3), 'ShelveLoc'])
X = design.fit_transform(Carseats)
sm.OLS(Y, X).fit().params
```

Compare:

```{code-cell} ipython3
%%R
lm(Sales ~ poly(Income, 3) + ShelveLoc, data=Carseats)$coef
```

## Splines

Support for natural and B-splines is also included

```{code-cell} ipython3
from ISLP.models.model_spec import ns, bs, pca
design = ModelSpec([ns('Income', df=5), 'ShelveLoc'])
X = design.fit_transform(Carseats)
sm.OLS(Y, X).fit().params
```

```{code-cell} ipython3
%%R
library(splines)
lm(Sales ~ ns(Income, df=5) + ShelveLoc, data=Carseats)$coef
```

```{code-cell} ipython3
design = ModelSpec([bs('Income', df=7, degree=2), 'ShelveLoc'])
X = design.fit_transform(Carseats)
sm.OLS(Y, X).fit().params
```

```{code-cell} ipython3
%%R
lm(Sales ~ bs(Income, df=7, degree=2) + ShelveLoc, data=Carseats)$coef
```

## PCA

```{code-cell} ipython3
design = ModelSpec([pca(['Income', 
                           'Price', 
                           'Advertising', 
                           'Population'], 
                          n_components=2, 
                          name='myvars'), 'ShelveLoc'])
X = design.fit_transform(Carseats)
sm.OLS(Y, X).fit().params
```

```{code-cell} ipython3
%%R
lm(Sales ~ prcomp(cbind(Income, Price, Advertising, Population))$x[,1:2] + ShelveLoc, data=Carseats)
```

It is of course common to scale before running PCA.

```{code-cell} ipython3
design = ModelSpec([pca(['Income', 
                           'Price', 
                           'Advertising', 
                           'Population'], 
                          n_components=2, 
                          name='myvars',
                          scale=True), 'ShelveLoc'])
X = design.fit_transform(Carseats)
sm.OLS(Y, X).fit().params
```

```{code-cell} ipython3
%%R
lm(Sales ~ prcomp(cbind(Income, Price, Advertising, Population), scale=TRUE)$x[,1:2] + ShelveLoc, data=Carseats)
```

There will be some small differences in the coefficients due to `sklearn` use of `np.std(ddof=0)` instead
of `np.std(ddof=1)`.

```{code-cell} ipython3
np.array(sm.OLS(Y, X).fit().params)[1:3] * np.sqrt(X.shape[0] / (X.shape[0]-1))
```

## Model selection

Another task requiring different design matrices is model selection. Manipulating
the `terms` attribute of a `ModelSpec` (or more precisely its more uniform version `terms_`)
can clearly allow for both exhaustive and stepwise model selection.

```{code-cell} ipython3
from ISLP.models.strategy import (Stepwise, 
                                  min_max)
from ISLP.models.generic_selector import FeatureSelector
```

### Best subsets

```{code-cell} ipython3
design = ModelSpec(['Price', 
                    'UIncome', 
                    'Advertising', 
                    'US', 
                    'Income',
                    'ShelveLoc',
                    'Education',
                    'Urban']).fit(Carseats)
strategy = min_max(design,
                   min_terms=0,
                   max_terms=3)
```

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression
selector = FeatureSelector(LinearRegression(fit_intercept=False),
                           strategy,
                           scoring='neg_mean_squared_error')
```

```{code-cell} ipython3
selector.fit(Carseats, Y)
```

```{code-cell} ipython3
selector.selected_state_
```

```{code-cell} ipython3
selector.results_.keys()
```

```{code-cell} ipython3
strategy = min_max(design,
                   min_terms=0,
                   max_terms=3,
                   lower_terms=['Price'],
                   upper_terms=['Price', 'Income', 'Advertising'])
selector = FeatureSelector(LinearRegression(fit_intercept=False),
                           strategy,
                           scoring='neg_mean_squared_error')
selector.fit(Carseats, Y)
selector.selected_state_
```

```{code-cell} ipython3
selector.results_.keys()
```

### Stepwise selection

```{code-cell} ipython3
strategy = Stepwise.first_peak(design,
                               min_terms=0,
                               max_terms=6,
                               lower_terms=['Price'],
                               upper_terms=['Price', 'Income', 'Advertising', 'ShelveLoc', 'UIncome', 'US'
                                     'Education', 'Urban'])
selector = FeatureSelector(LinearRegression(fit_intercept=False),
                           strategy,
                           scoring='neg_mean_squared_error',
                           cv=3)
selector.fit(Carseats, Y)
selector.selected_state_
```

```{code-cell} ipython3
selector.results_.keys()
```

```{code-cell} ipython3
selector.results_
```

```{code-cell} ipython3
selector.selected_state_
```

### Enforcing constraints

In models with interactions, we may often want to impose constraints on interactions and main effects.
This can be achieved here by use of a `validator` that checks whether a given model is valid.

Suppose we want to have the following constraint: `ShelveLoc` may not be in the model unless
`Price` is in the following model.

```{code-cell} ipython3
design = ModelSpec(['Price', 
                    'Advertising', 
                    'Income',
                    'ShelveLoc']).fit(Carseats)
```

The constraints are described with a boolean matrix with `(i,j)` as `j` is a child of `i`: so `j` should not
be in the model when `i` is not and enforced with a callable `validator` that evaluates each candidate state.

Both `min_max_strategy` and `step_strategy` accept a `validator` argument.

```{code-cell} ipython3
from ISLP.models.strategy import validator_from_constraints
constraints = np.zeros((4, 4))
constraints[0,3] = 1
strategy = min_max(design,
                   min_terms=0,
                   max_terms=4,
                   validator=validator_from_constraints(design,
                                                        constraints))
selector = FeatureSelector(LinearRegression(fit_intercept=False),
                           strategy,
                           scoring='neg_mean_squared_error',
                           cv=3)
selector.fit(Carseats, Y)
selector.results_.keys()
```

```{code-cell} ipython3
selector.selected_state_
```

```{code-cell} ipython3
Hitters=load_data('Hitters')
```

```{code-cell} ipython3
Hitters.columns
```

```{code-cell} ipython3
Hitters = Hitters.dropna()
Y=Hitters['Salary']
X=Hitters.drop('Salary', axis=1)
design = ModelSpec(X.columns).fit(X)
strategy = Stepwise.first_peak(design,
                               direction='forward',
                               min_terms=0,
                               max_terms=19)
selector = FeatureSelector(LinearRegression(fit_intercept=False),
                           strategy,
                           scoring='neg_mean_squared_error', cv=None)
selector.fit(X, Y)
selector.results_.keys()
```

```{code-cell} ipython3
len(selector.selected_state_)
```

```{code-cell} ipython3
len(X.columns)
```

```{code-cell} ipython3
%%R -i Hitters
step(lm(Salary ~ 1, data=Hitters), scope=list(upper=lm(Salary ~ ., data=Hitters)), direction='forward', trace=TRUE)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
