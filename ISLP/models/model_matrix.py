from collections import namedtuple
from itertools import product
from typing import NamedTuple, Any

import numpy as np, pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import (OneHotEncoder,
                                   OrdinalEncoder)
from sklearn.exceptions import NotFittedError

from .columns import _get_column_info, Column
from ..transforms import (Poly,
                          BSpline,
                          NaturalSpline,
                          Interaction)

class Variable(NamedTuple):

    variables: tuple
    name: str
    encoder: Any

class ModelMatrix(TransformerMixin, BaseEstimator):

    def __init__(self,
                 intercept=True,
                 terms=None,
                 categorical_features=None,
                 default_encoders={'categorical': OneHotEncoder(drop=None, sparse=False),
                                   'ordinal': OrdinalEncoder()}
                 ):

        '''

        Parameters
        ----------

        degree : int
            Degree of polynomial.

        intercept : bool (optional)
            Include a column for intercept?

        terms : sequence (optional)
            Sequence of sets whose
            elements are columns of `X` when fit.
            For `pd.DataFrame` these can be column
            names.

        categorical_features : array-like of {bool, int} of shape (n_features) 
                or shape (n_categorical_features,), default=None.
            Indicates the categorical features. Will be ignored if `X` is a `pd.DataFrame`
            or `pd.Series`.

            - None : no feature will be considered categorical for `np.ndarray`.
            - boolean array-like : boolean mask indicating categorical features.
            - integer array-like : integer indices indicating categorical
              features.

        transforms : dict
            Dictionary whose keys are elements of `terms` and values
            are transforms to be applied to the associate columns in the model matrix
            by running the `fit_transform` method when `fit` is called and overwriting
            these values in the dictionary.

        '''
        
        self.intercept = intercept
        self.terms = terms
        self.categorical_features = categorical_features
        self.default_encoders = default_encoders
        
    def fit(self, X, y=None):

        """
        Construct parameters for orthogonal
        polynomials in the feature X.

        Parameters
        ----------
        X : array-like
            X on which model matrix will be evaluated.
            If a `pd.DataFrame` or `pd.Series`, variables that are of
            categorical dtype will be treated as categorical.

        """
        
        if isinstance(X, (pd.DataFrame, pd.Series)):
            is_dataframe = True
            (categorical_features,
             self.is_ordinal_) = _categorical_from_df(X)
            (self.is_categorical_,
             self.known_categories_) = _check_categories(categorical_features,
                                                         X)
            self.columns_ = X.columns
            if self.is_categorical_ is None:
                self.is_categorical_ = np.zeros(X.shape[1], np.bool)
            self.is_ordinal_ = pd.Series(self.is_ordinal_,
                                         index=self.columns_)
            self.is_categorical_ = pd.Series(self.is_categorical_,
                                             index=self.columns_)
        else:
            is_dataframe = False
            categorical_features = self.categorical_features
            (self.is_categorical_,
             self.known_categories_) = _check_categories(categorical_features,
                                                         X)
            if self.is_categorical_ is None:
                self.is_categorical_ = np.zeros(X.shape[1], np.bool)
            self.is_ordinal_ = np.zeros(self.is_categorical_.shape,
                                        np.bool)
            self.columns_ = np.arange(X.shape[1])

        self.variables_ = {}
        self.encoders_ = {}

        self.column_info_ = _get_column_info(X,
                                             self.columns_,
                                             self.is_categorical_,
                                             self.is_ordinal_,
                                             default_encoders=self.default_encoders)
        # include each column as a Variable
        # so that their columns are built if needed

        for col_ in self.columns_:
            self.variables_[col_] = Variable((col_,), str(col_), None) 

        # find possible interactions and other variables

        for term in self.terms:
            if isinstance(term, Variable):
                self.variables_[term] = term
                self.build_columns(term, X, fit=True) # these encoders won't have been fit yet
                for var in term.variables:
                    if var not in self.variables_ and isinstance(var, Variable):
                            self.variables_[var] = var
            # a tuple of variables represents an interaction
            elif term not in self.column_info_ and type(term) == type((1,)): 
                names = []
                column_map = {}
                idx = 0
                for var in term:
                    if var in self.variables_:
                        var = self.variables_[var]
                    cols = self.build_columns(var, X, fit=True) # these encoders won't have been fit yet
                    column_map[var.name] = range(idx, idx + cols.shape[1])
                    idx += cols.shape[1]                 
                    names.append(var.name)
                encoder_ = Interaction(names, column_map)
                self.variables_[term] = Variable(term, ':'.join(n for n in names), encoder_)
            elif term not in self.column_info_:
                raise ValueError('each element in a term should be a Variable or identify a column')

        # build the mapping of terms to columns and column names

        self.column_names_ = {}
        self.column_map_ = {}
        self.terms_ = [self.variables_[t] for t in self.terms]
        
        idx = 0
        if self.intercept:
            self.column_map_['intercept'] = slice(0, 1)
            idx += 1 # intercept will be first column
        
        for term, term_ in zip(self.terms, self.terms_):
            term_df = self.build_columns(term_, X)
            self.column_names_[term] = term_df.columns
            self.column_map_[term] = slice(idx, idx + term_df.shape[1])
            idx += term_df.shape[1]
    
        return self
    
    def transform(self, X):

        """
        Construct parameters for orthogonal
        polynomials in the feature X.

        Parameters
        ----------
        X : array-like
            X on which model matrix will be evaluated.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        D : np.ndarray
            Design matrix.
        """

        check_is_fitted(self)

        dfs = []

        if self.intercept:
            dfs.append(pd.DataFrame({'intercept':np.ones(X.shape[0])}))

        for term_ in self.terms_:
            term_df = self.build_columns(term_, X)
            dfs.append(term_df)

        df = pd.concat(dfs, axis=1)
        if isinstance(X, (pd.Series, pd.DataFrame)):
            return df
        else:
            return df.values

    def fit_encoder(self, var, X):
        """
        Fit an encoder if not already registered
        in `self.encoders_`.

        Return the fit encoder.

        Parameters
        ----------

        var : Variable
            Variable whose encoder will be fit.

        X : array-like
            X on which encoder will be fit.

        """
        if var.encoder in self.encoders_:
            return var.encoder
        else:
            self.encoders_[var] = var.encoder.fit(X)
        
    def build_columns(self, var, X, fit=False):
        """
        Build columns for a Variable from X.

        Parameters
        ----------

        var : Variable
            Variable whose columns will be built.

        X : array-like
            X on which columns are evaluated.

        fit : bool (optional)
            If True, then try to fit encoder.
            Will raise an error if encoder has already been fit.

        """

        if var in self.column_info_:
            var = self.column_info_[var]

        if isinstance(var, Column):
            cols = var.get_columns(X)
            if not var.columns:
                names = [var.name]
            else:
                names = ['{0}[{1}]'.format(var.name, c) for c in var.columns]
        elif isinstance(var, Variable):
            cols = []
            for v in var.variables:
                cur = self.build_columns(v, X, fit=fit)
                cols.append(cur)
            cols = np.column_stack(cols)

            if var.encoder:
                cols = np.column_stack(cols)
                try:
                    check_is_fitted(var.encoder)
                    if fit and var not in self.encoders_:
                        raise ValueError('encoder has already been fit previously')
                except NotFittedError as e:
                    if fit:
                        self.fit_encoder(var, cols)
                    else:
                        raise(e)
                cols = np.column_stack(cols)
                cols = var.encoder.transform(cols)

            if not hasattr(cols, 'columns'):
                names = ['{0}[{1}]'.format(var.name, j) for j in range(cols.shape[1])]
            else:
                names = cols.columns
        else:
            raise ValueError('expecting either a column or a Variable')
        val = pd.DataFrame(np.asarray(cols), columns=names)
        return val

# extracted from method of BaseHistGradientBoosting from
# https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py
# max_bins is ignored
def _check_categories(categorical_features, X):
    """Check and validate categorical features in X

    Return
    ------
    is_categorical : ndarray of shape (n_features,) or None, dtype=bool
        Indicates whether a feature is categorical. If no feature is
        categorical, this is None.
    known_categories : list of size n_features or None
        The list contains, for each feature:
            - an array of shape (n_categories,) with the unique cat values
            - None if the feature is not categorical
        None if no feature is categorical.
    """
    if categorical_features is None:
        return None, None

    categorical_features = np.asarray(categorical_features)

    if categorical_features.size == 0:
        return None, None

    if categorical_features.dtype.kind not in ('i', 'b'):
        raise ValueError("categorical_features must be an array-like of "
                         "bools or array-like of ints.")

    n_features = X.shape[1]

    # check for categorical features as indices
    if categorical_features.dtype.kind == 'i':
        if (np.max(categorical_features) >= n_features
                or np.min(categorical_features) < 0):
            raise ValueError("categorical_features set as integer "
                             "indices must be in [0, n_features - 1]")
        is_categorical = np.zeros(n_features, dtype=bool)
        is_categorical[categorical_features] = True
    else:
        if categorical_features.shape[0] != n_features:
            raise ValueError("categorical_features set as a boolean mask "
                             "must have shape (n_features,), got: "
                             f"{categorical_features.shape}")
        is_categorical = categorical_features

    if not np.any(is_categorical):
        return None, None

    # compute the known categories in the training data. We need to do
    # that here instead of in the BinMapper because in case of early
    # stopping, the mapper only gets a fraction of the training data.
    known_categories = []

    if isinstance(X, (pd.DataFrame, pd.Series)):
        X_list = [X[c] for c in X.columns]
    else:
        X_list = X.T

    for f_idx in range(n_features):
        if is_categorical[f_idx]:
            categories = np.unique(X_list[f_idx])
            missing = []
            for c in categories:
                try:
                    missing.append(np.isnan(c))
                except TypeError: # not a float
                    missing.append(False)
            missing = np.array(missing)
            if missing.any():
                categories = categories[~missing]

        else:
            categories = None
        known_categories.append(categories)

    return is_categorical, known_categories

def _categorical_from_df(df):
    """
    Find
    """
    is_categorical = []
    is_ordinal = []
    for c in df.columns:
        try:
            is_categorical.append(df[c].dtype == 'category')
            is_ordinal.append(df[c].cat.ordered)
        except TypeError:
            is_categorical.append(False)
            is_ordinal.append(False)
    is_categorical = np.array(is_categorical)
    is_ordinal = np.array(is_ordinal)

    return is_categorical, is_ordinal


    
if __name__ == "__main__":

    test_interaction()
    test_ndarray()
    test_dataframe1()
    test_dataframe2()
    test_dataframe3()
    test_dataframe4()
    test_dataframe5()
    test_dataframe6()
    test_dataframe7()
    test_dataframe8()
    test_dataframe9()
    test_dataframe10()
    pass
