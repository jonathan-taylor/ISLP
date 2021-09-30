from collections import namedtuple
from itertools import product
from typing import NamedTuple, Any

import numpy as np, pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import (OneHotEncoder,
                                   OrdinalEncoder)
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

from .columns import (_get_column_info,
                      Column,
                      _categorical_from_df,
                      _check_categories)

from ..transforms import (Poly,
                          BSpline,
                          NaturalSpline,
                          Interaction)

class Variable(NamedTuple):

    """
    An element in a model matrix that will build
    columns from an X.
    """

    variables: tuple
    name: str
    encoder: Any

    def hashable(self):
        return self._replace(variables=tuple(self.variables))

def _hashable(obj):
    if isinstance(obj, Variable):
        return obj.hashable()
    return obj

class ModelMatrix(TransformerMixin, BaseEstimator):

    def __init__(self,
                 terms,
                 intercept=True,
                 categorical_features=None,
                 default_encoders={'categorical': OneHotEncoder(drop='first', sparse=False),
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
            elif term not in self.column_info_:
                # a tuple of variables represents an interaction
                if type(term) == type((1,)): 
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
                elif isinstance(term, Column):
                    self.variables_[term] = Variable((term,), term.name, None)
                else:
                    raise ValueError('each element in a term should be a Variable, Column or identify a column')
                
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
    
    def transform(self, X, y=None):
        """
        Build design on X after fitting.

        Parameters
        ----------
        X : array-like

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.
        """
        return self.build_submodel(X, self.terms_)
    
    # ModelMatrix specific methods

    def build_submodel(self, X, terms):

        """
        Construct design matrix on a
        sequence of terms and X after 
        fitting.

        Parameters
        ----------
        X : array-like
            X on which model matrix will be evaluated.

        Returns
        -------
        df : np.ndarray or pd.DataFrame
            Design matrix.
        """

        check_is_fitted(self)

        dfs = []

        if self.intercept:
            dfs.append(pd.DataFrame({'intercept':np.ones(X.shape[0])}))

        for term_ in terms:
            term_df = self.build_columns(term_, X)
            dfs.append(term_df)

        df = pd.concat(dfs, axis=1)
        if isinstance(X, (pd.Series, pd.DataFrame)):
            df.index = X.index
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
            cols = var.get_columns(X, fit=fit)
            names = var.columns
        elif isinstance(var, Variable):
            cols = []
            for v in var.variables:
                cur = self.build_columns(v, X, fit=fit)
                cols.append(cur)
            cols = np.column_stack(cols)
            
            if var.encoder:
                try:
                    check_is_fitted(var.encoder)
                    if fit and var not in self.encoders_:
                        raise ValueError('encoder has already been fit previously')
                except NotFittedError as e:
                    if fit:
                        self.fit_encoder(var, cols)
                    else:
                        raise(e)
                cols = var.encoder.transform(cols)

            if not hasattr(cols, 'columns'):
                if cols.shape[1] > 1:
                    names = ['{0}[{1}]'.format(var.name, j) for j in range(cols.shape[1])]
                else:
                    names = [var.name]
            else:
                names = cols.columns
        else:
            raise ValueError('expecting either a column or a Variable')
        val = pd.DataFrame(np.asarray(cols), columns=names)
        if isinstance(X, (pd.DataFrame, pd.Series)):
            val.index = X.index
        return val

    def build_sequence(self, X):
        """
        Build implied sequence of submodels 
        based on successively including more terms.
        """

        check_is_fitted(self)

        dfs = []

        if self.intercept:
            df_int = pd.DataFrame({'intercept':np.ones(X.shape[0])})
            if isinstance(X, (pd.Series, pd.DataFrame)):
                df_int.index = X.index
            dfs.append(df_int)

        for term_ in self.terms_:
            term_df = self.build_columns(term_, X)
            if isinstance(X, (pd.Series, pd.DataFrame)):
                term_df.index = X.index

            dfs.append(term_df)

        return (pd.concat(dfs[:i], axis=1) for i in range(1, len(dfs)+1))

def from_encoder(encoder, *variables, name=None):
    """
    Create a Variable
    by applying a transform to some variables.
    
    Parameters
    ----------

    encoder :  transform-like
        Transform obeying sklearn fit/transform convention.

    variabless : column identifier, Column, Variable
        Variables to apply transform to. Could be
        column identifiers or variables: all columns
        will be stacked before encoding.

    name : str (optional)
        Defaults to `str(encoder)`.

    Returns
    -------

    var : Variable
    """

    if name is None:
        name = str(encoder)
    return Variable(variables, name, encoder)

def poly(col, *args, intercept=False, **kwargs):
    """
    Create a polynomial Variable
    for a given column.
    
    Additional `args` and `kwargs`
    are passed to `Poly`.

    Parameters
    ----------

    col : column identifier or Column
        Column to transform.

    intercept : bool
        Include an intercept column.

    Returns
    -------

    var : Variable
    """
    shortname, klass = 'poly', Poly
    if isinstance(col, Column):
        name = col.name
    else:
        name = str(col)
    encoder = klass(*args,
                    intercept=intercept,
                    **kwargs) 
    return from_encoder(encoder,
                        col,
                        name=f'{shortname}({name})')

def ns(col, *args, intercept=False, **kwargs):
    """
    Create a natural spline Variable
    for a given column.
    
    Additional `args` and `kwargs`
    are passed to `NaturalSpline`.

    Parameters
    ----------

    col : column identifier or Column

    intercept : bool
        Include an intercept column.

    Returns
    -------

    var : Variable

    """
    shortname, klass = 'ns', NaturalSpline
    if isinstance(col, Column):
        name = col.name
    else:
        name = str(col)
    encoder = klass(*args,
                    intercept=intercept,
                    **kwargs) 
    return from_encoder(encoder,
                        col,
                        name=f'{shortname}({name})')

def bs(col, *args, intercept=False, **kwargs):
    """
    Create a B-spline Variable
    for a given column.
    
    Additional `args` and `kwargs`
    are passed to `BSpline`.

    Parameters
    ----------

    col : column identifier or Column

    intercept : bool
        Include an intercept column.

    Returns
    -------

    var : Variable

    """
    shortname, klass = 'bs', BSpline
    if isinstance(col, Column):
        name = col.name
    else:
        name = str(col)
    encoder = klass(*args,
                    intercept=intercept,
                    **kwargs) 
    return from_encoder(encoder,
                        col,
                        name=f'{shortname}({name})')

def pca(variables, name, *args, **kwargs):
    """
    Create PCA encoding of features
    from a sequence of variables.
    
    Additional `args` and `kwargs`
    are passed to `PCA`.

    Parameters
    ----------

    variables : [column identifier, Column or Variable]
        Sequence whose columns will be encoded by PCA.

    Returns
    -------

    var : Variable

    """
    shortname, klass = 'pca', PCA
    encoder = klass(*args,
                    **kwargs) 
    return from_encoder(encoder,
                        *variables,
                        name=f'{shortname}({name})')
