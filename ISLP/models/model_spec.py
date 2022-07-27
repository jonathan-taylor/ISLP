from collections import namedtuple
from itertools import product
from typing import NamedTuple, Any
from copy import copy

import numpy as np, pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import (OneHotEncoder,
                                   OrdinalEncoder,
                                   StandardScaler)
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.exceptions import NotFittedError
from joblib import hash as joblib_hash

from .columns import (_get_column_info,
                      Column,
                      _categorical_from_df,
                      _check_categories)

from ..transforms import (Poly,
                          BSpline,
                          NaturalSpline,
                          Interaction)

DOCACHE = False

class Variable(NamedTuple):

    """
    An element in a model matrix that will build
    columns from an X.
    """

    variables: tuple
    name: str
    encoder: Any
    use_transform: bool=True   # if False use the predict method
    pure_columns: bool=False
    override_encoder_colnames: bool=True
    
#### contrast specific code

class Contrast(TransformerMixin, BaseEstimator):
    """
    Contrast encoding for categorical variables.
    """

    def __init__(self,
                 method='drop',
                 drop_level=None):

        self.method = method
        self.drop_level = drop_level
        
    def fit(self, X):

        Xa = np.asarray(X).reshape((-1,1))
        self.encoder_ = OneHotEncoder(drop=None,
                                      sparse=False).fit(Xa)
        cats = self.encoder_.categories_[0]
        column_names = [str(n) for n in cats]

        if X.dtype == 'category':
            Xcats = list(X.dtype.categories)
        else:
            Xcats = copy(column_names)
        if self.drop_level is None:
            if self.method == 'drop': # these defaults consistent with R
                drop_level = Xcats[0]
            elif self.method == 'sum':
                drop_level = Xcats[-1]
        else:
            drop_level = self.drop_level

        if self.method in ['drop', 'sum']:
            drop_idx = column_names.index(str(drop_level))
            Xcats.remove(drop_level)
            column_names.pop(drop_idx)

        colmap = [column_names.index(str(j)) for j in Xcats]
        cols = self.encoder_.transform(Xa)

        if self.method == 'drop':
            self.columns_ = [column_names[j] for j in colmap]
            self.contrast_matrix_ = np.identity(len(cats))
            keep = np.ones(len(cats), np.bool)
            keep[drop_idx] = 0
            self.contrast_matrix_ = self.contrast_matrix_[:,keep]
            self.contrast_matrix_ = self.contrast_matrix_[:,colmap]            
        elif self.method == 'sum':
            self.columns_ = [column_names[j] for j in colmap]
            self.contrast_matrix_ = np.zeros((len(cats), len(cats)-1))
            self.contrast_matrix_[:drop_idx][:,:drop_idx] = np.identity(drop_idx)
            self.contrast_matrix_[drop_idx+1:,drop_idx:] = np.identity(len(cats) - drop_idx - 1)
            self.contrast_matrix_[drop_idx] = -1
            self.contrast_matrix_ = self.contrast_matrix_[:,colmap]
        elif callable(self.method):
            self.contrast_matrix_ = self.method(len(cats))
        elif self.method is None:
            self.contrast_matrix_ = np.identity(len(cats))
            self.columns_ = column_names
        else:
            raise ValueError('method must be one of ["drop", "sum", None] or a callable' +
                             'that returns a contrast matrix and column names given the number' +
                             ' of levels')
        return self

    def transform(self, X):
        if not hasattr(self, 'encoder_'):
            self.fit(X)
        Xa = np.asarray(X).reshape((-1,1))
        D = self.encoder_.transform(Xa)
        value = D.dot(self.contrast_matrix_)

        if isinstance(X, (pd.DataFrame, pd.Series)):
            df = pd.DataFrame(value, columns=self.columns_)
            df.index = X.index
            return df
        return value

def contrast(col, method, drop_level=None):
    """
    Create PCA encoding of features
    from a sequence of variables.
    
    Additional `args` and `kwargs`
    are passed to `PCA`.

    Parameters
    ----------

    col: column identifier

    method: 

    drop_level: level identifier

    Returns
    -------

    var : Variable

    """

    if isinstance(col, Column):
        col = col.idx
        name = col.name
    else:
        name = str(col)
    encoder = Contrast(method,
                       drop_level=drop_level)
    return Column(col,
                  name,
                  is_categorical=True,
                  encoder=encoder)

class ModelSpec(TransformerMixin, BaseEstimator):

    def __init__(self,
                 terms,
                 intercept=True,
                 categorical_features=None,
                 default_encoders={'categorical': Contrast(method='drop'),
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
            self.variables_[col_] = Variable((col_,), str(col_), None, pure_columns=True) 

        # find possible interactions and other variables

        tmp_cache = {}

        for term in self.terms:
            if isinstance(term, Variable):
                self.variables_[term] = term
                self.build_columns(X,
                                   term,
                                   col_cache=tmp_cache,
                                   fit=True) # these encoders won't have been fit yet
                for var in term.variables:
                    if var not in self.variables_ and isinstance(var, Variable):
                            self.variables_[var] = var
            elif term not in self.column_info_:
                # a tuple of variables represents an interaction
                if type(term) == type((1,)): 
                    names = []
                    column_map = {}
                    column_names = {}
                    idx = 0
                    for var in term:
                        if var in self.variables_:
                            var = self.variables_[var]
                        cols, cur_names = self.build_columns(X,
                                                             var,
                                                             col_cache=tmp_cache,
                                                             fit=True) # these encoders won't have been fit yet
                        column_map[var.name] = range(idx, idx + cols.shape[1])
                        column_names[var.name] = cur_names
                        idx += cols.shape[1]                 
                        names.append(var.name)
                    encoder_ = Interaction(names, column_map, column_names)
                    self.variables_[term] = Variable(term, ':'.join(n for n in names), encoder_)
                elif isinstance(term, Column):
                    self.variables_[term] = Variable((term,), term.name, None, pure_columns=True)
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
            term_df, term_names = self.build_columns(X, term_)
            self.column_names_[term] = term_names
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
    
    # ModelSpec specific methods

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

        col_cache = {}  # avoid recomputing the same columns

        if self.intercept:
            df = pd.DataFrame({'intercept':np.ones(X.shape[0])})
            if isinstance(X, (pd.Series, pd.DataFrame)):
                df.index = X.index
            dfs.append(df)

        for term_ in terms:
            term_df = self.build_columns(X,
                                         term_,
                                         col_cache=col_cache,
                                         fit=False)[0]
            dfs.append(term_df)

        if len(dfs):
            if isinstance(X, (pd.Series, pd.DataFrame)):
                df = pd.concat(dfs, axis=1)
                df.index = X.index
                return df
            else:
                return np.column_stack(dfs)
        else:  # return a 0 design
            zero = np.zeros(X.shape[0])
            if isinstance(X, (pd.Series, pd.DataFrame)):
                df = pd.DataFrame({'zero': zero})
                df.index = X.index
                return df
            else:
                return zero

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

        if var.encoder not in self.encoders_:
            self.encoders_[var] = var.encoder.fit(X)
            
    def build_columns(self, X, var, col_cache={}, fit=False):
        """
        Build columns for a Variable from X.

        Parameters
        ----------

        X : array-like
            X on which columns are evaluated.

        var : Variable
            Variable whose columns will be built.

        col_cache: 
            Dict where columns will be stored --
            if `var.name` in `col_cache` then just
            returns those columns.

        fit : bool (optional)
            If True, then try to fit encoder.
            Will raise an error if encoder has already been fit.

        """

        if var in self.column_info_:
            var = self.column_info_[var]

        if DOCACHE and joblib_hash([var, X]) in col_cache:
            return col_cache[joblib_hash([var, X])]
        
        if isinstance(var, Column):
            if DOCACHE:
                if joblib_hash([var, X]) not in col_cache:
                    cols, names = var.get_columns(X, fit=fit)
                    col_cache[joblib_hash([var, X])] = cols, names
                cols, name = col_cache[joblib_hash([var, X])]
            else:
                cols, names = var.get_columns(X, fit=fit)
        elif isinstance(var, Variable):
            cols = []
            names = []
            for v in var.variables:
                cur, cur_names = self.build_columns(X,
                                                    v,
                                                    col_cache=col_cache,
                                                    fit=fit)
                cols.append(cur)
                names.extend(cur_names)
            cols = np.column_stack(cols)
            if len(names) != cols.shape[1]:
                names = ['{0}[{1}]'.format(var.name, j) for j in range(cols.shape[1])]

            if var.encoder:
                try:
                    check_is_fitted(var.encoder)
                    if fit and var not in self.encoders_:
                        raise ValueError('encoder has already been fit previously')
                except NotFittedError as e:
                    if fit:
                        self.fit_encoder(var, pd.DataFrame(np.asarray(cols),
                                                           columns=names))
                    # known issue with Pipeline
                    # https://github.com/scikit-learn/scikit-learn/issues/18648
                    elif isinstance(var.encoder, Pipeline):  
                        check_is_fitted(var.encoder, 'n_features_in_') 
                    else:
                        raise(e)
                except Exception as e:  # was not the NotFitted
                    raise ValueError(e)
                if var.use_transform:
                    cols = var.encoder.transform(cols)
                else:
                    cols = var.encoder.predict(cols)
                if hasattr(var.encoder, 'columns_') and not var.override_encoder_colnames:
                    names = var.encoder.columns_
                else:
                    if cols.ndim > 1 and cols.shape[1] > 1:
                        names = ['{0}[{1}]'.format(var.name, j) for j in range(cols.shape[1])]
                    else:
                        names = [var.name]

            
        else:
            raise ValueError('expecting either a column or a Variable')
        val = pd.DataFrame(np.asarray(cols), columns=names)

        if isinstance(X, (pd.DataFrame, pd.Series)):
            val.index = X.index

        if DOCACHE:
            col_cache[joblib_hash([var.name, X])] = (val, names)
        return val, names

    def build_sequence(self, X, anova_type='sequential'):
        """
        Build implied sequence of submodels 
        based on successively including more terms.
        """

        check_is_fitted(self)

        col_cache = {}  # avoid recomputing the same columns

        dfs = []

        if self.intercept:
            df_int = pd.DataFrame({'intercept':np.ones(X.shape[0])})
            if isinstance(X, (pd.Series, pd.DataFrame)):
                df_int.index = X.index
            dfs.append(df_int)
        else:
            df_int = pd.DataFrame({'zero':np.zeros(X.shape[0])})
            if isinstance(X, (pd.Series, pd.DataFrame)):
                df_int.index = X.index
            dfs.append(df_int)

        for term_ in self.terms_:
            term_df, _  = self.build_columns(X,
                                             term_,
                                             col_cache=col_cache,
                                             fit=False)
            if isinstance(X, (pd.Series, pd.DataFrame)):
                term_df.index = X.index

            dfs.append(term_df)

        if anova_type == 'sequential':
            if isinstance(X, (pd.Series, pd.DataFrame)):
                return (pd.concat(dfs[:i], axis=1) for i in range(1, len(dfs)+1))
            else:
                return (np.column_stack(dfs[:i]) for i in range(1, len(dfs)+1))
        elif anova_type == 'drop':
            if isinstance(X, (pd.Series, pd.DataFrame)):
                return (pd.concat([dfs[j] for j in range(len(dfs)) if j != i], axis=1) for i in range(len(dfs)))
            else:
                return (np.column_stack([dfs[j] for j in range(len(dfs)) if j != i]) for i in range(len(dfs)))
        else:
            raise ValueError('anova_type must be one of ["sequential", "drop"]')

def derived_variable(*variables, encoder=None, name=None, use_transform=True):
    """
    Create a Variable, optionally
    applying an encoder to the stacked columns.
    
    Parameters
    ----------

    variables : column identifier, Column, Variable
        Variables to apply transform to. Could be
        column identifiers or variables: all columns
        will be stacked before encoding.

    name : str (optional)
        Defaults to `str(encoder)`.

    encoder :  transform-like (optional)
        Transform obeying sklearn fit/transform convention.

    Returns
    -------

    var : Variable
    """

    if name is None:
        name = str(encoder)
    var = Variable(variables, name, encoder, use_transform=use_transform)
    return var

def poly(col, *args, intercept=False, name=None, **kwargs):
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

    name : str (optional)
        Defaults to one derived from col.

    Returns
    -------

    var : Variable
    """
    shortname, klass = 'poly', Poly
    encoder = klass(*args,
                    intercept=intercept,
                    **kwargs) 
    if name is None:
        if isinstance(col, Column):
            name = col.name
        else:
            name = str(col)

        _args = _argstring(*args, **kwargs)
        if _args:
            name = ', '.join([name, _args])

        name = f'{shortname}({name})'

    return derived_variable(col,
                            name=name,
                            encoder=encoder)

def ns(col, *args, intercept=False, name=None, **kwargs):
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

    name : str (optional)
        Defaults to one derived from col.

    Returns
    -------

    var : Variable

    """
    shortname, klass = 'ns', NaturalSpline
    if name is None:
        if isinstance(col, Column):
            name = col.name
        else:
            name = str(col)

        _args = _argstring(*args, **kwargs)
        if _args:
            name = ', '.join([name, _args])

        name = f'{shortname}({name})'
    encoder = klass(*args,
                    intercept=intercept,
                    **kwargs) 
    return derived_variable(col,
                            name=name,
                            encoder=encoder)

def bs(col, *args, intercept=False, name=None, **kwargs):
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

    name : str (optional)
        Defaults to one derived from col.

    Returns
    -------

    var : Variable

    """
    shortname, klass = 'bs', BSpline
    if name is None:
        if isinstance(col, Column):
            name = col.name
        else:
            name = str(col)

        _args = _argstring(*args, **kwargs)
        if _args:
            name = ', '.join([name, _args])

        name = f'{shortname}({name})'
    encoder = klass(*args,
                    intercept=intercept,
                    **kwargs) 
    return derived_variable(col,
                            name=name,
                            encoder=encoder)

def pca(variables, name, *args, scale=False, **kwargs):
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
    if scale:
        scaler = StandardScaler(with_mean=True,
                                with_std=True)
        encoder = make_pipeline(scaler, encoder)

    _args = _argstring(*args, **kwargs)
    if _args:
        name = ', '.join([name, _args])

    return derived_variable(*variables,
                            name=f'{shortname}({name})',
                            encoder=encoder)

def clusterer(variables, name, transform, scale=False):
    """
    Create PCA encoding of features
    from a sequence of variables.
    
    Additional `args` and `kwargs`
    are passed to `PCA`.

    Parameters
    ----------

    variables : [column identifier, Column or Variable]
        Sequence whose columns will be encoded by PCA.

    name: str
        name for the Variable

    transform: Transformer
        A transform with a `predict` method.

    Returns
    -------

    var : Variable

    """

    if scale:
        scaler = StandardScaler(with_mean=True,
                                with_std=True)
        encoder = make_pipeline(scaler, transform)
    else:
        encoder = transform

    intermed = Variable((derived_variable(*variables,
                                          name='cluster_intermed',
                                          encoder=encoder,
                                          use_transform=False),),
                            name=f'Cat({encoder}({name}))',
                            encoder=Contrast(method='drop'))

    return intermed

def _argstring(*args, **kwargs):
    return ', '.join([str(a) for a in args]) + ', '.join([f'{k}={v}' for k, v in kwargs.items()])
