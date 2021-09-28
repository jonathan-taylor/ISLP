from collections import namedtuple
from itertools import product
import numpy as np, pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import (OneHotEncoder,
                                   OrdinalEncoder)
from sklearn.exceptions import NotFittedError

from transforms import (Poly,
                        BSpline,
                        NaturalSpline,
                        Interaction)

class ModelMatrix(TransformerMixin, BaseEstimator):

    def __init__(self,
                 intercept=True,
                 terms=None,
                 categorical_features=None,
                 transforms={},
                 default_encoders={'categorical': OneHotEncoder(drop=None, sparse=False),
                                   'ordinal': OrdinalEncoder()},
                 encoders={}):

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
        self.transforms = transforms 
        self.default_encoders = default_encoders
        self.encoders = encoders
        
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
            X_list = X
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
            X_list = X.T

        self.variables_ = {}
        for idx, col_ in enumerate(self.columns_):
            if self.is_categorical_[idx]:
                if col_ not in self.encoders:
                    Xa = np.asarray(X_list[col_]).reshape((-1,1))
                    if not self.is_ordinal_[idx]:
                        encoder_ = clone(self.default_encoders['categorical']).fit(Xa)
                        column_names_ = ['Cat({0})[{1}]'.format(col_, i) for i in
                                         range(len(encoder_.categories_[0]))]
                    else:
                        encoder_ = clone(self.default_encoders['ordinal']).fit(Xa)
                        column_names_ = 'Ord({})'.format(col_)
                else:
                    encoder_ = self.encoders[col_] # don't clone prespecified encoders
                    encoder_ = encoder_.fit(Xa)
                    if hasattr(encoder_, 'columns_'):
                        column_names_ = encoder_.columns_
                    else:
                        Xt = encoder_.transform(Xa)
                        column_names_ = ['Cat({0})[{1}]'.format(col_, i) for i in
                                         range(Xt.shape[1])]
            else:
                encoder_ = None
                column_names_ = [str(col_)]
            self.variables_[col_] = column(col_, column_names_, encoder_)

        if self.terms is None:
            terms_ = [(t,) for t in self.columns_]
        else:
            terms_ = self.terms

        # find possible wilds, elements of terms that are not in self.columns_

        for t in terms_:
            for v in t:
                if v not in self.variables_:
                    if isinstance(v, wild):
                        self.variables_[v] = v
                    else:
                        raise ValueError('each element in a term should be a column identifier or a wild')

        self.terms_ = [frozenset(t) for t in terms_]
        self.terms_ = terms_

        # build the categorical variable transforms

        self.transforms_ = {}
        self.column_names_ = {}
        self.column_map_ = {}

        idx = 0
        if self.intercept:
            self.column_map_['intercept'] = slice(0, 1)
            idx += 1 # intercept will be first column
        
        has_transform = {frozenset(term):term for term in self.transforms.keys()}
        has_transform = {term:term for term in self.transforms.keys()}
        
        for term in self.terms_:
            term_dfs = []
            for col in term:
                var = self.variables_[col]
                term_dfs.append(_build_columns(var, X_list, self.variables_, fit=True))
              
            if term in has_transform.keys():
                term_cols = pd.concat(term_dfs, axis=1)
                self.transforms_[term] = self.transforms[has_transform[term]].fit(term_cols)
                if hasattr(self.transforms_[term], 'columns_'):
                    term_names = self.transforms_[term].columns_
                else:
                    term_names = ['T({0})[{1}]'.format(','.join(*term), i) for i in range(term_cols.shape[1])]
            else:
                if len(term) > 1:
                    inter_df = _interaction(term_dfs)
                    term_cols, term_names = inter_df.values, inter_df.columns
                else:
                    term_cols, term_names = term_dfs[0].values, term_dfs[0].columns
            self.column_names_[term] = term_names
            self.column_map_[term] = slice(idx, idx + term_cols.shape[1])
            idx += term_cols.shape[1]
    
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

        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_list = X
        else:
            X_list = X.T
            
        dfs = []

        if self.intercept:
            dfs.append(pd.DataFrame({'intercept':np.ones(X.shape[0])}))

        has_transform = {frozenset(term):term for term in self.transforms.keys()}
        has_transform = {term:term for term in self.transforms.keys()}

        for term in self.terms_:
            term_dfs = []
            for col in term:
                var = self.variables_[col]
                term_dfs.append(_build_columns(var, X_list, self.variables_))

            if term in self.transforms_:
                term_cols = pd.concat(term_dfs, axis=1)
                term_cols = self.transforms_[term].transform(term_cols)
                term_df = pd.DataFrame(term_cols, columns=self.column_names_[term])
            else:
                if len(term) > 1:
                    term_df = _interaction(term_dfs)
                else:
                    term_df  = term_dfs[0]
            dfs.append(term_df)

        df = pd.concat(dfs, axis=1)
        if isinstance(X, (pd.Series, pd.DataFrame)):
            return df
        else:
            return df.values

def main_effects(columns):
    """
    Make a sequence of terms from `columns`

    Parameters
    ----------

    columns : sequence
       Sequence of column identifiers

    Returns
    -------

    terms : sequence
       Sequence of singleton terms of the form 
       `[(col,) for col in columns]`
    """

    return [(col,) for col in columns]

column = namedtuple('Column', ['idx', 'column_names', 'encoder'])
wild = namedtuple('Wild', ['variables', 'name', 'encoder'])


def _interaction(dfs):

    term_cols = []
    term_names = []
    for cols in product(*[list(df) for df in dfs]):
        term_names.append(':'.join(cols))
        cur_col = np.ones(dfs[0][cols[0]].shape[0])
        for col, df in zip(cols, dfs):
            cur_col *= df[col]
        term_cols.append(cur_col)

    return pd.DataFrame(np.column_stack(term_cols), columns=term_names)

def _build_columns(var, X_list, variables, fit=False):

    if isinstance(var, column):
        col = var.idx
        if var.encoder: # columns with encoders should be fit previously
            Xa = np.asarray(X_list[col]).reshape((-1,1))
            cols = var.encoder.transform(Xa).T
        else:
            cols = [X_list[col]]
        names = var.column_names
    elif isinstance(var, wild):
        cols = []
        for v in var.variables:
            cur = _build_columns(variables[v], X_list, variables)
            cols.append(cur)
        if var.encoder:
            cols = np.column_stack(cols)
            try:
                check_is_fitted(var.encoder)
                if fit:
                    raise ValueError('encoder has already been fit')
            except NotFittedError as e:
                if fit:
                    var.encoder.fit(cols)
                else:
                    raise(e)
            cols = var.encoder.transform(cols).T
        cols = np.column_stack(cols).T
        names = ['{0}[{1}]'.format(var.name, j) for j in range(len(cols))]
    else:
        raise ValueError('expecting either a column or a wild')
    cols = np.column_stack(cols)
    return pd.DataFrame(cols, columns=names)


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


def test_interaction():

    I = Interaction(['V', 'U'],
                    {'V':[0,2],
                     'U':[1,3,5]})
    X = np.random.standard_normal((50,10))
    W = I.fit_transform(X)

    W2 = np.array([X[:,0]*X[:,1],
                   X[:,0]*X[:,3],
                   X[:,0]*X[:,5],
                   X[:,2]*X[:,1],
                   X[:,2]*X[:,3],
                   X[:,2]*X[:,5]]).T
    print(np.linalg.norm(W-W2))
    print(W.columns)

def test_ndarray():
    
    X = np.random.standard_normal((50,5))

    M = ModelMatrix(terms=[(1,), (3,2)])
    clone(M)
    MX = M.fit_transform(X)

    np.testing.assert_allclose(X[:,1], MX[:,1])
    np.testing.assert_allclose(X[:,2] * X[:,3], MX[:,2])
    
def test_dataframe1():
    
    X = np.random.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    
    M = ModelMatrix(terms=[('A',), ('D',), ('D','E')])
    clone(M)
    MX = np.asarray(M.fit_transform(D))

    np.testing.assert_allclose(X[:,0], MX[:,1])
    np.testing.assert_allclose(X[:,3], MX[:,2])
    np.testing.assert_allclose(X[:,3]*X[:,4], MX[:,3])    

def test_dataframe2():
    
    X = np.random.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['V','B','A','D','E'])
    
    M = ModelMatrix(terms=[('A',), ('D',), ('B',), ('D','E'), ('V',)],
                    transforms={('A',):Poly(degree=3),
                                ('B',):BSpline(df=7),
                                ('V',):NaturalSpline(df=6)})
    clone(M)

    MX = M.fit_transform(D)

    X2 = D.copy()
    MX2 = M.transform(D)
    np.testing.assert_allclose(MX, MX2)
    
def test_dataframe3():
    
    X = np.random.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    D['E'] = pd.Categorical(np.random.choice(range(4,8), 50, replace=True))
    
    M = ModelMatrix(terms=[('A',), ('E',), ('D','E')])
    MX = np.asarray(M.fit_transform(D))
    M2 = clone(M)

    DE = pd.get_dummies(D['E'])
    np.testing.assert_allclose(X[:,0], MX[:,1])
    np.testing.assert_allclose(DE, MX[:,2:6])    
    np.testing.assert_allclose(np.asarray(DE) * np.multiply.outer(X[:,3], np.ones(4)), MX[:,-4:])

    X2 = D.copy()
    MX2 = M.transform(D)
    np.testing.assert_allclose(MX, MX2)

def test_dataframe4():
    
    X = np.random.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    D['D'] = pd.Categorical(np.random.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(np.random.choice(range(4,8), 50, replace=True))
    
    M = ModelMatrix(terms=[('A',), ('E',), ('D','E')])
    MX = np.asarray(M.fit_transform(D))

    DE = pd.get_dummies(D['E'])
    np.testing.assert_allclose(X[:,0], MX[:,1])
    np.testing.assert_allclose(DE, MX[:,2:6])    

    # check they agree on copy of dataframe

    X2 = D.copy()
    MX2 = M.transform(D)
    np.testing.assert_allclose(MX, MX2)

def test_dataframe5():
    
    from transforms import Poly

    X = np.random.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    D['D'] = pd.Categorical(np.random.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(np.random.choice(range(4,8), 50, replace=True))
    
    M = ModelMatrix(terms=[('A',), ('E',), ('D','E')], 
                    transforms={('A',):Poly(degree=3)})
    MX = np.asarray(M.fit_transform(D))

    # check they agree on copy of dataframe

    X2 = D.copy()
    MX2 = M.transform(D)
    np.testing.assert_allclose(MX, MX2)
    
def test_dataframe6():
    
    from transforms import Poly

    X = np.random.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    W = wild(('A','E'), 'AE', None)
    D['D'] = pd.Categorical(np.random.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(np.random.choice(range(4,8), 50, replace=True))
    
    M = ModelMatrix(terms=[('A',), (W,), (W, 'D',)])
    MX = M.fit_transform(D)

    MX = np.asarray(MX)

def test_dataframe7():
    
    from transforms import Poly

    X = np.random.standard_normal((50,6))
    D = pd.DataFrame(X, columns=['AA','Bbbb','C','Ddd','Y','Eee'])
    D['Ddd'] = pd.Categorical(np.random.choice(['a','b','c'], 50, replace=True))
    D['Eee'] = pd.Categorical(np.random.choice(range(4,8), 50, replace=True))
        
    M = ModelMatrix(terms=main_effects(D.columns.drop(['Y','C'])))
    MX = M.fit_transform(D)
    print(MX.columns)
    MX = np.asarray(MX)

def test_dataframe8():
    
    from transforms import Poly

    X = np.random.standard_normal((50,6))
    D = pd.DataFrame(X, columns=['A','B','C','D','Y','E'])
    D['D'] = pd.Categorical(np.random.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(np.random.choice(range(4,8), 50, replace=True))
    
    poly =  Poly(degree=3)
    # raises a ValueError because poly will have been already fit -- need new instance of Poly
    W = wild(('A',), 'poly(A)', poly)
    M = ModelMatrix(terms=main_effects(D.columns.drop(['Y','C'])) + [(W,'E')])
    MX = M.fit_transform(D)

    print(MX.columns)
    MX = np.asarray(MX)

def test_dataframe9():
    
    from transforms import Poly

    X = np.random.standard_normal((50,6))
    D = pd.DataFrame(X, columns=['A','B','C','D','Y','E'])
    D['D'] = pd.Categorical(np.random.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(np.random.choice(range(4,8), 50, replace=True))
    
    poly =  Poly(degree=3)
    # raises a ValueError because poly will have been already fit -- need new instance of Poly
    W = wild(('A',), 'poly(A)', poly)
    U = wild(('B',), 'poly(B)', poly)
    M = ModelMatrix(terms=main_effects(D.columns.drop(['Y','C'])) + [(W,), (U,)])
    MX = M.fit_transform(D)

    print(MX.columns)
    MX = np.asarray(MX)

def test_dataframe10():
    
    from transforms import Poly

    X = np.random.standard_normal((50,5))
    D = pd.DataFrame(X, columns=['A','B','C','D','E'])
    W = wild(('A','E'), 'AE', None)
    U = wild((W, 'C'), 'WC', None)
    D['D'] = pd.Categorical(np.random.choice(['a','b','c'], 50, replace=True))
    D['E'] = pd.Categorical(np.random.choice(range(4,8), 50, replace=True))
    
    M = ModelMatrix(terms=[('A',), ('E',), ('C',), (W,), (W, 'D',), (U,)])
    MX = M.fit_transform(D)
    print(MX.columns)
    MX = np.asarray(MX)

    V = MX[:,-6:]
    V2 = np.column_stack([MX[:,M.column_map_[('A',)]],
                          MX[:,M.column_map_[('E',)]],
                          MX[:,M.column_map_[('C',)]]])
    print(np.linalg.norm(V-V2))
    
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
    test_dataframe10()
    pass
