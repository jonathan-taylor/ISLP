from itertools import product
import numpy as np, pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from scipy.interpolate import splev

class Poly(TransformerMixin, BaseEstimator):

    def __init__(self,
                 degree=1,
                 intercept=False,
                 raw=False):

        '''

        Parameters
        ----------

        degree : int
            Degree of polynomial.

        intercept : bool (optional)
            Include a column for intercept?

        raw : bool (optional)
            If False, perform a QR decomposition on the resulting
            matrix of powers of centered and / or scaled features.
        '''
        
        self.degree = degree
        self.raw = raw
        self.intercept = intercept

    def fit(self, X, y=None):

        """
        Construct parameters for orthogonal
        polynomials in the feature X.

        Parameters
        ----------
        X : array-like

            Single feature on which polynomial features are fit
            by first demeaning and / or scaling, followed
            by QR decomposition on the matrix of powers of resulting
            feature.

        """
        
        X_orig = X
        X = np.squeeze(np.asarray(X).astype(float).copy())
        n = X.shape[0]
        if X.reshape(-1).shape[0] != n:
            raise ValueError('expecting a single column feature')

        self.mean_ = X.mean()
        
        if not self.raw:
            powX = np.power.outer(X - self.mean_, np.arange(0, self.degree+1))
            # Following R's poly construction
            Q, R = np.linalg.qr(powX)
            Z = Q * np.diag(R)[None,:]
            self.norm2_ = (Z**2).sum(0)
            self.alpha_ = ((X[:,None] * Z**2).sum(0) / self.norm2_)[:self.degree]
            self.norm2_ = np.hstack([1, self.norm2_])

        # for pandas 

        self.columns_ = range(self.degree+1)
        if not self.intercept:
            self.columns_ = self.columns_[:-1]

        if isinstance(X_orig, (pd.Series, pd.DataFrame)):
            if isinstance(X_orig, pd.Series):
                name = X_orig.name
            else:
                name = X_orig.columns[0]
            self.columns_ = ['{0}[{1}]'.format(self, d)
                             for d in self.columns_]

        return self
    
    def transform(self, X):

        """
        Construct parameters for orthogonal
        polynomials in the feature X.

        Parameters
        ----------
        X : array-like
            X on which features will be evaluated.

        Returns
        -------
        XP : np.ndarray
            Evaluated polynomial features.
        """
        check_is_fitted(self)

        X_orig = X
        
        X = np.squeeze(np.asarray(X).astype(float).copy())
        n = X.shape[0]
        if X.reshape(-1).shape[0] != n:
            raise ValueError('expecting a single column feature')

        if not self.raw:
            Z = np.ones((n, self.degree+1))
            Z[:,1] = X - self.alpha_[0]

            if self.degree > 1:
                for i in range(1, self.degree):
                    Z[:,i+1] = ((X - self.alpha_[i]) * Z[:,i] -
                                self.norm2_[i+1] / self.norm2_[i] * Z[:,i-1])
            Z /= np.sqrt(self.norm2_[1:])
            powX = Z
        else:
            powX = np.power.outer(X, np.arange(0, self.degree+1))

        if not self.intercept:
            powX = powX[:,1:]

        if isinstance(X_orig, (pd.Series, pd.DataFrame)):
            df = pd.DataFrame(powX,
                              columns=self.columns_)
            df.index = X_orig.index
            return df
        else:
            return powX

#### Interaction transform

class Interaction(TransformerMixin, BaseEstimator):

    def __init__(self,
                 variables,
                 columns,
                 column_names):

        '''

        Form the tensor product interaction
        from a group of columns.

        Parameters
        ----------

        variables : sequence

        columns : dict
            Mapping from variable names to columns

        '''
        
        self.variables = variables
        self.columns = columns
        self.column_names = column_names
        self.column_names_ = {} 
        self.columns_ = []

        variable_names = []
        for variable in self.variables:
            cols = self.columns[variable]
            col_names = ['{0}[{1}]'.format(variable, i) for i in range(len(self.columns[variable]))]
            if variable in column_names:
                col_names = [str(c) for c in column_names[variable]]
            if len(cols) > 1:
                variable_names.append(col_names)
            else:
                variable_names.append(['{0}'.format(variable)])

        for names in product(*variable_names):
            self.columns_.append(':'.join(names))

    def fit(self, X, y=None):

        """
        Nothing to be computed for the fit.

        Parameters
        ----------
        X : array-like

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        """
        
        return self
    
    def transform(self, X):

        """
        Construct parameters for orthogonal
        polynomials in the feature X.

        Parameters
        ----------
        X : array-like
            X on which features will be evaluated.

        Returns
        -------
        XP : np.ndarray
            Evaluated polynomial features.
        """
        check_is_fitted(self)

        X_orig = X
        X = np.asarray(X)

        X_lists = []
        for variable in self.variables:
            X_lists.append(X[:,self.columns[variable]].T)

        cols = []
        for X_list in product(*X_lists):
            col = np.ones(X.shape[0])
            for x in X_list:
                col *= x
            cols.append(col)

        df = pd.DataFrame(np.column_stack(cols),
                          columns=self.columns_)
        if isinstance(X_orig, (pd.DataFrame, pd.Series)):
            df.index = X_orig.index

        return df

#### Spline specific code

def _onehot(p, j):
    v = np.zeros(p)
    v[j] = 1
    return v

def _splevf(x,
            tk,
            der=0,
            ext=0):
    """
    Full B-spline matrix at x
    """
    x = np.asarray(x)
    knots, degree = tk
    nbasis = len(knots) - (degree + 1)
    tcks = [(knots, _onehot(nbasis, j), degree) for
            j in range(nbasis)]
    return np.column_stack([splev(x, tck, ext=ext, der=der)
                            for tck in tcks])

def _splev_taylor(x,
                  basept,
                  tk,
                  order):
    """
    Taylor expansion of B-spline at x around basept
    """
    x = np.asarray(x)
    derivs = np.array([_splevf([basept],
                               tk,
                               der=o,
                               ext=0).reshape(-1)
                       for o in range(order+1)])
    polys = np.power.outer(x-basept,
                           np.arange(order+1))
    fact = np.concatenate([[1], np.cumprod(np.arange(1, order+1))])
    polys /= fact[None,:]

    return polys.dot(derivs)

class BSpline(TransformerMixin, BaseEstimator):

    def __init__(self,
                 degree=3,
                 intercept=False,
                 lower_bound=None,
                 upper_bound=None,
                 internal_knots=None,
                 df=None,
                 ext=0):

        '''

        Parameters
        ----------

        degree : int
            Degree of polynomial.

        intercept : bool
            If False, a column of basis is dropped so that by
            adding an intercept column design stays full rank.

        lower_bound : float
            Lower boundary not. Will be set to minimal value if not supplied.

        upper_bound : float
            Upper boundary not. Will be set to maximal value if not supplied.

        internal_knots : array-like
            Optional internal knots of B-spline. Will be set to
            appropriate quantiles based on `df`.

        df : int
            Degrees of freedom for regression. Defaults to `degree + intercept`.
        ext : int
            How B-splines are to be extended beyond the boundary using
            `scipy.interpolate.splev`.

        '''
        
        self.degree = degree
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.internal_knots = internal_knots
        self.df = df
        self.ext = ext
        self.intercept = intercept
        
    def fit(self, X, y=None):

        """
        Compute knots for B-spline representation.

        Parameters
        ----------
        X : array-like

            Single feature on which B-spline will be evaluated.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        """
        
        X_orig = X
        X = np.squeeze(np.asarray(X).astype(float).copy())
        n = X.shape[0]
        if X.reshape(-1).shape[0] != n:
            raise ValueError('expecting a single column feature')

        order = self.degree + 1

        if self.lower_bound is None:
            self.lower_bound = X.min()

        if self.upper_bound is None:
            self.upper_bound = X.max()
            
        if self.df is None:
            self.df = order - 1 + self.intercept

        if self.df < order - 1 + self.intercept:
            raise ValueError('df must be greater than or equal to %d' % (order - 1 + self.intercept))
        ninternal = self.df - (order - 1 + self.intercept)
        if self.internal_knots is None:
            if self.df is None:
                raise ValueError('if internal_knots unspecified then df must be specified')
            percs = 100*np.linspace(0, 1, ninternal+2)[1:-1]
            internal_knots = np.percentile(X, percs)
        else:
            internal_knots = self.internal_knots

        if self.lower_bound >= self.upper_bound:
            raise ValueError('lower_bound must be smaller than upper_bound')
        
        self.internal_knots_ = internal_knots

        self.knots_ = np.sort(np.concatenate([[self.lower_bound]*order,
                                              [self.upper_bound]*order,
                                              internal_knots]))
        if self.knots_[0] < self.lower_bound:
            raise ValueError('internal_knots should be greater than our equal to lower_bound')
        if self.knots_[-1] > self.upper_bound:
            raise ValueError('internal_knots should be less than our equal to upper_bound')
        
        self.boundary_knots_ = [self.lower_bound, self.upper_bound]

        # for pandas 

        self.nbasis_ = len(self.knots_) - (self.degree + 1)
        self.columns_ = range(self.nbasis_)

        if not self.intercept:
            self.columns_ = self.columns_[:-1]

        if isinstance(X_orig, (pd.Series, pd.DataFrame)):
            if isinstance(X_orig, pd.Series):
                name = X_orig.name
            else:
                name = X_orig.columns[0] # a pd.DataFrame
            self.columns_ = ['{0}[{1}]'.format(self, d)
                             for d in self.columns_]
        return self
    
    def transform(self, X):

        """
        Construct design for B-splines
        in the feature X.

        Parameters
        ----------
        X : array-like
            X on which features will be evaluated.

        Returns
        -------
        XP : np.ndarray
            Evaluated polynomial features.
        """
        check_is_fitted(self)

        X_orig = X
        X = np.squeeze(np.asarray(X).astype(float).copy())
        n = X.shape[0]
        if X.reshape(-1).shape[0] != n:
            raise ValueError('expecting a single column feature')

        value = _splevf(X,
                        (self.knots_, self.degree),
                        der=0,
                        ext=self.ext)
        if not self.intercept:
            value = value[:,1:]
        columns_ = self.columns_

        if isinstance(X_orig, (pd.Series, pd.DataFrame)):
            df = pd.DataFrame(value,
                              columns=columns_)
            df.index = X_orig.index
            return df
        else:
            return value

class NaturalSpline(TransformerMixin, BaseEstimator):

    def __init__(self,
                 intercept=False,
                 lower_bound=None,
                 upper_bound=None,
                 internal_knots=None,
                 df=None,
                 ext=0):

        '''

        Natural cubic spline.

        Parameters
        ----------

        intercept : bool
            If False, a column of basis is dropped so that by
            adding an intercept column design stays full rank.

        lower_bound : float
            Lower boundary not. Will be set to minimal value if not supplied.

        upper_bound : float
            Upper boundary not. Will be set to maximal value if not supplied.

        internal_knots : array-like
            Optional internal knots of B-spline. Will be set to
            appropriate quantiles based on `df`.

        df : int
            Degrees of freedom for regression. Defaults to `3 + intercept`.

        ext : int
            How B-splines are to be extended beyond the boundary using
            `scipy.interpolate.splev`.

        '''
        
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.internal_knots = internal_knots
        self.df = df
        self.intercept = intercept
        self.ext = ext
        
    def fit(self, X, y=None):

        """
        Compute knots for natural spline representation.

        Parameters
        ----------
        X : array-like

            Single feature on which B-spline will be evaluated.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        """
        
        X_orig = X
        X = np.squeeze(np.asarray(X).astype(float).copy())
        n = X.shape[0]
        if X.reshape(-1).shape[0] != n:
            raise ValueError('expecting a single column feature')

        order = 4

        lower_bound = self.lower_bound
        if lower_bound is None:
            lower_bound = X.min()

        upper_bound = self.upper_bound
        if upper_bound is None:
            upper_bound = X.max()
            
        if self.df is None:
            self.df = order - 1 + self.intercept - 2 # -2 for constraints

        if self.df < order:
            raise ValueError('df must be greater than or equal to %d' % (order - 1 + self.intercept - 2))
        ninternal = self.df - (order - 1 + self.intercept - 2)
        if self.internal_knots is None:
            if self.df is None:
                raise ValueError('if internal_knots unspecified then df must be specified')
            percs = 100*np.linspace(0, 1, ninternal + 2)[1:-1]
            internal_knots = np.percentile(X, percs)
        else:
            internal_knots = np.asarray(sorted(self.internal_knots))
            if ninternal != internal_knots.shape[0]:
                raise ValueErorr('conflict between df and internal_knots')

        if lower_bound >= upper_bound:
            raise ValueError('lower_bound must be smaller than upper_bound')
        
        self.internal_knots_ = internal_knots
        self.knots_ = np.sort(np.concatenate([[lower_bound]*order,
                                              [upper_bound]*order,
                                              internal_knots]))
        if self.knots_[0] < lower_bound:
            raise ValueError('internal_knots should be greater than our equal to lower_bound')
        if self.knots_[-1] > upper_bound:
            raise ValueError('internal_knots should be less than our equal to upper_bound')
        
        self.boundary_knots_ = [lower_bound, upper_bound]
        self.nbasis_ = self.knots_.shape[0] - order

        # now enforce constraint that 2nd derivative at two boundary knots is 0
        # this depends on the X that is passed to fit!

        boundary_vals = _splevf(self.boundary_knots_,
                                (self.knots_, 3),
                                der=2,
                                ext=0)
        if not self.intercept:
            boundary_vals = boundary_vals[:,1:]
        Q_, R_ = np.linalg.qr(boundary_vals.T,
                              mode='complete')
        self.Qproj_ = Q_[:,2:]

        # for pandas 

        self.columns_ = range(self.nbasis_-2) # two constraints

        if not self.intercept:
            self.columns_ = self.columns_[:-1]

        if isinstance(X_orig, (pd.Series, pd.DataFrame)):
            if isinstance(X_orig, pd.Series):
                name = X_orig.name
            else:
                name = X_orig.columns[0] # a pd.DataFrame
            self.columns_ = ['{0}[{1}]'.format(self, d)
                             for d in self.columns_]
        return self
    
    def transform(self, X):

        """
        Construct design for B-splines
        in the feature X.

        Parameters
        ----------
        X : array-like
            X on which features will be evaluated.

        Returns
        -------
        XP : np.ndarray
            Evaluated polynomial features.
        """

        check_is_fitted(self)

        lower_bound, upper_bound = self.boundary_knots_

        X_orig = X
        X = np.squeeze(np.asarray(X).astype(float).copy())
        n = X.shape[0]
        if X.reshape(-1).shape[0] != n:
            raise ValueError('expecting a single column feature')

        value = np.empty((X.shape[0], self.nbasis_))
        inside = (X >= lower_bound) * (X <= upper_bound)
        if np.any(inside):
            value[inside] = _splevf(X[inside],
                                    (self.knots_, 3),
                                    der=0,
                                    ext=self.ext)

        to_the_right = (X > upper_bound)
        if np.any(to_the_right):
            value[to_the_right] = _splev_taylor(X[to_the_right],
                                                upper_bound,
                                                (self.knots_, 3),
                                                order=1)

        to_the_left = (X < lower_bound)
        if np.any(to_the_left):
            value[to_the_left] = _splev_taylor(X[to_the_left],
                                                lower_bound,
                                                (self.knots_, 3),
                                                order=1)

        # possibly drop column for intercept

        if not self.intercept:
            value = value[:,1:]

        # map onto the natural splines

        value = value.dot(self.Qproj_)
        if isinstance(X_orig, (pd.Series, pd.DataFrame)):
            df = pd.DataFrame(value,
                              columns=self.columns_)
            df.index = X_orig.index
            return df
        else:
            return value

