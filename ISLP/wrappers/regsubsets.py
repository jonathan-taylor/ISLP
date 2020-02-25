from functools import partial

import numpy as np, pandas as pd

import rpy2.robjects as rpy

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_validate, check_cv
from sklearn.metrics import mean_squared_error
from sklearn.externals.joblib.parallel import Parallel, delayed

class Subset(BaseEstimator, RegressorMixin): 

    def __init__(self, 
                 formula_str, 
                 nvar, 
                 method='exhaustive'):

        self.formula_str = formula_str
        self.formula = rpy.r(formula_str)
        self.method = method
        self.nvar = nvar

    def fit(self, X, y):
        """Fit best subsets regression for a given `nvar`

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training data. Pass directly as Fortran-contiguous data
            to avoid unnecessary memory duplication. If y is mono-output,
            X can be sparse.

        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values
        """
        rpy.pandas2ri.activate() # for data frame conversion
        D = pd.concat([X, y], axis=1) # reconstitute data
        _regfit = rpy.r['regsubsets'](self.formula, 
                                      data=D, 
                                      method=self.method,
                                      nvmax=self.nvar)
        self._which = rpy.r['summary'](_regfit).rx2('which').astype(np.bool)
        _names = _regfit.rx2('xnames')
        self._nz_coef = rpy.r['coef'](_regfit, self.nvar)
        self.coef_ = pd.Series(self._nz_coef, index=_names[self._which[self.nvar-1]])
        rpy.pandas2ri.deactivate() 

    def predict(self, X):
        rpy.pandas2ri.activate() # for data frame conversion
        _X = rpy.r['model.matrix'](self.formula, data=X)
        _X = _X[:, self._which[self.nvar-1]] # 0-based indexing
        rpy.pandas2ri.deactivate() 
        return _X.dot(self._nz_coef)

    def score(self, X, y, sample_weight=None):
        return -mean_squared_error(y, 
                                   self.predict(X), 
                                   sample_weight=sample_weight,
                                   multioutput='uniform_average')

    def predict(self, D):
        return _predict(self.formula, D, self._which, self.nvar, self.coef_)

class SubsetCV(BaseEstimator, RegressorMixin): 

    def __init__(self, 
                 formula_str, 
                 nvmax, 
                 method='exhaustive', 
                 cv=3, 
                 n_jobs=-1,
                 verbose=False):
        self.formula_str = formula_str
        self.formula = rpy.r(formula_str)
        self.method = method
        self.cv = cv
        self.nvmax = nvmax
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):
        """Fit cross-validated best subsets regression

        Fit is over a grid of sizes up to `nvmax`.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training data. Pass directly as Fortran-contiguous data
            to avoid unnecessary memory duplication. If y is mono-output,
            X can be sparse.

        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values
        """

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have inconsistent dimensions (%d != %d)"
                             % (X.shape[0], y.shape[0]))

        cv = check_cv(self.cv)

        # Compute path for all folds and compute MSE to get the best alpha
        folds = list(cv.split(X, y))
        best_mse = np.inf

        # Loop over folds, computing mse path
        # for each (train, test)
        jobs = (delayed(_regsubsets_MSE)(X, 
                                         y, 
                                         self.formula, 
                                         self.method, 
                                         self.nvmax, 
                                         train, 
                                         test)
                for train, test in folds)

        # Execute the jobs in parallel using joblib
        MSEs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                        backend="threading")(jobs)
        MSEs = np.asarray(MSEs)
        self.mse_path = MSEs.mean(0)

        # Find best index by minimizing MSEs

        self.best_index = np.argmin(self.mse_path) + 1

        # Refit with best index hyperparameter
        self.best_estimator = Subset(self.formula_str, 
                                     self.best_index,
                                     method=self.method)
        self.best_estimator.fit(X, y)
        self.coef_ = self.best_estimator.coef_
        return self

    def predict(self, D):
        return self.best_estimator.predict(D)

def _regsubsets_MSE(X, 
                    y, 
                    formula, 
                    method, 
                    nvar, 
                    train, 
                    test):
                    
        rpy.pandas2ri.activate() # for data frame conversion
        D = pd.concat([X, y], axis=1) # reconstitute data

        Dtrain = D.loc[D.index[train]]
        Dtest = D.loc[D.index[test]]

        _regfit = rpy.r['regsubsets'](formula, 
                                      data=Dtrain, 
                                      method=method,
                                      nvmax=nvar)
        _which = rpy.r['summary'](_regfit).rx2('which').astype(np.bool)
        _Xtest = rpy.r['model.matrix'](formula, data=Dtest)
        _coef = np.zeros(_Xtest.shape[1])

        _MSEs = []

        _y_test = y.loc[y.index[test]]
        for ivar in range(1, nvar+1):
            yhat = np.zeros_like(y[test])
            rpy.numpy2ri.activate()
            _nz_coef = rpy.r['coef'](_regfit, ivar)
            _mask = _which[ivar-1]
            _coef *= 0
            _coef[_mask] = _nz_coef
            rpy.numpy2ri.deactivate()
            _y_hat = _Xtest.dot(_coef)
            _MSEs.append(((_y_test - _y_hat)**2).mean())
        rpy.pandas2ri.deactivate()
        return _MSEs

def _predict(formula, D, _which, nvar, _nz_coef):
    rpy.pandas2ri.activate() # for data frame conversion
    _X = rpy.r['model.matrix'](formula, data=D)
    _X = _X[:, _which[nvar-1]] # 0-based indexing
    rpy.pandas2ri.deactivate() 
    return _X.dot(_nz_coef)
    
