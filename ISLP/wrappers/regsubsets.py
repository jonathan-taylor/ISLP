import uuid

import numpy as np, pandas as pd

import rpy2.robjects as rpy
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_validate, check_cv
from sklearn.metrics import mean_squared_error
from joblib.parallel import Parallel, delayed

rpy.r('library(leaps)')

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
        ---------                                                         
        X : pd.DataFrame
            Training data. 

        y : ignored 
            Outcome for formula is assumed to be a column in the
            data frame X.

        """
        leaps = importr('leaps')
        base = importr('base')

        D = _convert_df(X.copy())
        _regfit = rpy.r['regsubsets'](self.formula,
                                      data=D,
                                      method=self.method,
                                      nvmax=self.nvar)
        self._which = np.asarray(rpy.r['summary'](_regfit).rx2('which')).astype(np.bool)
        _names = np.asarray(_regfit.rx2('xnames'))
        self._nz_coef = rpy.r['coef'](_regfit, self.nvar)
        self.coef_ = pd.Series(self._nz_coef, index=_names[self._which[self.nvar-1]])

    def predict(self, X):
        stats = importr('stats')
        with localconverter(rpy.default_converter + pandas2ri.converter):
            _X = stats.model_matrix(self.formula, data=X)
            _X = _X[:, self._which[self.nvar-1]] # 0-based indexing             
        return _X.dot(self._nz_coef)
    
class SubsetCV(BaseEstimator, RegressorMixin): 

    def __init__(self, 
                 formula_str, 
                 nvmax, 
                 method='exhaustive', 
                 cv=3, 
                 n_jobs=-1,
                 verbose=False):
        self.formula_str = formula_str
        self._y_label = formula_str.split('~')[0].strip()
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
        ---------                                                         
        X : pd.DataFrame
            Training data. 

        y : ignored 
            Outcome for formula is assumed to be a column in the
            data frame X.

        """

        cv = check_cv(self.cv)

        # Compute path for all folds and compute MSE to get the best alpha
        folds = list(cv.split(X))
        best_mse = np.inf

        # Loop over folds, computing mse path
        # for each (train, test)

        jobs = (delayed(_regsubsets_MSE)(X,
                                         self._y_label,
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

        self.best_index = int(np.argmin(self.mse_path) + 1)

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
                    y_label,
                    formula, 
                    method, 
                    nvar, 
                    train, 
                    test):

    D = X # pd.concat([X, y], axis=1) # reconstitute data

    Dtrain = _convert_df(D.loc[D.index[train]].copy())
    Dtest = _convert_df(D.loc[D.index[test]].copy())

    _regfit = rpy.r['regsubsets'](formula, 
                                  data=Dtrain, 
                                  method=method,
                                  nvmax=nvar)
    _which = np.asarray(rpy.r['summary'](_regfit).rx2('which')).astype(np.bool)
    _Xtest = np.asarray(rpy.r['model.matrix'](formula, data=Dtest))
    _coef = np.zeros(_Xtest.shape[1])

    _MSEs = []

    y = D[y_label]
    _y_test = y.loc[y.index[test]]
    for ivar in range(1, nvar+1):
        _nz_coef = rpy.r['coef'](_regfit, ivar)
        _mask = _which[ivar-1]
        _coef *= 0
        _coef[_mask] = _nz_coef
        _y_hat = _Xtest.dot(_coef)
        _MSEs.append(((_y_test - _y_hat)**2).mean())

    return _MSEs

def _convert_df(pd_df):
    with localconverter(rpy.default_converter + pandas2ri.converter):
        r_from_pd_df = rpy.conversion.py2rpy(pd_df)
    return r_from_pd_df

def _predict(formula, D, _which, nvar, _nz_coef):
    D_ = _convert_df(D)
    _X = np.asarray(rpy.r['model_matrix'](formula, data=D_))
    _X = _X[:, _which[nvar-1]] # 0-based indexing                           
    return _X.dot(_nz_coef)

