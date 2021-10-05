import numpy as np, pandas as pd

from sklearn.base import (BaseEstimator,
                          RegressorMixin,
                          TransformerMixin)
from sklearn.utils.validation import check_is_fitted

import statsmodels.api as sm
from patsy import dmatrix, dmatrices

class sklearn_sm(BaseEstimator,
                 RegressorMixin): 

    def __init__(self,
                 model_type,
                 model_matrix=None,
                 model_args={}):
        """
        Parameters
        ----------

        model_type: class
            A model type from statsmodels, e.g. sm.OLS or sm.GLM

        model_matrix: ModelMatrix
            Specify the design matrix.

        model_args: dict (optional)
            Arguments passed to the statsmodels model.

        Notes
        -----

        If model_str is present, then X and Y are presumed
        to be pandas objects that are placed
        into a dataframe before formula is evaluated.
        This affects `fit` and `predict` methods.

        """
        self.model_type = model_type
        self.model_matrix = model_matrix
        self.model_args = model_args
        
    def fit(self, X, y):
        """
        Fit a statsmodel model
        with design matrix 
        determined from X and response y.

        Parameters
        ----------

        X : array-like
            Design matrix.

        y : array-like
            Response vector.
        """

        if self.model_matrix is not None:
            self.model_matrix_ = self.model_matrix.fit(X)
            X = self.model_matrix_.transform(X)
        self.model_ = self.model_type(y, X, **self.model_args)
        self.results_ = self.model_.fit()

    def predict(self, X):
        """
        Compute predictions
        for design matrix X.

        Parameters
        ----------

        X : array-like
            Design matrix.

        """
        if self.model_matrix is not None:
            X = self.model_matrix_.transform(X)
        return self.results_.predict(exog=X)
 
    def score(self, X, y, sample_weight=None):
        """
        Score a statsmodel model
        with test design matrix X and 
        test response y.

        If model_type is OLS, use MSE. For
        a GLM this computes (average) deviance.

        Parameters
        ----------

        X : array-like
            Design matrix.

        y : array-like
            Response vector.

        sample_weight : None
            Optional sample weights.
        """

        yhat = self.predict(X)
        if isinstance(self.model_, sm.OLS):
            if sample_weight is None:
                return np.mean((y-yhat)**2)
            else:
                return (np.mean((y-yhat)**2*sample_weight) /
                        np.mean(sample_weight))
                
        elif isinstance(self.model_, sm.GLM):
            if sample_weight is None:
                return self.model_.family.deviance(y,
                                                   yhat).mean()
            else:
                value = self.model_.family.deviance(y,
                                                    yhat,
                                                    freq_weights=sample_weight).mean()

                return value / np.mean(sample_weight)

