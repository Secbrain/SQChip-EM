"""
R0: Constant Baseline - predicts mean/median of training set
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class ConstantBaseline(BaseEstimator, RegressorMixin):
    """
    Constant baseline regressor.
    Predicts the mean or median of the training target for all samples.
    """

    def __init__(self, strategy='mean'):
        """
        Parameters
        ----------
        strategy : str, default='mean'
            Strategy to compute the constant prediction.
            'mean': use training set mean
            'median': use training set median
        """
        self.strategy = strategy
        self.constant_ = None

    def fit(self, X, y):
        """
        Fit the constant baseline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (ignored, only y is used)
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self
        """
        y = np.asarray(y)
        if self.strategy == 'mean':
            self.constant_ = np.mean(y)
        elif self.strategy == 'median':
            self.constant_ = np.median(y)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        return self

    def predict(self, X):
        """
        Predict using the constant value.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values (all equal to constant_)
        """
        X = np.asarray(X)
        return np.full(X.shape[0], self.constant_)

    def get_params(self, deep=True):
        return {'strategy': self.strategy}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
