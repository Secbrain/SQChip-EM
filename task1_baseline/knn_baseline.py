"""
R1: kNN Baseline - K-Nearest Neighbors Regressor
"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


class KNNBaseline:
    """
    kNN baseline regressor with optional feature scaling.
    """

    def __init__(self, n_neighbors=5, weights='uniform', scale_features=True):
        """
        Parameters
        ----------
        n_neighbors : int, default=5
            Number of neighbors to use
        weights : str, default='uniform'
            Weight function: 'uniform' or 'distance'
        scale_features : bool, default=True
            Whether to standardize features before fitting
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.scale_features = scale_features
        self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights
        )
        self.scaler = StandardScaler() if scale_features else None

    def fit(self, X, y):
        """
        Fit the kNN regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if self.scaler is not None:
            X = self.scaler.fit_transform(X)

        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict using kNN.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        X = np.asarray(X)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        return self.model.predict(X)

    def get_params(self, deep=True):
        return {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'scale_features': self.scale_features
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        # Rebuild model with new params
        self.model = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights
        )
        if self.scale_features:
            self.scaler = StandardScaler()
        else:
            self.scaler = None
        return self
