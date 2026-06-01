"""
R2: Random Forest Baseline - Random Forest Regressor
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestBaseline:
    """
    Random Forest baseline regressor.
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, random_state=42):
        """
        Parameters
        ----------
        n_estimators : int, default=100
            Number of trees in the forest
        max_depth : int or None, default=None
            Maximum depth of trees
        min_samples_split : int, default=2
            Minimum samples required to split a node
        min_samples_leaf : int, default=1
            Minimum samples required at a leaf node
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_importances_ = None

    def fit(self, X, y):
        """
        Fit the Random Forest regressor.

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

        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_
        return self

    def predict(self, X):
        """
        Predict using Random Forest.

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
        return self.model.predict(X)

    def get_feature_importances(self, feature_names=None):
        """
        Get feature importances.

        Parameters
        ----------
        feature_names : list of str, optional
            Feature names for output

        Returns
        -------
        dict : feature name -> importance
        """
        if self.feature_importances_ is None:
            raise ValueError("Model not fitted yet")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importances_))]

        return dict(zip(feature_names, self.feature_importances_))

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        # Rebuild model with new params
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1
        )
        return self
