"""
C1: Logistic Regression Baseline
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LogisticRegressionBaseline:
    """
    Logistic Regression baseline classifier with optional feature scaling.
    """

    def __init__(self, C=1.0, penalty='l2', solver='lbfgs', max_iter=1000,
                 scale_features=True, random_state=42):
        """
        Parameters
        ----------
        C : float, default=1.0
            Inverse of regularization strength
        penalty : str, default='l2'
            Regularization type ('l1', 'l2', 'elasticnet', 'none')
        solver : str, default='lbfgs'
            Optimization algorithm
        max_iter : int, default=1000
            Maximum iterations
        scale_features : bool, default=True
            Whether to standardize features
        random_state : int, default=42
            Random seed
        """
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.scale_features = scale_features
        self.random_state = random_state

        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state
        )
        self.scaler = StandardScaler() if scale_features else None
        self.coef_ = None

    def fit(self, X, y):
        """
        Fit the logistic regression classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels

        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if self.scaler is not None:
            X = self.scaler.fit_transform(X)

        self.model.fit(X, y)
        self.coef_ = self.model.coef_
        return self

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels
        """
        X = np.asarray(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Return probability estimates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
        X = np.asarray(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def get_feature_weights(self, feature_names=None):
        """
        Get feature weights (coefficients).

        Parameters
        ----------
        feature_names : list of str, optional
            Feature names

        Returns
        -------
        dict : feature name -> weight
        """
        if self.coef_ is None:
            raise ValueError("Model not fitted yet")

        coef = self.coef_.flatten()
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coef))]

        return dict(zip(feature_names, coef))

    def get_params(self, deep=True):
        return {
            'C': self.C,
            'penalty': self.penalty,
            'solver': self.solver,
            'max_iter': self.max_iter,
            'scale_features': self.scale_features,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.scaler = StandardScaler() if self.scale_features else None
        return self
