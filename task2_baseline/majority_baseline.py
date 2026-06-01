"""
C0: Majority Baseline - Always predicts the most frequent class
"""

import numpy as np
from collections import Counter


class MajorityBaseline:
    """
    Majority baseline classifier.
    Always predicts the most frequent class in the training set.
    """

    def __init__(self):
        self.majority_class_ = None
        self.class_probs_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit the majority baseline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (not used, but kept for API consistency)
        y : array-like of shape (n_samples,)
            Target labels

        Returns
        -------
        self
        """
        y = np.asarray(y)
        counter = Counter(y)
        self.classes_ = np.array(sorted(counter.keys()))
        self.majority_class_ = counter.most_common(1)[0][0]

        # Store class probabilities for predict_proba
        total = len(y)
        self.class_probs_ = np.array([counter[c] / total for c in self.classes_])

        return self

    def predict(self, X):
        """
        Predict using majority class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels (all same class)
        """
        X = np.asarray(X)
        return np.full(X.shape[0], self.majority_class_)

    def predict_proba(self, X):
        """
        Return probability estimates.

        Returns the training set class distribution for all samples.

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
        n_samples = X.shape[0]
        return np.tile(self.class_probs_, (n_samples, 1))

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self
