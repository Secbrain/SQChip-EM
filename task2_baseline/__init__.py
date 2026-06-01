"""
Task 2 Baseline Models for Classification
"""

from .majority_baseline import MajorityBaseline
from .lr_baseline import LogisticRegressionBaseline
from .rf_classifier_baseline import RandomForestClassifierBaseline

__all__ = [
    'MajorityBaseline',
    'LogisticRegressionBaseline',
    'RandomForestClassifierBaseline'
]
