"""
Baseline 3.3: Surrogate → Retrieval
先训练预测器从设计特征预测目标参数，然后用预测值做检索排序。

Models:
- Ridge (线性，最稳定，几乎不调参)
- RandomForest (表格数据强基线)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Single retrieval result"""
    sample_id: str
    json_path: str
    gds_path: str
    fq_GHz: float
    fr_GHz: float
    chi_MHz: float
    g_MHz: float
    kappa_MHz: float
    distance: float  # cost value for ranking


class SurrogateRetriever:
    """
    Baseline 3.3: Surrogate → Retrieval

    1. Train a surrogate model: x (design features) → y (fq, fr, chi, g, kappa)
    2. At retrieval time, use predicted values for cost-based ranking
    """

    def __init__(
        self,
        model_type: str = 'ridge',  # 'ridge' or 'random_forest'
        feature_cols: Optional[List[str]] = None,
        target_cols: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        normalize_cost: bool = True,
        **model_kwargs
    ):
        """
        Args:
            model_type: 'ridge' or 'random_forest'
            feature_cols: Design parameter columns as input features
            target_cols: Target columns to predict
            weights: Weight dict for cost function
            normalize_cost: Whether to normalize cost by target value
            model_kwargs: Additional kwargs for the model
        """
        self.model_type = model_type
        self.feature_cols = feature_cols or [
            'cap_gap_um', 'pad_gap_um', 'dx_mm', 'dy_mm',
            'Lj_nH', 'Cj_fF', 'tee_finger_length_um', 'tee_finger_count', 'ro_L_mm'
        ]
        self.target_cols = target_cols or [
            'fq_GHz', 'fr_GHz', 'chi_MHz', 'g_over_2pi_MHz', 'kappa_over_2pi_MHz'
        ]
        self.weights = weights or {
            'fq_GHz': 1.0,
            'fr_GHz': 1.0,
            'chi_MHz': 1.0,
            'g_over_2pi_MHz': 1.0,
            'kappa_over_2pi_MHz': 1.0
        }
        self.normalize_cost = normalize_cost
        self.model_kwargs = model_kwargs

        # Initialize scaler and model
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.model = None
        self.train_df = None
        self.data_dir = None
        self.predictions = None  # Cached predictions for train set

    def _create_model(self):
        """Create the surrogate model based on model_type."""
        if self.model_type == 'ridge':
            base_model = Ridge(alpha=self.model_kwargs.get('alpha', 1.0))
            return MultiOutputRegressor(base_model)
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.model_kwargs.get('n_estimators', 100),
                max_depth=self.model_kwargs.get('max_depth', 10),
                min_samples_leaf=self.model_kwargs.get('min_samples_leaf', 2),
                random_state=self.model_kwargs.get('random_state', 42),
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, train_df: pd.DataFrame, data_dir: str):
        """
        Train the surrogate model on training data.

        Args:
            train_df: Training dataframe with design parameters and targets
            data_dir: Root directory containing json/ and gds/ folders
        """
        self.train_df = train_df.copy()
        self.data_dir = data_dir

        # Prepare features and targets, drop rows with NaN
        all_cols = self.feature_cols + self.target_cols
        valid_mask = train_df[all_cols].notna().all(axis=1)
        clean_df = train_df[valid_mask]

        X = clean_df[self.feature_cols].values
        y = clean_df[self.target_cols].values

        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)

        # For Ridge, also scale targets for better performance
        if self.model_type == 'ridge':
            y_scaled = self.target_scaler.fit_transform(y)
        else:
            y_scaled = y

        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_scaled, y_scaled)

        # Cache predictions for ALL train samples (including those with NaN features → use 0)
        X_all = train_df[self.feature_cols].fillna(0).values
        X_all_scaled = self.feature_scaler.transform(X_all)
        if self.model_type == 'ridge':
            self.predictions = self.target_scaler.inverse_transform(
                self.model.predict(X_all_scaled)
            )
        else:
            self.predictions = self.model.predict(X_all_scaled)

        return self

    def _compute_cost(
        self,
        predicted: np.ndarray,
        target_fq: float,
        target_fr: Optional[float],
        target_chi: Optional[float],
        target_g: Optional[float],
        target_kappa: Optional[float]
    ) -> float:
        """
        Compute cost between predicted values and targets.

        cost = Σ_i w_i * (predicted_i - target_i)^2 / scale_i^2
        """
        cost = 0.0
        targets = [target_fq, target_fr, target_chi, target_g, target_kappa]

        for i, (col, target) in enumerate(zip(self.target_cols, targets)):
            if target is None:
                continue

            diff = predicted[i] - target
            weight = self.weights.get(col, 1.0)

            if self.normalize_cost and target != 0:
                cost += weight * (diff ** 2) / (target ** 2)
            else:
                cost += weight * (diff ** 2)

        return cost

    def retrieve(
        self,
        target_fq: float,
        target_fr: Optional[float] = None,
        target_chi: Optional[float] = None,
        target_g: Optional[float] = None,
        target_kappa: Optional[float] = None,
        top_k: int = 5,
        constraints: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve top-K designs using surrogate predictions.

        Args:
            target_fq: Target qubit frequency (GHz)
            target_fr: Target readout frequency (GHz), optional
            target_chi: Target chi coupling (MHz), optional
            target_g: Target g coupling (MHz), optional
            target_kappa: Target kappa (MHz), optional
            top_k: Number of results to return
            constraints: Optional constraints dict

        Returns:
            List of RetrievalResult objects sorted by cost (ascending)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Apply constraints if provided
        if constraints:
            mask = np.ones(len(self.train_df), dtype=bool)
            for col, val in constraints.items():
                if col in self.train_df.columns:
                    if isinstance(val, tuple):
                        mask &= (self.train_df[col] >= val[0]) & (self.train_df[col] <= val[1])
                    else:
                        mask &= (self.train_df[col] == val)
            search_indices = np.where(mask)[0]
            search_df = self.train_df.iloc[search_indices]
            search_predictions = self.predictions[search_indices]
        else:
            search_indices = np.arange(len(self.train_df))
            search_df = self.train_df
            search_predictions = self.predictions

        if len(search_df) == 0:
            return []

        # Compute cost for all candidates using predicted values
        costs = []
        for i, pred in enumerate(search_predictions):
            cost = self._compute_cost(
                pred, target_fq, target_fr, target_chi, target_g, target_kappa
            )
            costs.append((search_indices[i], cost))

        # Sort by cost (ascending) and take top-K
        costs.sort(key=lambda x: x[1])
        top_indices = [c[0] for c in costs[:top_k]]
        top_costs = [c[1] for c in costs[:top_k]]

        # Build results using ACTUAL values (not predicted)
        results = []
        for i, idx in enumerate(top_indices):
            row = self.train_df.iloc[idx]
            result = RetrievalResult(
                sample_id=row['sample_id'],
                json_path=f"{self.data_dir}/json/{row['sample_id']}.json",
                gds_path=f"{self.data_dir}/gds/{row['sample_id']}.gds",
                fq_GHz=row['fq_GHz'],
                fr_GHz=row['fr_GHz'],
                chi_MHz=row['chi_MHz'],
                g_MHz=row['g_over_2pi_MHz'],
                kappa_MHz=row['kappa_over_2pi_MHz'],
                distance=top_costs[i]
            )
            results.append(result)

        return results

    def batch_retrieve(
        self,
        queries: pd.DataFrame,
        top_k: int = 5,
        target_cols_map: Optional[Dict[str, str]] = None
    ) -> List[List[RetrievalResult]]:
        """
        Batch retrieval for multiple queries.

        Args:
            queries: DataFrame with query targets
            top_k: Number of results per query
            target_cols_map: Map from query columns to expected names

        Returns:
            List of result lists, one per query
        """
        target_cols_map = target_cols_map or {
            'fq_GHz': 'fq_GHz',
            'fr_GHz': 'fr_GHz',
            'chi_MHz': 'chi_MHz',
            'g_over_2pi_MHz': 'g_over_2pi_MHz',
            'kappa_over_2pi_MHz': 'kappa_over_2pi_MHz'
        }

        all_results = []
        for _, row in queries.iterrows():
            results = self.retrieve(
                target_fq=row.get(target_cols_map.get('fq_GHz', 'fq_GHz')),
                target_fr=row.get(target_cols_map.get('fr_GHz', 'fr_GHz')),
                target_chi=row.get(target_cols_map.get('chi_MHz', 'chi_MHz')),
                target_g=row.get(target_cols_map.get('g_over_2pi_MHz', 'g_over_2pi_MHz')),
                target_kappa=row.get(target_cols_map.get('kappa_over_2pi_MHz', 'kappa_over_2pi_MHz')),
                top_k=top_k
            )
            all_results.append(results)

        return all_results

    def get_model_info(self) -> Dict:
        """Get information about the trained model."""
        info = {
            'model_type': self.model_type,
            'feature_cols': self.feature_cols,
            'target_cols': self.target_cols,
            'n_train_samples': len(self.train_df) if self.train_df is not None else 0,
        }

        if self.model_type == 'random_forest' and self.model is not None:
            info['feature_importances'] = dict(zip(
                self.feature_cols,
                self.model.feature_importances_.tolist()
            ))

        return info
