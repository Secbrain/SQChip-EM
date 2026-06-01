"""
Baseline A: NearestNeighbors Retriever
Uses design parameter features to build nearest neighbor index for retrieval.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
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
    distance: float  # distance/cost for ranking


class NearestNeighborRetriever:
    """
    Baseline A: Feature-space nearest neighbor retrieval.
    Uses design parameters (or target parameters) to find similar designs.
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        target_cols: Optional[List[str]] = None,
        n_neighbors: int = 10,
        metric: str = 'euclidean'
    ):
        """
        Args:
            feature_cols: Design parameter columns for feature space
            target_cols: Target parameter columns (fq, fr, chi, g, kappa)
            n_neighbors: Number of neighbors to retrieve
            metric: Distance metric ('euclidean', 'manhattan', etc.)
        """
        self.feature_cols = feature_cols or [
            'cap_gap_um', 'pad_gap_um', 'dx_mm', 'dy_mm',
            'Lj_nH', 'Cj_fF', 'tee_finger_length_um', 'tee_finger_count', 'ro_L_mm'
        ]
        self.target_cols = target_cols or [
            'fq_GHz', 'fr_GHz', 'chi_MHz', 'g_over_2pi_MHz', 'kappa_over_2pi_MHz'
        ]
        self.n_neighbors = n_neighbors
        self.metric = metric

        self.scaler = StandardScaler()
        self.nn_model = None
        self.train_df = None
        self.data_dir = None

    def fit(self, train_df: pd.DataFrame, data_dir: str):
        """
        Build the nearest neighbor index from training data.

        Args:
            train_df: Training dataframe with design parameters and targets
            data_dir: Root directory containing json/ and gds/ folders
        """
        self.train_df = train_df.copy()
        self.data_dir = data_dir

        # Use target columns for retrieval (find designs with similar output)
        X = train_df[self.target_cols].values
        X_scaled = self.scaler.fit_transform(X)

        self.nn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(train_df)),
            metric=self.metric,
            algorithm='auto'
        )
        self.nn_model.fit(X_scaled)

        return self

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
        Retrieve top-K designs closest to target parameters.

        Args:
            target_fq: Target qubit frequency (GHz)
            target_fr: Target readout frequency (GHz), optional
            target_chi: Target chi coupling (MHz), optional
            target_g: Target g coupling (MHz), optional
            target_kappa: Target kappa (MHz), optional
            top_k: Number of results to return
            constraints: Optional constraints dict, e.g. {'Lj_nH': 12.0}

        Returns:
            List of RetrievalResult objects sorted by distance
        """
        if self.nn_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Apply constraints if provided
        search_df = self.train_df.copy()
        if constraints:
            for col, val in constraints.items():
                if col in search_df.columns:
                    if isinstance(val, tuple):
                        # Range constraint
                        search_df = search_df[
                            (search_df[col] >= val[0]) & (search_df[col] <= val[1])
                        ]
                    else:
                        # Exact match
                        search_df = search_df[search_df[col] == val]

        if len(search_df) == 0:
            return []

        # Build query vector using available targets
        # Use mean of training data for missing values
        train_means = self.train_df[self.target_cols].mean()
        query = np.array([
            target_fq,
            target_fr if target_fr is not None else train_means['fr_GHz'],
            target_chi if target_chi is not None else train_means['chi_MHz'],
            target_g if target_g is not None else train_means['g_over_2pi_MHz'],
            target_kappa if target_kappa is not None else train_means['kappa_over_2pi_MHz']
        ]).reshape(1, -1)

        query_scaled = self.scaler.transform(query)

        # Get candidates from constrained set
        if constraints and len(search_df) < len(self.train_df):
            # Refit on constrained data
            X_constrained = search_df[self.target_cols].values
            X_scaled = self.scaler.transform(X_constrained)

            nn_temp = NearestNeighbors(
                n_neighbors=min(top_k, len(search_df)),
                metric=self.metric
            )
            nn_temp.fit(X_scaled)
            distances, indices = nn_temp.kneighbors(query_scaled)
            candidate_df = search_df.iloc[indices[0]]
        else:
            distances, indices = self.nn_model.kneighbors(
                query_scaled, n_neighbors=min(top_k, len(self.train_df))
            )
            candidate_df = self.train_df.iloc[indices[0]]

        # Build results
        results = []
        for i, (idx, row) in enumerate(candidate_df.iterrows()):
            result = RetrievalResult(
                sample_id=row['sample_id'],
                json_path=f"{self.data_dir}/json/{row['sample_id']}.json",
                gds_path=f"{self.data_dir}/gds/{row['sample_id']}.gds",
                fq_GHz=row['fq_GHz'],
                fr_GHz=row['fr_GHz'],
                chi_MHz=row['chi_MHz'],
                g_MHz=row['g_over_2pi_MHz'],
                kappa_MHz=row['kappa_over_2pi_MHz'],
                distance=distances[0][i]
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
            queries: DataFrame with query targets (fq_GHz, fr_GHz, etc.)
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
