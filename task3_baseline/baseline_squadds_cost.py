"""
Baseline B: SQuADDS-style Cost Function Retriever
Uses weighted normalized cost function for multi-target retrieval.

Reference: SQuADDS approach for "best-guess design" based on target parameters.
Cost function: F({P_i}, {p_i}) = Σ_i w_i * (P_i - p_i)^2 / P_i^2
"""

import numpy as np
import pandas as pd
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
    distance: float  # cost value for ranking


class SQuADDSCostRetriever:
    """
    Baseline B: SQuADDS-style cost function retrieval.
    Uses weighted normalized squared error for multi-target ranking.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        normalize: bool = True
    ):
        """
        Args:
            weights: Weight dict for each target parameter
            normalize: Whether to normalize by target value squared
        """
        # Default weights (can be customized)
        self.weights = weights or {
            'fq_GHz': 1.0,
            'fr_GHz': 1.0,
            'chi_MHz': 1.0,
            'g_over_2pi_MHz': 1.0,
            'kappa_over_2pi_MHz': 1.0
        }
        self.normalize = normalize
        self.train_df = None
        self.data_dir = None

    def fit(self, train_df: pd.DataFrame, data_dir: str):
        """
        Store training data (database) for retrieval.

        Args:
            train_df: Training dataframe with design parameters and targets
            data_dir: Root directory containing json/ and gds/ folders
        """
        self.train_df = train_df.copy()
        self.data_dir = data_dir
        return self

    def _compute_cost(
        self,
        row: pd.Series,
        target_fq: float,
        target_fr: Optional[float],
        target_chi: Optional[float],
        target_g: Optional[float],
        target_kappa: Optional[float]
    ) -> float:
        """
        Compute cost for a single design.

        Cost = Σ_i w_i * (actual_i - target_i)^2 / target_i^2  (if normalized)
        Cost = Σ_i w_i * (actual_i - target_i)^2               (if not normalized)
        """
        cost = 0.0

        # fq (always required)
        diff_fq = row['fq_GHz'] - target_fq
        if self.normalize and target_fq != 0:
            cost += self.weights.get('fq_GHz', 1.0) * (diff_fq ** 2) / (target_fq ** 2)
        else:
            cost += self.weights.get('fq_GHz', 1.0) * (diff_fq ** 2)

        # fr (optional)
        if target_fr is not None:
            diff_fr = row['fr_GHz'] - target_fr
            if self.normalize and target_fr != 0:
                cost += self.weights.get('fr_GHz', 1.0) * (diff_fr ** 2) / (target_fr ** 2)
            else:
                cost += self.weights.get('fr_GHz', 1.0) * (diff_fr ** 2)

        # chi (optional)
        if target_chi is not None:
            diff_chi = row['chi_MHz'] - target_chi
            if self.normalize and target_chi != 0:
                cost += self.weights.get('chi_MHz', 1.0) * (diff_chi ** 2) / (target_chi ** 2)
            else:
                cost += self.weights.get('chi_MHz', 1.0) * (diff_chi ** 2)

        # g (optional)
        if target_g is not None:
            diff_g = row['g_over_2pi_MHz'] - target_g
            if self.normalize and target_g != 0:
                cost += self.weights.get('g_over_2pi_MHz', 1.0) * (diff_g ** 2) / (target_g ** 2)
            else:
                cost += self.weights.get('g_over_2pi_MHz', 1.0) * (diff_g ** 2)

        # kappa (optional)
        if target_kappa is not None:
            diff_kappa = row['kappa_over_2pi_MHz'] - target_kappa
            if self.normalize and target_kappa != 0:
                cost += self.weights.get('kappa_over_2pi_MHz', 1.0) * (diff_kappa ** 2) / (target_kappa ** 2)
            else:
                cost += self.weights.get('kappa_over_2pi_MHz', 1.0) * (diff_kappa ** 2)

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
        Retrieve top-K designs with lowest cost to target parameters.

        Args:
            target_fq: Target qubit frequency (GHz)
            target_fr: Target readout frequency (GHz), optional
            target_chi: Target chi coupling (MHz), optional
            target_g: Target g coupling (MHz), optional
            target_kappa: Target kappa (MHz), optional
            top_k: Number of results to return
            constraints: Optional constraints dict, e.g. {'Lj_nH': 12.0}

        Returns:
            List of RetrievalResult objects sorted by cost (ascending)
        """
        if self.train_df is None:
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

        # Compute cost for all candidates
        costs = []
        for idx, row in search_df.iterrows():
            cost = self._compute_cost(
                row, target_fq, target_fr, target_chi, target_g, target_kappa
            )
            costs.append((idx, cost))

        # Sort by cost (ascending) and take top-K
        costs.sort(key=lambda x: x[1])
        top_indices = [c[0] for c in costs[:top_k]]
        top_costs = [c[1] for c in costs[:top_k]]

        # Build results
        results = []
        for i, idx in enumerate(top_indices):
            row = search_df.loc[idx]
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


class SQuADDSCostRetrieverFqOnly(SQuADDSCostRetriever):
    """
    Simplified version: Only uses fq for cost computation.
    Cost = (fq - fq*)^2 / (fq*)^2
    """

    def __init__(self, normalize: bool = True):
        super().__init__(
            weights={'fq_GHz': 1.0},
            normalize=normalize
        )

    def _compute_cost(
        self,
        row: pd.Series,
        target_fq: float,
        target_fr: Optional[float] = None,
        target_chi: Optional[float] = None,
        target_g: Optional[float] = None,
        target_kappa: Optional[float] = None
    ) -> float:
        """Only compute cost based on fq."""
        diff_fq = row['fq_GHz'] - target_fq
        if self.normalize and target_fq != 0:
            return (diff_fq ** 2) / (target_fq ** 2)
        return diff_fq ** 2
