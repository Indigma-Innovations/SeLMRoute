from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RoutingMetrics:
    avg_score: float
    random_score: float
    best_single_score: float
    oracle_score: float
    gain_over_random: float
    gain_over_best_single: float
    gap_to_oracle: float
    normalized_oracle_gap: float | None
    mean_regret: float

    def to_dict(self): return asdict(self)


def outcome_matrix(outcomes: pd.DataFrame, sample_ids: list[str] | np.ndarray, model_names: list[str] | None = None) -> pd.DataFrame:
    pivot = outcomes.pivot(index="sample_id", columns="model", values="score")
    if model_names is not None:
        pivot = pivot.reindex(columns=model_names)
    pivot = pivot.reindex(sample_ids)
    if pivot.isna().any().any():
        raise ValueError("Outcome matrix contains missing sample/model values; use complete cases or imputation explicitly")
    return pivot


def best_single_model(train_Y: pd.DataFrame) -> str:
    return str(train_Y.mean(axis=0).idxmax())


def routing_metrics(Y: pd.DataFrame, selected_indices: np.ndarray, best_single: str) -> RoutingMetrics:
    arr = Y.to_numpy(dtype=float)
    idx = np.asarray(selected_indices, dtype=int)
    achieved = arr[np.arange(len(arr)), idx]
    random_score = float(arr.mean(axis=1).mean())
    best_single_score = float(Y[best_single].mean())
    oracle = arr.max(axis=1)
    oracle_score = float(oracle.mean())
    avg = float(achieved.mean())
    denom = oracle_score - best_single_score
    normalized = (oracle_score - avg) / denom if denom > 1e-12 else None
    return RoutingMetrics(
        avg_score=avg,
        random_score=random_score,
        best_single_score=best_single_score,
        oracle_score=oracle_score,
        gain_over_random=avg-random_score,
        gain_over_best_single=avg-best_single_score,
        gap_to_oracle=oracle_score-avg,
        normalized_oracle_gap=normalized,
        mean_regret=float(np.mean(oracle-achieved)),
    )


def selected_frame(sample_ids, model_names, predictions, selected_indices, Y) -> pd.DataFrame:
    arr = Y.to_numpy(dtype=float)
    idx = np.asarray(selected_indices, dtype=int)
    top = np.sort(np.asarray(predictions, dtype=float), axis=1)
    margin = top[:, -1] - top[:, -2] if top.shape[1] > 1 else np.full(len(top), np.inf)
    return pd.DataFrame({
        "sample_id": list(sample_ids),
        "selected_model": [model_names[i] for i in idx],
        "selected_score": arr[np.arange(len(arr)), idx],
        "oracle_score": arr.max(axis=1),
        "regret": arr.max(axis=1)-arr[np.arange(len(arr)), idx],
        "prediction_margin": margin,
    })
