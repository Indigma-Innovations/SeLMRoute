import numpy as np


def select_quality(predicted_quality: np.ndarray) -> np.ndarray:
    return np.asarray(predicted_quality).argmax(axis=1)


def normalize_columns(values: np.ndarray, mins: np.ndarray | None = None, maxs: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    mins = values.min(axis=0) if mins is None else np.asarray(mins, dtype=float)
    maxs = values.max(axis=0) if maxs is None else np.asarray(maxs, dtype=float)
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    return (values - mins) / denom, mins, maxs


def select_cost_aware(predicted_quality: np.ndarray, predicted_cost: np.ndarray, lam: float, q_bounds=None, c_bounds=None) -> np.ndarray:
    if not 0 <= lam <= 1:
        raise ValueError("lam must be in [0,1]")
    qn, _, _ = normalize_columns(predicted_quality, *(q_bounds or (None, None)))
    cn, _, _ = normalize_columns(predicted_cost, *(c_bounds or (None, None)))
    utility = (1.0 - lam) * qn - lam * cn
    return utility.argmax(axis=1)
