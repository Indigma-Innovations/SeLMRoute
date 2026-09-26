from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


class QualityRouter(Protocol):
    model_names: list[str]
    feature_names: list[str]
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "QualityRouter": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def save(self, path: str | Path) -> None: ...


def numeric_matrix(features: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    arr = features[feature_names].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return arr.to_numpy(dtype=float)
