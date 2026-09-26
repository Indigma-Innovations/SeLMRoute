import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import numeric_matrix


class RidgeQualityRouter:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        self.feature_names: list[str] = []
        self.model_names: list[str] = []

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "RidgeQualityRouter":
        self.feature_names = [c for c in X.columns if c != "sample_id" and not c.startswith("meta_")]
        self.model_names = list(Y.columns)
        self.model.fit(numeric_matrix(X, self.feature_names), Y.to_numpy(dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.asarray(self.model.predict(numeric_matrix(X, self.feature_names)), dtype=float)
        return out[:, None] if out.ndim == 1 else out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")
        (path / "metadata.json").write_text(json.dumps({"feature_names": self.feature_names, "model_names": self.model_names, "alpha": self.alpha}, indent=2), encoding="utf-8")
