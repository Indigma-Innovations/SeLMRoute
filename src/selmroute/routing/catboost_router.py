import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .base import numeric_matrix


class CatBoostQualityRouter:
    def __init__(self, *, iterations: int = 500, depth: int = 6, learning_rate: float = 0.05, seed: int = 3407, verbose: bool = False):
        self.params = dict(iterations=iterations, depth=depth, learning_rate=learning_rate, random_seed=seed, verbose=verbose, loss_function="MultiRMSE", allow_writing_files=False)
        self.model = CatBoostRegressor(**self.params)
        self.feature_names: list[str] = []
        self.model_names: list[str] = []

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "CatBoostQualityRouter":
        self.feature_names = [c for c in X.columns if c != "sample_id" and not c.startswith("meta_")]
        self.model_names = list(Y.columns)
        self.model.fit(numeric_matrix(X, self.feature_names), Y.to_numpy(dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.asarray(self.model.predict(numeric_matrix(X, self.feature_names)), dtype=float)
        if out.ndim == 1:
            out = out[:, None]
        return out

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.get_feature_importance(), index=self.feature_names).sort_values(ascending=False)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path / "model.cbm")
        (path / "metadata.json").write_text(json.dumps({"feature_names": self.feature_names, "model_names": self.model_names, "params": self.params}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CatBoostQualityRouter":
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        obj = cls()
        obj.model = CatBoostRegressor()
        obj.model.load_model(path / "model.cbm")
        obj.feature_names = meta["feature_names"]
        obj.model_names = meta["model_names"]
        obj.params = meta.get("params", {})
        return obj
