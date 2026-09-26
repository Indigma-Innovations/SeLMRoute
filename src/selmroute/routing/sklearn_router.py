import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import numeric_matrix


class _SklearnQualityRouter:
    name = "sklearn"

    def __init__(self, model):
        self.model = model
        self.feature_names: list[str] = []
        self.model_names: list[str] = []

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
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
        (path / "metadata.json").write_text(
            json.dumps({"router": self.name, "feature_names": self.feature_names, "model_names": self.model_names}, indent=2),
            encoding="utf-8",
        )


class OLSQualityRouter(_SklearnQualityRouter):
    name = "ols"

    def __init__(self):
        super().__init__(make_pipeline(StandardScaler(), LinearRegression()))


class RandomForestQualityRouter(_SklearnQualityRouter):
    name = "random_forest"

    def __init__(self, *, seed: int = 3407, n_estimators: int = 400, min_samples_leaf: int = 2):
        super().__init__(RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        ))


class MLPQualityRouter(_SklearnQualityRouter):
    name = "mlp"

    def __init__(self, *, seed: int = 3407, hidden_layer_sizes: tuple[int, ...] = (64, 32), alpha: float = 1e-4):
        super().__init__(make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=alpha,
                batch_size="auto",
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=seed,
            ),
        ))
