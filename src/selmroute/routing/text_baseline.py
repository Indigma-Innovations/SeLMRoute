import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


class TfidfQualityRouter:
    def __init__(self, max_features: int = 50000, ngram_range: tuple[int, int] = (1, 2), alpha: float = 5.0):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, sublinear_tf=True, min_df=2)
        self.regressor = Ridge(alpha=alpha)
        self.model_names: list[str] = []

    def fit(self, texts: pd.Series, Y: pd.DataFrame) -> "TfidfQualityRouter":
        X = self.vectorizer.fit_transform(texts.astype(str).tolist())
        self.model_names = list(Y.columns)
        self.regressor.fit(X, Y.to_numpy(dtype=float))
        return self

    def predict(self, texts: pd.Series) -> np.ndarray:
        out = np.asarray(self.regressor.predict(self.vectorizer.transform(texts.astype(str).tolist())), dtype=float)
        return out[:, None] if out.ndim == 1 else out
