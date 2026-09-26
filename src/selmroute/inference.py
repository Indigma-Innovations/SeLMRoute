import json
from pathlib import Path

import numpy as np
import pandas as pd

from selmroute.routing.catboost_router import CatBoostQualityRouter


def load_router(backend: str = "laya", *, root: str | Path = ".") -> CatBoostQualityRouter:
    backend = backend.strip().lower()
    if backend not in {"laya", "jev"}:
        raise ValueError("backend must be 'laya' or 'jev'")
    return CatBoostQualityRouter.load(Path(root) / "models" / backend)


def align_features(features: pd.DataFrame, router: CatBoostQualityRouter) -> pd.DataFrame:
    frame = features.copy()
    if "sample_id" not in frame.columns:
        frame.insert(0, "sample_id", [f"query-{i}" for i in range(len(frame))])
    missing = [c for c in router.feature_names if c not in frame.columns]
    if missing:
        raise ValueError(f"Semantic feature frame is missing {len(missing)} router features: {missing[:8]}")
    out = frame[["sample_id", *router.feature_names]].copy()
    values = out[router.feature_names].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Semantic features contain NaN or infinite values")
    return out


def rank_from_features(
    features: pd.DataFrame,
    *,
    backend: str = "laya",
    root: str | Path = ".",
    top_k: int = 5,
) -> list[dict[str, object]]:
    if len(features) != 1:
        raise ValueError("rank_from_features expects exactly one query row")
    router = load_router(backend, root=root)
    X = align_features(features, router)
    pred = np.asarray(router.predict(X), dtype=float)[0]
    order = np.argsort(-pred)
    k = max(1, min(int(top_k), len(order)))
    return [
        {
            "rank": rank + 1,
            "model": router.model_names[int(idx)],
            "predicted_score": float(pred[int(idx)]),
        }
        for rank, idx in enumerate(order[:k])
    ]


def route_published_sample(
    sample_id: str,
    *,
    backend: str = "laya",
    root: str | Path = ".",
    top_k: int = 5,
    verbose: bool = False,
) -> dict[str, object]:
    """
    Route one published benchmark sample using frozen semantic evidence
    and the released backend-specific CatBoost router.

    Important:
    This function does NOT call Laya or JEV. The semantic backend was
    already executed when the published feature artifact was created.

    Pipeline:
        published query
            -> frozen Laya/JEV semantic evidence
            -> exact 40-dimensional ProbabilityMass vector
            -> released CatBoost router
            -> predicted candidate ranking

    Parameters
    ----------
    sample_id:
        Benchmark sample identifier.

    backend:
        Semantic backend whose frozen features and trained router should
        be used. Currently "laya" or "jev".

    root:
        Root of the public SeLMRoute repository.

    top_k:
        Number of candidate models returned in the ranking.

    verbose:
        If True, return an additional ``trace`` object describing the
        semantic-backend input, frozen semantic output, and exact
        CatBoost input.

    Returns
    -------
    dict
        Routing result. When ``verbose=True``, the result additionally
        contains an inspectable ``trace``.
    """

    backend = backend.strip().lower()

    if backend not in {"laya", "jev"}:
        raise ValueError(
            f"Unsupported backend={backend!r}. "
            "Expected 'laya' or 'jev'."
        )

    root = Path(root)

    feature_path = (
        root / "data/performance" / f"features_{backend}.csv"
    )
    sample_path = (
        root / "data/performance/samples.csv"
    )
    outcome_path = (
        root / "data/performance/outcomes.csv"
    )
    metadata_path = (
        root / "models" / backend / "metadata.json"
    )

    features = pd.read_csv(
        feature_path,
        dtype={"sample_id": str},
    )
    samples = pd.read_csv(
        sample_path,
        dtype={"sample_id": str},
    )
    outcomes = pd.read_csv(
        outcome_path,
        dtype={"sample_id": str},
    )

    sid = str(sample_id)

    row = features[
        features["sample_id"] == sid
    ]

    if len(row) != 1:
        raise ValueError(
            f"sample_id={sid!r} was not found exactly once "
            f"in {feature_path}"
        )

    meta = samples[
        samples["sample_id"] == sid
    ]

    if len(meta) != 1:
        raise ValueError(
            f"sample_id={sid!r} was not found exactly once "
            "in samples.csv"
        )

    query = str(meta.iloc[0]["query"])

    # Load the released model metadata so that we can show the exact
    # feature order expected by CatBoost.
    model_metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    feature_names = model_metadata["feature_names"]

    missing = [
        name
        for name in feature_names
        if name not in row.columns
    ]

    if missing:
        raise ValueError(
            "Published feature row does not contain all features "
            f"required by the {backend} release model. "
            f"Missing: {missing}"
        )

    # Vector passed to CatBoost, including column order.
    catboost_input = (
        row[feature_names]
        .iloc[0]
        .astype(float)
    )

    ranking = rank_from_features(
        row,
        backend=backend,
        root=root,
        top_k=top_k,
    )

    actual = (
        outcomes[
            outcomes["sample_id"] == sid
        ]
        .set_index("model")["score"]
        .to_dict()
    )

    for item in ranking:
        item["observed_score"] = (
            float(actual[item["model"]])
            if item["model"] in actual
            else None
        )

    result: dict[str, object] = {
        "sample_id": sid,
        "query": query,
        "dataset": str(meta.iloc[0]["dataset"]),
        "domain": (
            str(meta.iloc[0]["domain"])
            if "domain" in meta.columns
            else None
        ),
        "backend": backend,
        "selected_model": ranking[0]["model"],
        "ranking": ranking,
    }

    if verbose:
        result["trace"] = {
            "semantic_backend": {
                "backend": backend,
                "mode": "frozen_published_features",
                "live_call_performed": False,
                "input_query": query,
                "explanation": (
                    f"The query shown here was supplied to {backend} "
                    "when the published semantic feature artifact was "
                    "generated. This notebook does not call the backend "
                    "again."
                ),
            },

            "semantic_output": {
                "representation": "ProbabilityMass",
                "feature_count": len(feature_names),
                "features": {
                    name: float(catboost_input[name])
                    for name in feature_names
                },
            },

            "catboost_input": {
                "feature_count": len(feature_names),
                "feature_order": feature_names,
                "vector": [
                    float(catboost_input[name])
                    for name in feature_names
                ],
            },

            "catboost_output": {
                "selected_model": ranking[0]["model"],
                "ranking": ranking,
            },
        }

    return result
