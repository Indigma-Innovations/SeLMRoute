import json
from pathlib import Path

import pandas as pd

from selmroute.data.splits import dataset_ood_split, domain_ood_split, id_split
from selmroute.evaluation.metrics import best_single_model, outcome_matrix, routing_metrics, selected_frame
from selmroute.routing.catboost_router import CatBoostQualityRouter
from selmroute.routing.linear_router import RidgeQualityRouter
from selmroute.routing.policy import select_quality
from selmroute.routing.sklearn_router import MLPQualityRouter, OLSQualityRouter, RandomForestQualityRouter
from selmroute.routing.text_baseline import TfidfQualityRouter


def select_feature_columns(features: pd.DataFrame, feature_set: str) -> list[str]:
    candidates=[c for c in features.columns if c!="sample_id" and not c.startswith("meta_")]
    if feature_set == "all":
        return candidates
    if feature_set == "deterministic":
        return [c for c in candidates if c.startswith("det_")]
    if feature_set in {"semantic", "semantic_manual"}:
        return [c for c in candidates if c.startswith("jev_") and "task_family" not in c]
    if feature_set == "semantic_all":
        return [c for c in candidates if c.startswith("jev_")]
    if feature_set == "domain_only":
        return [c for c in candidates if c.startswith("jev_task_family__")]
    if feature_set == "semantic_plus_deterministic":
        return [c for c in candidates if c.startswith("jev_") or c.startswith("det_")]
    raise ValueError(f"unknown feature_set={feature_set}")


def train_evaluate(
    samples: pd.DataFrame,
    outcomes: pd.DataFrame,
    features: pd.DataFrame,
    output_dir: str | Path,
    *,
    router: str="catboost",
    split: str="id",
    holdout: str|None=None,
    train_ratio: float=0.7,
    seed: int=3407,
    feature_set: str="semantic_plus_deterministic",
) -> dict:
    if split == "id":
        train_ids, test_ids = id_split(samples, train_ratio, seed)
    elif split == "id-grouped":
        train_ids, test_ids = id_split(samples, train_ratio, seed, group_duplicates=True)
    elif split == "dataset-ood":
        if not holdout:
            raise ValueError("dataset-ood requires holdout")
        train_ids, test_ids = dataset_ood_split(samples, holdout)
    elif split == "domain-ood":
        if not holdout:
            raise ValueError("domain-ood requires holdout")
        train_ids, test_ids = domain_ood_split(samples, holdout)
    else:
        raise ValueError(f"unknown split={split}")

    models=sorted(outcomes["model"].unique())
    train_Y=outcome_matrix(outcomes, train_ids, models)
    test_Y=outcome_matrix(outcomes, test_ids, models)
    best=best_single_model(train_Y)
    feature_cols=select_feature_columns(features, feature_set)
    if not feature_cols and router != "tfidf":
        raise ValueError(f"No columns found for feature set {feature_set}")
    indexed=features.set_index("sample_id").copy()
    train_X=indexed.reindex(train_ids).reset_index()[["sample_id",*feature_cols]].copy()
    test_X=indexed.reindex(test_ids).reset_index()[["sample_id",*feature_cols]].copy()
    if train_X[feature_cols].isna().all(axis=1).any() or test_X[feature_cols].isna().all(axis=1).any():
        raise ValueError("Missing feature rows for some samples")

    if router == "catboost":
        r=CatBoostQualityRouter(seed=seed).fit(train_X, train_Y)
        pred=r.predict(test_X)
    elif router == "ridge":
        r=RidgeQualityRouter().fit(train_X, train_Y)
        pred=r.predict(test_X)
    elif router == "ols":
        r=OLSQualityRouter().fit(train_X, train_Y)
        pred=r.predict(test_X)
    elif router == "random_forest":
        r=RandomForestQualityRouter(seed=seed).fit(train_X, train_Y)
        pred=r.predict(test_X)
    elif router == "mlp":
        r=MLPQualityRouter(seed=seed).fit(train_X, train_Y)
        pred=r.predict(test_X)
    elif router == "tfidf":
        sidx=samples.set_index("sample_id")
        r=TfidfQualityRouter().fit(sidx.loc[train_ids,"query"], train_Y)
        pred=r.predict(sidx.loc[test_ids,"query"])
    else:
        raise ValueError(f"unknown router={router}")
    selected=select_quality(pred)
    metrics=routing_metrics(test_Y, selected, best)
    out=Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload={**metrics.to_dict(), "best_single_model":best, "models":models, "router":router, "split":split, "holdout":holdout, "seed":seed, "train_ratio":train_ratio, "feature_set":feature_set, "n_train":len(train_ids), "n_test":len(test_ids), "feature_count":len(feature_cols)}
    (out/"metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    split_df = pd.concat(
        [
            pd.DataFrame({"sample_id": train_ids, "partition": "train"}),
            pd.DataFrame({"sample_id": test_ids, "partition": "test"}),
        ],
        ignore_index=True,
    )
    split_df.to_csv(out / "split.csv", index=False)
    selected_frame(test_ids, models, pred, selected, test_Y).to_csv(out/"predictions.csv", index=False)
    if hasattr(r,"feature_importance"):
        r.feature_importance().rename("importance").to_csv(out/"feature_importance.csv")
    if hasattr(r,"save") and router != "tfidf":
        r.save(out/"model")
    return payload
