import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from selmroute.evaluation.metrics import outcome_matrix, selected_frame
from selmroute.features.embeddings import load_embeddings
from selmroute.paper.ablation import build_semantic_ablation_features
from selmroute.paper.benchmark import benchmark_metrics
from selmroute.routing.catboost_router import CatBoostQualityRouter
from selmroute.routing.linear_router import RidgeQualityRouter
from selmroute.routing.policy import select_quality
from selmroute.routing.sklearn_router import MLPQualityRouter, OLSQualityRouter, RandomForestQualityRouter


def dataset_stratified_folds(samples: pd.DataFrame, n_folds: int = 5, seed: int = 3407, *, group_duplicates: bool = False) -> pd.DataFrame:
    if n_folds < 2:
        raise ValueError("n_folds must be >=2")
    s = samples.copy()
    s["sample_id"] = s["sample_id"].astype(str)
    rng = np.random.default_rng(seed)
    rows = []
    for dataset, group in s.groupby("dataset", sort=True):
        if not group_duplicates:
            ids = group["sample_id"].astype(str).to_numpy(copy=True)
            rng.shuffle(ids)
            for i, sid in enumerate(ids):
                rows.append({"sample_id": sid, "dataset": str(dataset), "fold": int(i % n_folds)})
            continue
        if "query" not in group.columns:
            raise ValueError("group-aware OOF requires a 'query' column in samples")
        q = group["query"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        work = group.assign(_q=q)
        groups = [g["sample_id"].astype(str).tolist() for _, g in work.groupby("_q", sort=True)]
        order = np.arange(len(groups))
        rng.shuffle(order)
        fold_for_group = {int(gidx): int(pos % n_folds) for pos, gidx in enumerate(order)}
        for gidx, ids in enumerate(groups):
            fold = fold_for_group[gidx]
            for sid in ids:
                rows.append({"sample_id": sid, "dataset": str(dataset), "fold": fold})
    return pd.DataFrame(rows)


def _router(name: str, seed: int):
    if name == "catboost":
        return CatBoostQualityRouter(seed=seed)
    if name == "ridge":
        return RidgeQualityRouter()
    if name == "ols":
        return OLSQualityRouter()
    if name == "random_forest":
        return RandomForestQualityRouter(seed=seed)
    if name == "mlp":
        return MLPQualityRouter(seed=seed)
    raise ValueError(f"unknown router={name}")


def semantic_oof(
    samples: pd.DataFrame,
    outcomes: pd.DataFrame,
    features: pd.DataFrame,
    output_dir: str | Path,
    *,
    representation: str = "probability_mass",
    router: str = "catboost",
    n_folds: int = 5,
    seed: int = 3407,
    semantic_prefix: str = "jev",
    group_duplicates: bool = False,
) -> dict[str, object]:
    s = samples.copy()
    o = outcomes.copy()
    f = features.copy()
    for frame in (s, o, f):
        frame["sample_id"] = frame["sample_id"].astype(str)
    Xall = build_semantic_ablation_features(f, representation, semantic_prefix=semantic_prefix)
    Xidx = Xall.set_index("sample_id")
    folds = dataset_stratified_folds(s, n_folds=n_folds, seed=seed, group_duplicates=group_duplicates)
    models = sorted(o["model"].astype(str).unique())
    preds = []
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out / "folds.csv", index=False)
    for fold in range(n_folds):
        test_ids = folds.loc[folds.fold == fold, "sample_id"].tolist()
        train_ids = folds.loc[folds.fold != fold, "sample_id"].tolist()
        train_X = Xidx.reindex(train_ids).reset_index()
        test_X = Xidx.reindex(test_ids).reset_index()
        train_Y = outcome_matrix(o, train_ids, models)
        test_Y = outcome_matrix(o, test_ids, models)
        r = _router(router, seed + fold * 1009).fit(train_X, train_Y)
        pred = r.predict(test_X)
        selected = select_quality(pred)
        pf = selected_frame(test_ids, models, pred, selected, test_Y)
        pf["fold"] = fold
        pf.to_csv(out / f"fold_{fold}_predictions.csv", index=False)
        preds.append(pf)
    combined = pd.concat(preds, ignore_index=True)
    if combined["sample_id"].duplicated().any() or len(combined) != len(s):
        raise ValueError("OOF predictions must contain each sample exactly once")
    combined.to_csv(out / "oof_predictions.csv", index=False)
    metrics = benchmark_metrics(s, o, combined)
    payload = {**metrics, "representation": representation, "router": router, "n_folds": n_folds, "seed": seed, "feature_count": len(Xall.columns)-1, "semantic_prefix": semantic_prefix, "group_duplicates": group_duplicates}
    (out / "oof_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def embedding_oof(
    samples: pd.DataFrame,
    outcomes: pd.DataFrame,
    embeddings_path: str | Path,
    output_dir: str | Path,
    *, n_folds: int = 5, seed: int = 3407, group_duplicates: bool = False,
) -> dict[str, object]:
    s = samples.copy()
    o = outcomes.copy()
    s["sample_id"] = s["sample_id"].astype(str)
    o["sample_id"] = o["sample_id"].astype(str)
    ids, emb, meta = load_embeddings(embeddings_path)
    pos = {sid:i for i,sid in enumerate(ids)}
    folds = dataset_stratified_folds(s, n_folds=n_folds, seed=seed, group_duplicates=group_duplicates)
    models = sorted(o["model"].astype(str).unique())
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out / "folds.csv", index=False)
    preds=[]
    for fold in range(n_folds):
        test_ids=folds.loc[folds.fold==fold,"sample_id"].tolist()
        train_ids=folds.loc[folds.fold!=fold,"sample_id"].tolist()
        Xtr=emb[[pos[x] for x in train_ids]]
        Xte=emb[[pos[x] for x in test_ids]]
        Ytr=outcome_matrix(o,train_ids,models)
        Yte=outcome_matrix(o,test_ids,models)
        reg=CatBoostRegressor(iterations=500,depth=6,learning_rate=.05,random_seed=seed+fold*1009,verbose=False,loss_function="MultiRMSE",allow_writing_files=False)
        reg.fit(Xtr,Ytr.to_numpy(float))
        pred=np.asarray(reg.predict(Xte),float)
        sel=select_quality(pred)
        pf=selected_frame(test_ids,models,pred,sel,Yte)
        pf["fold"]=fold
        preds.append(pf)
    combined=pd.concat(preds,ignore_index=True)
    if combined["sample_id"].duplicated().any() or len(combined)!=len(s):
        raise ValueError("OOF predictions must contain each sample exactly once")
    combined.to_csv(out/"oof_predictions.csv",index=False)
    metrics=benchmark_metrics(s,o,combined)
    payload={**metrics,"router":"embedding_catboost","embedding_model":meta["model_name"],"embedding_revision":meta["revision"],"n_folds":n_folds,"seed":seed,"group_duplicates":group_duplicates}
    (out/"oof_metrics.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    return payload
