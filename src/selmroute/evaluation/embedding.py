import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge

from selmroute.data.splits import dataset_ood_split, domain_ood_split, id_split
from selmroute.evaluation.metrics import best_single_model, outcome_matrix, routing_metrics, selected_frame
from selmroute.features.embeddings import load_embeddings
from selmroute.routing.policy import select_quality


def embedding_evaluate(
    samples: pd.DataFrame,
    outcomes: pd.DataFrame,
    embeddings_path: str | Path,
    output_dir: str | Path,
    *,
    split: str = "id",
    holdout: str | None = None,
    train_ratio: float = 0.7,
    seed: int = 3407,
    router: str = "ridge",
    alpha: float = 5.0,
    iterations: int = 500,
    depth: int = 6,
    learning_rate: float = 0.05,
    task_type: str = "CPU",
) -> dict:
    samples = samples.copy()
    outcomes = outcomes.copy()
    samples["sample_id"] = samples["sample_id"].astype(str)
    outcomes["sample_id"] = outcomes["sample_id"].astype(str)

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

    train_ids = np.asarray(train_ids, dtype=str)
    test_ids = np.asarray(test_ids, dtype=str)
    ids, emb, meta = load_embeddings(embeddings_path)
    pos = {sid: i for i, sid in enumerate(ids)}
    missing = [sid for sid in np.concatenate([train_ids, test_ids]) if sid not in pos]
    if missing:
        raise ValueError(f"Embedding artifact is missing {len(missing)} required samples")
    X_train = emb[[pos[sid] for sid in train_ids]]
    X_test = emb[[pos[sid] for sid in test_ids]]

    models = sorted(outcomes["model"].astype(str).unique())
    train_Y = outcome_matrix(outcomes, train_ids, models)
    test_Y = outcome_matrix(outcomes, test_ids, models)
    best = best_single_model(train_Y)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    router_name: str
    if router == "ridge":
        reg = Ridge(alpha=alpha)
        reg.fit(X_train, train_Y.to_numpy(float))
        pred = np.asarray(reg.predict(X_test), dtype=float)
        router_name = "precomputed_embedding_ridge"
        model_meta: dict[str, object] = {"alpha": alpha}
    elif router == "catboost":
        task = task_type.upper()
        if task not in {"CPU", "GPU"}:
            raise ValueError("task_type must be CPU or GPU")
        params = {
            "iterations": int(iterations),
            "depth": int(depth),
            "learning_rate": float(learning_rate),
            "random_seed": int(seed),
            "verbose": False,
            "loss_function": "MultiRMSE",
            "allow_writing_files": False,
            "task_type": task,
        }
        reg = CatBoostRegressor(**params)
        reg.fit(X_train, train_Y.to_numpy(float))
        pred = np.asarray(reg.predict(X_test), dtype=float)
        router_name = "precomputed_embedding_catboost"
        model_meta = params
        reg.save_model(out / "model.cbm")
    else:
        raise ValueError("router must be ridge or catboost")

    if pred.ndim == 1:
        pred = pred[:, None]
    selected = select_quality(pred)
    metrics = routing_metrics(test_Y, selected, best)

    payload = {
        **metrics.to_dict(),
        "best_single_model": best,
        "models": models,
        "router": router_name,
        "embedding_model": meta["model_name"],
        "embedding_revision": meta["revision"],
        "embedding_dimension": int(emb.shape[1]),
        "router_params": model_meta,
        "split": split,
        "holdout": holdout,
        "seed": seed,
        "train_ratio": train_ratio,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.concat(
        [
            pd.DataFrame({"sample_id": train_ids, "partition": "train"}),
            pd.DataFrame({"sample_id": test_ids, "partition": "test"}),
        ],
        ignore_index=True,
    ).to_csv(out / "split.csv", index=False)
    selected_frame(test_ids, models, pred, selected, test_Y).to_csv(out / "predictions.csv", index=False)
    return payload
