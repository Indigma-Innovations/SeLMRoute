import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from selmroute.data.splits import id_split
from selmroute.paper.ablation import build_semantic_ablation_features
from selmroute.routing.base import numeric_matrix


def _partial_matrix(
    outcomes: pd.DataFrame,
    ids: Iterable[str],
    models: list[str],
    value: str,
) -> pd.DataFrame:
    """Pivot outcomes while preserving genuine missing model/query cells."""
    p = outcomes.pivot(index="sample_id", columns="model", values=value)
    return p.reindex(index=list(ids), columns=models).astype(float)


def _macro_accuracy(samples: pd.DataFrame, ids: list[str], scores: np.ndarray) -> float:
    meta = samples.set_index("sample_id").reindex(ids)
    frame = pd.DataFrame(
        {"dataset": meta["dataset"].astype(str).to_numpy(), "score": np.asarray(scores, float)}
    )
    return float(frame.groupby("dataset", sort=True)["score"].mean().mean())


def _pareto_flags(scores: np.ndarray, costs: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, float)
    costs = np.asarray(costs, float)
    flags = np.ones(len(scores), dtype=bool)
    for i in range(len(scores)):
        dominated = (
            (scores >= scores[i])
            & (costs <= costs[i])
            & ((scores > scores[i]) | (costs < costs[i]))
        ).any()
        flags[i] = not dominated
    return flags


def _fit_per_model_quality(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    seed: int,
) -> list[CatBoostRegressor]:
    cols = [c for c in X.columns if c != "sample_id" and not c.startswith("meta_")]
    xarr = numeric_matrix(X, cols)
    models: list[CatBoostRegressor] = []
    for j in range(Y.shape[1]):
        y = Y.iloc[:, j].to_numpy(float)
        mask = np.isfinite(y)
        if mask.sum() < 2:
            raise ValueError(
                f"Model {Y.columns[j]!r} has fewer than two observed training outcomes; "
                "cannot fit non-rectangular quality model."
            )
        model = CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            random_seed=seed,
            verbose=False,
            loss_function="RMSE",
            allow_writing_files=False,
        )
        model.fit(xarr[mask], y[mask])
        models.append(model)
    return models


def _predict_per_model_quality(models: list[CatBoostRegressor], X: pd.DataFrame) -> np.ndarray:
    cols = [c for c in X.columns if c != "sample_id" and not c.startswith("meta_")]
    xarr = numeric_matrix(X, cols)
    return np.column_stack([np.asarray(m.predict(xarr), float) for m in models])


def _fit_per_model_cost(X: pd.DataFrame, C: pd.DataFrame) -> list[object]:
    cols = [c for c in X.columns if c != "sample_id" and not c.startswith("meta_")]
    xarr = numeric_matrix(X, cols)
    models: list[object] = []
    for j in range(C.shape[1]):
        y = C.iloc[:, j].to_numpy(float)
        mask = np.isfinite(y)
        if mask.sum() < 2:
            raise ValueError(
                f"Model {C.columns[j]!r} has fewer than two observed training costs; "
                "cannot fit non-rectangular cost model."
            )
        model = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
        model.fit(xarr[mask], y[mask])
        models.append(model)
    return models


def _predict_per_model_cost(models: list[object], X: pd.DataFrame) -> np.ndarray:
    cols = [c for c in X.columns if c != "sample_id" and not c.startswith("meta_")]
    xarr = numeric_matrix(X, cols)
    return np.column_stack([np.asarray(m.predict(xarr), float) for m in models])


def _select_baseline_model(
    samples: pd.DataFrame,
    test_ids: list[str],
    qarr: np.ndarray,
    availability: np.ndarray,
    models: list[str],
    baseline_model: str | None,
) -> tuple[str, int, float]:
    if baseline_model is not None:
        if baseline_model not in models:
            raise ValueError(f"baseline_model={baseline_model!r} is not in candidate pool")
        j = models.index(baseline_model)
        if not availability[:, j].all():
            missing = int((~availability[:, j]).sum())
            raise ValueError(
                f"baseline_model={baseline_model!r} is unavailable on {missing} test samples; "
                "a Figure-6/8 style baseline must cover every test sample."
            )
        return baseline_model, j, _macro_accuracy(samples, test_ids, qarr[:, j])

    candidates: list[tuple[float, str, int]] = []
    for j, model in enumerate(models):
        if availability[:, j].all():
            candidates.append((_macro_accuracy(samples, test_ids, qarr[:, j]), model, j))
    if not candidates:
        raise ValueError(
            "No candidate model has complete test coverage. Pass --baseline-model for a known "
            "complete reference model or revise the selected benchmark pool."
        )
    acc, model, j = max(candidates, key=lambda x: x[0])
    return model, j, acc


def _selected_frame(
    samples: pd.DataFrame,
    test_ids: list[str],
    models: list[str],
    selected: np.ndarray,
    qarr: np.ndarray,
    carr: np.ndarray,
    availability: np.ndarray,
) -> pd.DataFrame:
    meta = samples.set_index("sample_id").reindex(test_ids)
    rows = np.arange(len(test_ids))
    return pd.DataFrame(
        {
            "sample_id": test_ids,
            "dataset": meta["dataset"].astype(str).to_numpy(),
            "selected_model": [models[j] for j in selected],
            "selected_score": qarr[rows, selected],
            "selected_cost": carr[rows, selected],
            "available_models": availability.sum(axis=1),
        }
    )


def paper_cost_evaluate(
    samples: pd.DataFrame,
    outcomes: pd.DataFrame,
    features: pd.DataFrame,
    output_dir: str | Path,
    *,
    seed: int = 3407,
    train_ratio: float = 0.7,
    representation: str = "probability_mass",
    lambdas: list[float] | None = None,
    allow_missing: bool = False,
    baseline_model: str | None = None,
) -> dict[str, object]:
    """Evaluate SeLMRoute on the LLMRouterBench performance-cost setting.

    With ``allow_missing=False`` this preserves the v0.7 rectangular-pool path.
    With ``allow_missing=True`` it supports the official 10-dataset flagship pool:
    each candidate is trained only on observed outcomes and unavailable candidates
    are masked at selection time. No score/cost imputation is used for evaluation.

    PerfGain and CostSave use Dataset-Avg accuracy and total test-set candidate-model
    inference cost, matching the public LLMRouterBench definitions. Router/JEV
    overhead is intentionally reported separately elsewhere.
    """
    s, o, f = samples.copy(), outcomes.copy(), features.copy()
    for frame in (s, o, f):
        frame["sample_id"] = frame["sample_id"].astype(str)
    o["model"] = o["model"].astype(str)
    if "cost" not in o.columns:
        raise ValueError("outcomes must contain per-query model cost")

    train_ids, test_ids = id_split(s, train_ratio, seed)
    train_ids = list(map(str, train_ids))
    test_ids = list(map(str, test_ids))
    models = sorted(o.model.unique())

    Ytr = _partial_matrix(o, train_ids, models, "score")
    Yte = _partial_matrix(o, test_ids, models, "score")
    Ctr = _partial_matrix(o, train_ids, models, "cost")
    Cte = _partial_matrix(o, test_ids, models, "cost")

    if not allow_missing:
        if Ytr.isna().any().any() or Yte.isna().any().any() or Ctr.isna().any().any() or Cte.isna().any().any():
            raise ValueError(
                "Missing model/query outcomes detected. Re-ingest with complete cases or pass "
                "--allow-missing for the non-rectangular 10-dataset flagship protocol."
            )

    X = build_semantic_ablation_features(f, representation).set_index("sample_id")
    Xtr = X.reindex(train_ids).reset_index()
    Xte = X.reindex(test_ids).reset_index()
    feature_cols = [c for c in Xtr.columns if c != "sample_id" and not c.startswith("meta_")]
    if Xtr[feature_cols].isna().all(axis=1).any() or Xte[feature_cols].isna().all(axis=1).any():
        raise ValueError("Missing feature rows for some cost-benchmark samples")

    if allow_missing:
        qmodels = _fit_per_model_quality(Xtr, Ytr, seed=seed)
        cmodels = _fit_per_model_cost(Xtr, Ctr)
        qp = _predict_per_model_quality(qmodels, Xte)
        cp = np.maximum(0.0, _predict_per_model_cost(cmodels, Xte))
    else:
        # Preserve the original multi-output estimators for rectangular pools.
        from selmroute.routing.catboost_router import CatBoostQualityRouter
        from selmroute.routing.linear_router import RidgeQualityRouter

        qr = CatBoostQualityRouter(seed=seed).fit(Xtr, Ytr)
        cr = RidgeQualityRouter(alpha=2.0).fit(Xtr, Ctr)
        qp = np.asarray(qr.predict(Xte), float)
        cp = np.maximum(0.0, np.asarray(cr.predict(Xte), float))

    qarr_full = Yte.to_numpy(float)
    carr_full = Cte.to_numpy(float)
    availability_full = np.isfinite(qarr_full) & np.isfinite(carr_full)
    if (~availability_full.any(axis=1)).any():
        raise ValueError("At least one test sample has no available candidate model")

    # Figure-6/8 metrics are defined relative to a fixed reference model (GPT-5
    # in LLMRouterBench). If that reference is genuinely missing for a benchmark
    # instance, compare the router and baseline on the same supported subset rather
    # than imputing a score/cost or silently switching baselines.
    baseline_missing_test = 0
    if baseline_model is not None:
        if baseline_model not in models:
            raise ValueError(f"baseline_model={baseline_model!r} is not in candidate pool")
        baseline_idx = models.index(baseline_model)
        eval_mask = availability_full[:, baseline_idx]
        baseline_missing_test = int((~eval_mask).sum())
        if not eval_mask.any():
            raise ValueError(
                f"baseline_model={baseline_model!r} has no observed test outcomes; "
                "cannot define benchmark-relative metrics."
            )
    else:
        eval_mask = np.ones(len(test_ids), dtype=bool)

    eval_test_ids = [sid for sid, keep in zip(test_ids, eval_mask, strict=True) if keep]
    qarr = qarr_full[eval_mask]
    carr = carr_full[eval_mask]
    availability = availability_full[eval_mask]
    qp = qp[eval_mask]
    cp = cp[eval_mask]

    # Normalize predictions on the actual evaluation support. Predictions exist for
    # every model, but unavailable candidate/query pairs are masked during choice.
    qmin, qmax = float(np.nanmin(qp)), float(np.nanmax(qp))
    cmin, cmax = float(np.nanmin(cp)), float(np.nanmax(cp))
    qn = (qp - qmin) / max(qmax - qmin, 1e-12)
    cn = (cp - cmin) / max(cmax - cmin, 1e-12)

    lambdas = lambdas or [i / 20 for i in range(21)]
    rows: list[dict[str, object]] = []
    selected_by_lambda: dict[float, np.ndarray] = {}
    for lam in lambdas:
        utility = (1.0 - lam) * qn - lam * cn
        utility = np.where(availability, utility, -np.inf)
        selected = np.argmax(utility, axis=1)
        selected_by_lambda[float(lam)] = selected
        rr = np.arange(len(selected))
        achieved = qarr[rr, selected]
        costs = carr[rr, selected]
        if not np.isfinite(achieved).all() or not np.isfinite(costs).all():
            raise AssertionError("availability masking failed: selected a missing outcome")
        rows.append(
            {
                "lambda": float(lam),
                "AvgAcc": _macro_accuracy(s, eval_test_ids, achieved),
                "total_cost": float(costs.sum()),
                "mean_cost": float(costs.mean()),
                "min_available_models": int(availability.sum(axis=1).min()),
                "mean_available_models": float(availability.sum(axis=1).mean()),
            }
        )

    sweep = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)
    sweep["pareto"] = _pareto_flags(
        sweep["AvgAcc"].to_numpy(float), sweep["total_cost"].to_numpy(float)
    )

    best_model, best_idx, best_acc = _select_baseline_model(
        s, eval_test_ids, qarr, availability, models, baseline_model
    )
    best_cost = float(carr[:, best_idx].sum())

    theta_star = sweep.loc[sweep.AvgAcc.idxmax()]
    eligible = sweep[sweep.AvgAcc >= best_acc - 1e-12]
    perf_gain = float(theta_star.AvgAcc / best_acc - 1.0) if best_acc else float("nan")
    cost_save = float(1.0 - eligible.total_cost.min() / best_cost) if len(eligible) and best_cost > 0 else None

    theta_dagger = None if eligible.empty else eligible.loc[eligible.total_cost.idxmin()]
    payload: dict[str, object] = {
        "best_single_model": best_model,
        "best_single_AvgAcc": best_acc * 100,
        "baseline_model": best_model,
        "baseline_AvgAcc": best_acc * 100,
        "best_single_total_cost": best_cost,
        "PerfGain_pct": perf_gain * 100,
        "CostSave_pct": None if cost_save is None else cost_save * 100,
        "theta_star_lambda": float(theta_star["lambda"]),
        "theta_star_AvgAcc": float(theta_star.AvgAcc * 100),
        "theta_star_total_cost": float(theta_star.total_cost),
        "theta_dagger_lambda": None if theta_dagger is None else float(theta_dagger["lambda"]),
        "theta_dagger_AvgAcc": None if theta_dagger is None else float(theta_dagger.AvgAcc * 100),
        "theta_dagger_total_cost": None if theta_dagger is None else float(theta_dagger.total_cost),
        "representation": representation,
        "seed": seed,
        "n_train": len(train_ids),
        "n_test": len(eval_test_ids),
        "n_test_split": len(test_ids),
        "n_test_eval": len(eval_test_ids),
        "baseline_missing_test": baseline_missing_test,
        "baseline_support_policy": "evaluate router and fixed baseline on the same baseline-supported test subset; no score/cost imputation",
        "n_datasets": int(s.set_index("sample_id").reindex(eval_test_ids)["dataset"].nunique()),
        "n_models": len(models),
        "allow_missing": bool(allow_missing),
        "min_available_models_test": int(availability.sum(axis=1).min()),
        "mean_available_models_test": float(availability.sum(axis=1).mean()),
        "feature_count": len(feature_cols),
        "cost_definition": "candidate-model inference cost only; router/JEV overhead excluded to match LLMRouterBench performance-cost convention",
        "accuracy_definition": "Dataset-Avg: unweighted mean of per-dataset mean raw scores",
        "pareto_definition": "SeLMRoute lambda-sweep nondominated points (higher AvgAcc, lower total_cost)",
        "ParetoDist": None,
        "ParetoDist_note": "Benchmark-comparable ParetoDist requires a cross-system reference frontier; SeLMRoute writes its frontier but does not invent competitor sweep coordinates.",
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out / "cost_sweep.csv", index=False)
    sweep[sweep.pareto].sort_values(["total_cost", "AvgAcc"]).to_csv(
        out / "pareto_frontier.csv", index=False
    )

    coverage = pd.DataFrame(availability, index=eval_test_ids, columns=models)
    pd.DataFrame(
        {
            "model": models,
            "test_available": coverage.sum(axis=0).to_numpy(int),
            "test_coverage_pct": 100.0 * coverage.mean(axis=0).to_numpy(float),
            "train_available": Ytr.notna().sum(axis=0).to_numpy(int),
        }
    ).to_csv(out / "model_coverage.csv", index=False)

    split_df = pd.concat(
        [
            pd.DataFrame({"sample_id": train_ids, "partition": "train"}),
            pd.DataFrame({"sample_id": test_ids, "partition": "test"}),
        ],
        ignore_index=True,
    )
    eval_set = set(eval_test_ids)
    split_df["evaluation_included"] = split_df.apply(
        lambda row: row["partition"] == "train" or str(row["sample_id"]) in eval_set, axis=1
    )
    split_df.to_csv(out / "split.csv", index=False)

    if baseline_missing_test:
        excluded_ids = [sid for sid, keep in zip(test_ids, eval_mask, strict=True) if not keep]
        excluded = s.set_index("sample_id").reindex(excluded_ids).reset_index()
        excluded[[c for c in ["sample_id", "dataset", "split"] if c in excluded.columns]].to_csv(
            out / "baseline_excluded_samples.csv", index=False
        )

    star_sel = selected_by_lambda[float(theta_star["lambda"])]
    _selected_frame(s, eval_test_ids, models, star_sel, qarr, carr, availability).to_csv(
        out / "theta_star_predictions.csv", index=False
    )
    if theta_dagger is not None:
        dagger_sel = selected_by_lambda[float(theta_dagger["lambda"])]
        _selected_frame(s, eval_test_ids, models, dagger_sel, qarr, carr, availability).to_csv(
            out / "theta_dagger_predictions.csv", index=False
        )

    (out / "cost_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload



def _fit_predict_components(
    Xfit: pd.DataFrame,
    Yfit: pd.DataFrame,
    Cfit: pd.DataFrame,
    Xeval: pd.DataFrame,
    *,
    seed: int,
    allow_missing: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit quality/cost predictors and return eval + fit predictions.

    Fit-set predictions are used only to derive training-supported normalization
    ranges for the cost/quality utility. No held-out labels are used for scaling.
    """
    if allow_missing:
        qmodels = _fit_per_model_quality(Xfit, Yfit, seed=seed)
        cmodels = _fit_per_model_cost(Xfit, Cfit)
        qp_eval = _predict_per_model_quality(qmodels, Xeval)
        cp_eval = np.maximum(0.0, _predict_per_model_cost(cmodels, Xeval))
        qp_fit = _predict_per_model_quality(qmodels, Xfit)
        cp_fit = np.maximum(0.0, _predict_per_model_cost(cmodels, Xfit))
    else:
        from selmroute.routing.catboost_router import CatBoostQualityRouter
        from selmroute.routing.linear_router import RidgeQualityRouter

        qr = CatBoostQualityRouter(seed=seed).fit(Xfit, Yfit)
        cr = RidgeQualityRouter(alpha=2.0).fit(Xfit, Cfit)
        qp_eval = np.asarray(qr.predict(Xeval), float)
        cp_eval = np.maximum(0.0, np.asarray(cr.predict(Xeval), float))
        qp_fit = np.asarray(qr.predict(Xfit), float)
        cp_fit = np.maximum(0.0, np.asarray(cr.predict(Xfit), float))
    return qp_eval, cp_eval, qp_fit, cp_fit


def _normalization_from_fit(qp_fit: np.ndarray, cp_fit: np.ndarray) -> dict[str, float]:
    return {
        "qmin": float(np.nanmin(qp_fit)),
        "qmax": float(np.nanmax(qp_fit)),
        "cmin": float(np.nanmin(cp_fit)),
        "cmax": float(np.nanmax(cp_fit)),
    }


def _normalize_predictions(
    qp: np.ndarray,
    cp: np.ndarray,
    scale: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    qn = (qp - scale["qmin"]) / max(scale["qmax"] - scale["qmin"], 1e-12)
    cn = (cp - scale["cmin"]) / max(scale["cmax"] - scale["cmin"], 1e-12)
    return qn, cn


def _evaluation_support(
    Y: pd.DataFrame,
    C: pd.DataFrame,
    models: list[str],
    baseline_model: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    qarr_full = Y.to_numpy(float)
    carr_full = C.to_numpy(float)
    availability_full = np.isfinite(qarr_full) & np.isfinite(carr_full)
    if (~availability_full.any(axis=1)).any():
        raise ValueError("At least one evaluation sample has no available candidate model")
    baseline_missing = 0
    if baseline_model is not None:
        if baseline_model not in models:
            raise ValueError(f"baseline_model={baseline_model!r} is not in candidate pool")
        baseline_idx = models.index(baseline_model)
        eval_mask = availability_full[:, baseline_idx]
        baseline_missing = int((~eval_mask).sum())
        if not eval_mask.any():
            raise ValueError(
                f"baseline_model={baseline_model!r} has no observed evaluation outcomes"
            )
    else:
        eval_mask = np.ones(len(Y), dtype=bool)
    return qarr_full, carr_full, availability_full, baseline_missing, eval_mask


def _sweep_metrics(
    samples: pd.DataFrame,
    eval_ids: list[str],
    models: list[str],
    qarr: np.ndarray,
    carr: np.ndarray,
    availability: np.ndarray,
    qp: np.ndarray,
    cp: np.ndarray,
    scale: dict[str, float],
    lambdas: list[float],
) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    qn, cn = _normalize_predictions(qp, cp, scale)
    rows: list[dict[str, object]] = []
    selected_by_lambda: dict[float, np.ndarray] = {}
    for lam in lambdas:
        utility = (1.0 - lam) * qn - lam * cn
        utility = np.where(availability, utility, -np.inf)
        selected = np.argmax(utility, axis=1)
        selected_by_lambda[float(lam)] = selected
        rr = np.arange(len(selected))
        achieved = qarr[rr, selected]
        costs = carr[rr, selected]
        if not np.isfinite(achieved).all() or not np.isfinite(costs).all():
            raise AssertionError("availability masking failed: selected a missing outcome")
        rows.append(
            {
                "lambda": float(lam),
                "AvgAcc": _macro_accuracy(samples, eval_ids, achieved),
                "total_cost": float(costs.sum()),
                "mean_cost": float(costs.mean()),
                "min_available_models": int(availability.sum(axis=1).min()),
                "mean_available_models": float(availability.sum(axis=1).mean()),
            }
        )
    sweep = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)
    sweep["pareto"] = _pareto_flags(
        sweep["AvgAcc"].to_numpy(float), sweep["total_cost"].to_numpy(float)
    )
    return sweep, selected_by_lambda


def paper_cost_evaluate_validated(
    samples: pd.DataFrame,
    outcomes: pd.DataFrame,
    features: pd.DataFrame,
    output_dir: str | Path,
    *,
    seed: int = 3407,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.2,
    representation: str = "probability_mass",
    lambdas: list[float] | None = None,
    allow_missing: bool = False,
    baseline_model: str | None = None,
) -> dict[str, object]:
    """Leakage-resistant performance-cost evaluation.

    The outer test partition is never used for operating-point selection.
    Lambda is selected on a dataset-stratified validation split carved from the
    outer training partition. Predictors are then refit on all outer-training
    samples and the selected lambda(s) are evaluated exactly once on outer test.

    ``theta_star`` maximizes validation Dataset-Avg accuracy. ``theta_dagger`` is
    the cheapest validation configuration whose validation accuracy matches or
    exceeds the fixed baseline. Test CostSave is reported only when the fixed
    validation-selected theta_dagger also matches/exceeds the baseline on test.
    A raw cost reduction is retained separately for diagnostic transparency.
    """
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in (0, 1)")
    s, o, f = samples.copy(), outcomes.copy(), features.copy()
    for frame in (s, o, f):
        frame["sample_id"] = frame["sample_id"].astype(str)
    o["model"] = o["model"].astype(str)
    if "cost" not in o.columns:
        raise ValueError("outcomes must contain per-query model cost")

    outer_train_ids, test_ids = id_split(s, train_ratio, seed, group_duplicates=True)
    outer_train_ids = list(map(str, outer_train_ids))
    test_ids = list(map(str, test_ids))
    train_samples = s[s["sample_id"].isin(outer_train_ids)].copy()
    inner_fit_ids, val_ids = id_split(train_samples, 1.0 - validation_ratio, seed + 104729, group_duplicates=True)
    inner_fit_ids = list(map(str, inner_fit_ids))
    val_ids = list(map(str, val_ids))
    models = sorted(o.model.unique())
    lambdas = lambdas or [i / 20 for i in range(21)]

    Xall = build_semantic_ablation_features(f, representation).set_index("sample_id")
    def x(ids: list[str]) -> pd.DataFrame:
        z = Xall.reindex(ids).reset_index()
        cols = [c for c in z.columns if c != "sample_id" and not c.startswith("meta_")]
        if z[cols].isna().all(axis=1).any():
            raise ValueError("Missing feature rows for some cost-benchmark samples")
        return z

    # ---------------- validation selection: inner-fit -> validation ----------------
    Xfit, Xval = x(inner_fit_ids), x(val_ids)
    Yfit = _partial_matrix(o, inner_fit_ids, models, "score")
    Cfit = _partial_matrix(o, inner_fit_ids, models, "cost")
    Yval = _partial_matrix(o, val_ids, models, "score")
    Cval = _partial_matrix(o, val_ids, models, "cost")
    if not allow_missing and (
        Yfit.isna().any().any() or Cfit.isna().any().any()
        or Yval.isna().any().any() or Cval.isna().any().any()
    ):
        raise ValueError("Missing model/query outcomes detected; pass --allow-missing")

    qp_val, cp_val, qp_fit, cp_fit = _fit_predict_components(
        Xfit, Yfit, Cfit, Xval, seed=seed, allow_missing=allow_missing
    )
    val_qfull, val_cfull, val_afull, val_baseline_missing, val_mask = _evaluation_support(
        Yval, Cval, models, baseline_model
    )
    val_eval_ids = [sid for sid, keep in zip(val_ids, val_mask, strict=True) if keep]
    val_qarr = val_qfull[val_mask]
    val_carr = val_cfull[val_mask]
    val_availability = val_afull[val_mask]
    qp_val = qp_val[val_mask]
    cp_val = cp_val[val_mask]
    val_scale = _normalization_from_fit(qp_fit, cp_fit)
    val_sweep, val_selected = _sweep_metrics(
        s, val_eval_ids, models, val_qarr, val_carr, val_availability,
        qp_val, cp_val, val_scale, lambdas,
    )
    val_best_model, val_best_idx, val_best_acc = _select_baseline_model(
        s, val_eval_ids, val_qarr, val_availability, models, baseline_model
    )
    val_best_cost = float(val_carr[:, val_best_idx].sum())
    theta_star_val = val_sweep.loc[val_sweep.AvgAcc.idxmax()]
    val_eligible = val_sweep[val_sweep.AvgAcc >= val_best_acc - 1e-12]
    theta_dagger_val = None if val_eligible.empty else val_eligible.loc[val_eligible.total_cost.idxmin()]
    star_lambda = float(theta_star_val["lambda"])
    dagger_lambda = None if theta_dagger_val is None else float(theta_dagger_val["lambda"])

    # ---------------- final fit: outer train -> untouched test ----------------
    Xtrain, Xtest = x(outer_train_ids), x(test_ids)
    Ytrain = _partial_matrix(o, outer_train_ids, models, "score")
    Ctrain = _partial_matrix(o, outer_train_ids, models, "cost")
    Ytest = _partial_matrix(o, test_ids, models, "score")
    Ctest = _partial_matrix(o, test_ids, models, "cost")
    if not allow_missing and (
        Ytrain.isna().any().any() or Ctrain.isna().any().any()
        or Ytest.isna().any().any() or Ctest.isna().any().any()
    ):
        raise ValueError("Missing model/query outcomes detected; pass --allow-missing")

    qp_test, cp_test, qp_train, cp_train = _fit_predict_components(
        Xtrain, Ytrain, Ctrain, Xtest, seed=seed, allow_missing=allow_missing
    )
    test_qfull, test_cfull, test_afull, baseline_missing_test, test_mask = _evaluation_support(
        Ytest, Ctest, models, baseline_model
    )
    eval_test_ids = [sid for sid, keep in zip(test_ids, test_mask, strict=True) if keep]
    qarr = test_qfull[test_mask]
    carr = test_cfull[test_mask]
    availability = test_afull[test_mask]
    qp_test = qp_test[test_mask]
    cp_test = cp_test[test_mask]
    test_scale = _normalization_from_fit(qp_train, cp_train)

    # Predeclared lambda grid may be written diagnostically, but headline operating
    # points are the validation-selected lambdas above, never test-selected.
    test_sweep, test_selected = _sweep_metrics(
        s, eval_test_ids, models, qarr, carr, availability,
        qp_test, cp_test, test_scale, lambdas,
    )
    best_model, best_idx, best_acc = _select_baseline_model(
        s, eval_test_ids, qarr, availability, models, baseline_model
    )
    best_cost = float(carr[:, best_idx].sum())

    def row_for_lambda(lam: float) -> pd.Series:
        m = np.isclose(test_sweep["lambda"].to_numpy(float), lam)
        if not m.any():
            raise AssertionError(f"selected lambda {lam} missing from test sweep")
        return test_sweep.loc[m].iloc[0]

    star_test = row_for_lambda(star_lambda)
    perf_gain = float(star_test.AvgAcc / best_acc - 1.0) if best_acc else float("nan")

    dagger_test = None
    raw_cost_save = None
    cost_save = None
    dagger_meets_test_baseline = None
    if dagger_lambda is not None:
        dagger_test = row_for_lambda(dagger_lambda)
        raw_cost_save = float(1.0 - dagger_test.total_cost / best_cost) if best_cost > 0 else None
        dagger_meets_test_baseline = bool(dagger_test.AvgAcc >= best_acc - 1e-12)
        cost_save = raw_cost_save if dagger_meets_test_baseline else None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    val_sweep.to_csv(out / "validation_lambda_sweep.csv", index=False)
    test_sweep.assign(selection_role="diagnostic_only").to_csv(out / "test_lambda_sweep_diagnostic.csv", index=False)
    val_sweep[val_sweep.pareto].to_csv(out / "validation_pareto_frontier.csv", index=False)

    star_sel = test_selected[star_lambda]
    _selected_frame(s, eval_test_ids, models, star_sel, qarr, carr, availability).to_csv(
        out / "theta_star_predictions.csv", index=False
    )
    if dagger_lambda is not None:
        dagger_sel = test_selected[dagger_lambda]
        _selected_frame(s, eval_test_ids, models, dagger_sel, qarr, carr, availability).to_csv(
            out / "theta_dagger_predictions.csv", index=False
        )

    split_rows = []
    for sid in inner_fit_ids:
        split_rows.append({"sample_id": sid, "partition": "inner_fit"})
    for sid in val_ids:
        split_rows.append({"sample_id": sid, "partition": "validation"})
    for sid in test_ids:
        split_rows.append({"sample_id": sid, "partition": "test"})
    pd.DataFrame(split_rows).to_csv(out / "split.csv", index=False)

    payload: dict[str, object] = {
        "protocol": "nested_validation_lambda_selection",
        "selection_source": "validation_only",
        "split_grouping": "dataset + normalized router query",
        "best_single_model": best_model,
        "best_single_AvgAcc": best_acc * 100,
        "baseline_model": best_model,
        "baseline_AvgAcc": best_acc * 100,
        "best_single_total_cost": best_cost,
        "PerfGain_pct": perf_gain * 100,
        "CostSave_pct": None if cost_save is None else cost_save * 100,
        "CostSave_raw_pct": None if raw_cost_save is None else raw_cost_save * 100,
        "theta_star_lambda": star_lambda,
        "theta_star_validation_AvgAcc": float(theta_star_val.AvgAcc * 100),
        "theta_star_AvgAcc": float(star_test.AvgAcc * 100),
        "theta_star_total_cost": float(star_test.total_cost),
        "theta_dagger_lambda": dagger_lambda,
        "theta_dagger_validation_AvgAcc": None if theta_dagger_val is None else float(theta_dagger_val.AvgAcc * 100),
        "theta_dagger_AvgAcc": None if dagger_test is None else float(dagger_test.AvgAcc * 100),
        "theta_dagger_total_cost": None if dagger_test is None else float(dagger_test.total_cost),
        "theta_dagger_test_meets_baseline": dagger_meets_test_baseline,
        "validation_best_single_AvgAcc": val_best_acc * 100,
        "validation_best_single_total_cost": val_best_cost,
        "representation": representation,
        "seed": seed,
        "n_outer_train": len(outer_train_ids),
        "n_inner_fit": len(inner_fit_ids),
        "n_validation": len(val_ids),
        "n_test_split": len(test_ids),
        "n_test_eval": len(eval_test_ids),
        "validation_ratio_within_outer_train": validation_ratio,
        "baseline_missing_validation": val_baseline_missing,
        "baseline_missing_test": baseline_missing_test,
        "baseline_support_policy": "validation and test each use the same baseline-supported subset for router and fixed baseline; no score/cost imputation",
        "n_datasets": int(s.loc[s.sample_id.isin(eval_test_ids), "dataset"].nunique()),
        "n_models": len(models),
        "allow_missing": allow_missing,
        "min_available_models_test": int(availability.sum(axis=1).min()),
        "mean_available_models_test": float(availability.sum(axis=1).mean()),
        "feature_count": len([c for c in Xtrain.columns if c != "sample_id" and not c.startswith("meta_")]),
        "cost_definition": "candidate-model inference cost only; router/JEV overhead excluded to match LLMRouterBench performance-cost convention",
        "accuracy_definition": "Dataset-Avg: unweighted mean of per-dataset mean raw scores",
        "lambda_selection_definition": "theta_star maximizes validation AvgAcc; theta_dagger minimizes validation total cost subject to validation AvgAcc >= fixed-baseline validation AvgAcc",
        "CostSave_definition": "test cost reduction of validation-selected theta_dagger, reported as CostSave only if its untouched-test AvgAcc >= fixed-baseline untouched-test AvgAcc",
        "test_sweep_note": "test_lambda_sweep_diagnostic.csv contains the predeclared lambda grid for plotting/diagnostics only and is not used for headline operating-point selection",
        "ParetoDist": None,
    }
    (out / "cost_metrics_validated.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
