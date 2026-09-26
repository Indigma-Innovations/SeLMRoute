import json
from pathlib import Path

import numpy as np
import pandas as pd

from selmroute.data.splits import id_split
from selmroute.evaluation.metrics import best_single_model, outcome_matrix
from selmroute.evaluation.run import select_feature_columns
from selmroute.routing.catboost_router import CatBoostQualityRouter
from selmroute.routing.policy import select_quality


def semantic_entropy(features: pd.DataFrame, sample_ids: list[str] | np.ndarray) -> np.ndarray:
    f = features.copy()
    f["sample_id"] = f["sample_id"].astype(str)
    f = f.set_index("sample_id").reindex([str(x) for x in sample_ids])
    cols = [
        c
        for c in f.columns
        if c.startswith("jev_") and c.endswith("__entropy") and "task_family" not in c
    ]
    if not cols:
        raise ValueError("No fine-grained semantic entropy columns found")
    if f[cols].isna().all(axis=1).any():
        raise ValueError("Missing semantic entropy features")
    return f[cols].mean(axis=1).to_numpy(float)


def _calibration_split(non_holdout: pd.DataFrame, calibration_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    # id_split is stratified by dataset and therefore preserves all available
    # non-held-out datasets in the fit/calibration partition where possible.
    fit_ratio = 1.0 - calibration_ratio
    return id_split(non_holdout, fit_ratio, seed)


def _select_threshold(
    uncertainty: np.ndarray,
    router_score: np.ndarray,
    fallback_score: np.ndarray,
    *,
    min_coverage: float,
    grid_size: int,
) -> dict[str, float]:
    qs = np.linspace(min_coverage, 1.0, max(2, int(grid_size)))
    candidates: list[dict[str, float]] = []
    for q in qs:
        tau = float(np.quantile(uncertainty, q))
        route = uncertainty <= tau
        coverage = float(np.mean(route))
        if coverage + 1e-12 < min_coverage:
            continue
        score = float(np.mean(np.where(route, router_score, fallback_score)))
        candidates.append({"threshold": tau, "coverage": coverage, "score": score})
    if not candidates:
        raise ValueError("No threshold satisfies minimum coverage")
    # Prefer higher score; ties prefer higher coverage to avoid needless abstention.
    return max(candidates, key=lambda r: (r["score"], r["coverage"]))


def calibrated_domain_ood(
    samples: pd.DataFrame,
    outcomes: pd.DataFrame,
    features: pd.DataFrame,
    output_dir: str | Path,
    *,
    domains: list[str] | None = None,
    calibration_ratio: float = 0.2,
    min_coverage: float = 0.5,
    threshold_grid_size: int = 101,
    seed: int = 3407,
    feature_set: str = "semantic_manual",
) -> dict:
    samples = samples.copy()
    outcomes = outcomes.copy()
    features = features.copy()
    samples["sample_id"] = samples["sample_id"].astype(str)
    outcomes["sample_id"] = outcomes["sample_id"].astype(str)
    features["sample_id"] = features["sample_id"].astype(str)

    all_domains = sorted(samples["domain"].astype(str).unique())
    domains = domains or all_domains
    models = sorted(outcomes["model"].astype(str).unique())
    feature_cols = select_feature_columns(features, feature_set)
    indexed = features.set_index("sample_id").copy()
    rows: list[dict[str, object]] = []
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        heldout_mask = samples["domain"].astype(str).str.lower() == domain.lower()
        if not heldout_mask.any():
            raise ValueError(f"Unknown domain {domain}")
        train_pool = samples.loc[~heldout_mask].copy()
        test_ids = samples.loc[heldout_mask, "sample_id"].to_numpy(dtype=str)
        fit_ids, calibration_ids = _calibration_split(train_pool, calibration_ratio, seed)
        fit_ids = np.asarray(fit_ids, dtype=str)
        calibration_ids = np.asarray(calibration_ids, dtype=str)

        fit_Y = outcome_matrix(outcomes, fit_ids, models)
        cal_Y = outcome_matrix(outcomes, calibration_ids, models)
        test_Y = outcome_matrix(outcomes, test_ids, models)
        safe_model = best_single_model(fit_Y)

        fit_X = indexed.reindex(fit_ids).reset_index()[["sample_id", *feature_cols]].copy()
        cal_X = indexed.reindex(calibration_ids).reset_index()[["sample_id", *feature_cols]].copy()
        test_X = indexed.reindex(test_ids).reset_index()[["sample_id", *feature_cols]].copy()

        router = CatBoostQualityRouter(seed=seed).fit(fit_X, fit_Y)
        cal_pred = router.predict(cal_X)
        test_pred = router.predict(test_X)
        cal_selected = select_quality(cal_pred)
        test_selected = select_quality(test_pred)
        cal_router_score = cal_Y.to_numpy(float)[np.arange(len(cal_Y)), cal_selected]
        test_router_score = test_Y.to_numpy(float)[np.arange(len(test_Y)), test_selected]
        cal_fallback = cal_Y[safe_model].to_numpy(float)
        test_fallback = test_Y[safe_model].to_numpy(float)

        cal_u = semantic_entropy(features, calibration_ids)
        test_u = semantic_entropy(features, test_ids)
        choice = _select_threshold(
            cal_u,
            cal_router_score,
            cal_fallback,
            min_coverage=min_coverage,
            grid_size=threshold_grid_size,
        )
        tau = float(choice["threshold"])
        test_route = test_u <= tau
        final = np.where(test_route, test_router_score, test_fallback)

        row = {
            "domain": domain,
            "safe_model": safe_model,
            "threshold": tau,
            "calibration_coverage": float(choice["coverage"]),
            "calibration_score": float(choice["score"]),
            "test_coverage": float(np.mean(test_route)),
            "test_score": float(np.mean(final)),
            "router_only_score": float(np.mean(test_router_score)),
            "best_single_score": float(np.mean(test_fallback)),
            "oracle_score": float(test_Y.max(axis=1).mean()),
            "n_fit": int(len(fit_ids)),
            "n_calibration": int(len(calibration_ids)),
            "n_test": int(len(test_ids)),
        }
        rows.append(row)

        pd.DataFrame(
            {
                "sample_id": test_ids,
                "semantic_entropy": test_u,
                "route": test_route,
                "router_score": test_router_score,
                "fallback_score": test_fallback,
                "final_score": final,
            }
        ).to_csv(out / f"{domain}_predictions.csv", index=False)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "domain_results.csv", index=False)
    payload = {
        "macro_test_score": float(frame["test_score"].mean()),
        "macro_router_only_score": float(frame["router_only_score"].mean()),
        "macro_best_single_score": float(frame["best_single_score"].mean()),
        "macro_test_coverage": float(frame["test_coverage"].mean()),
        "calibration_ratio": calibration_ratio,
        "min_coverage": min_coverage,
        "threshold_grid_size": threshold_grid_size,
        "seed": seed,
        "feature_set": feature_set,
        "domains": domains,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def abstention_controls(
    predictions_root: str | Path,
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    output: str | Path,
    *,
    coverages: list[float] | tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    random_repeats: int = 1000,
    seed: int = 3407,
) -> pd.DataFrame:
    root = Path(predictions_root)
    outcomes = outcomes.copy()
    outcomes["sample_id"] = outcomes["sample_id"].astype(str)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for domain_dir in sorted(root.iterdir()):
        if not domain_dir.is_dir():
            continue
        run = domain_dir / "semantic_manual"
        pred_path = run / "predictions.csv"
        metrics_path = run / "metrics.json"
        if not pred_path.exists() or not metrics_path.exists():
            continue
        pred = pd.read_csv(pred_path, dtype={"sample_id": str})
        metrics = json.loads(metrics_path.read_text())
        safe_model = metrics["best_single_model"]
        ids = pred["sample_id"].tolist()
        fallback = (
            outcomes[(outcomes["model"] == safe_model) & outcomes["sample_id"].isin(ids)]
            .set_index("sample_id")
            .reindex(ids)["score"]
            .to_numpy(float)
        )
        router_score = pred["selected_score"].to_numpy(float)
        uncertainty = semantic_entropy(features, ids)
        entropy_order = np.argsort(uncertainty)
        advantage = router_score - fallback
        oracle_order = np.argsort(-advantage)

        for coverage in coverages:
            n_route = int(round(len(ids) * float(coverage)))
            n_route = max(0, min(len(ids), n_route))

            ent_mask = np.zeros(len(ids), dtype=bool)
            ent_mask[entropy_order[:n_route]] = True
            ent_score = float(np.mean(np.where(ent_mask, router_score, fallback)))

            oracle_mask = np.zeros(len(ids), dtype=bool)
            oracle_mask[oracle_order[:n_route]] = True
            oracle_score = float(np.mean(np.where(oracle_mask, router_score, fallback)))

            random_scores = []
            for _ in range(random_repeats):
                idx = rng.choice(len(ids), size=n_route, replace=False) if n_route else np.array([], dtype=int)
                mask = np.zeros(len(ids), dtype=bool)
                mask[idx] = True
                random_scores.append(float(np.mean(np.where(mask, router_score, fallback))))

            rows.extend(
                [
                    {"domain": domain_dir.name, "coverage": coverage, "selector": "semantic_entropy", "score": ent_score, "std": 0.0},
                    {"domain": domain_dir.name, "coverage": coverage, "selector": "random", "score": float(np.mean(random_scores)), "std": float(np.std(random_scores, ddof=1)) if len(random_scores) > 1 else 0.0},
                    {"domain": domain_dir.name, "coverage": coverage, "selector": "oracle", "score": oracle_score, "std": 0.0},
                ]
            )

    frame = pd.DataFrame(rows)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    macro = frame.groupby(["coverage", "selector"], as_index=False).agg(score=("score", "mean"), std=("std", "mean"))
    macro.to_csv(target.with_name(target.stem + "_macro.csv"), index=False)
    return macro
