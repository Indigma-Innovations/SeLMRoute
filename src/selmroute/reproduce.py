import json
from pathlib import Path

import numpy as np
import pandas as pd

from selmroute.evaluation.calibrated_selective import calibrated_domain_ood
from selmroute.evaluation.embedding import embedding_evaluate
from selmroute.evaluation.run import train_evaluate
from selmroute.evaluation.selective import aurc, selective_risk
from selmroute.paper.ablation import build_semantic_ablation_features
from selmroute.paper.benchmark import benchmark_metrics
from selmroute.paper.cost_benchmark import paper_cost_evaluate_validated
from selmroute.paper.grouped_main import rerun_main_grouped
from selmroute.paper.profiling import profile_jev_features
from selmroute.paper.statistics import paired_prediction_test
from selmroute.validation import default_paths, validate_release

# from selmroute.paper.probe_ablation import leave_one_probe_out

SEEDS = (42, 999, 2024, 2025, 3407)
COST_SEEDS = (42, 3407, 0, 1, 2)


def _load(root: Path):
    p = default_paths(root)
    samples = pd.read_csv(p["perf_samples"], dtype={"sample_id": str})
    outcomes = pd.read_csv(p["perf_outcomes"], dtype={"sample_id": str})
    jev = pd.read_csv(p["jev_features"], dtype={"sample_id": str})
    laya = pd.read_csv(p["laya_features"], dtype={"sample_id": str})
    return p, samples, outcomes, jev, laya


def _concat_predictions(run_dirs: list[Path], output: Path) -> pd.DataFrame:
    frames = [pd.read_csv(d / "predictions.csv", dtype={"sample_id": str}) for d in run_dirs]
    frame = pd.concat(frames, ignore_index=True)
    if frame["sample_id"].duplicated().any():
        raise ValueError("Combined OOD predictions contain duplicate sample IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def _ood_system(
    samples: pd.DataFrame,
    outcomes: pd.DataFrame,
    jev_features: pd.DataFrame,
    embeddings: Path,
    *,
    protocol: str,
    system: str,
    output_root: Path,
    seed: int = 3407,
) -> dict[str, object]:
    if protocol not in {"dataset-ood", "domain-ood"}:
        raise ValueError(protocol)
    holdouts = sorted(samples["dataset" if protocol == "dataset-ood" else "domain"].astype(str).unique())
    run_dirs: list[Path] = []

    if system == "probability_mass":
        feature_frame = build_semantic_ablation_features(jev_features, "probability_mass")
    elif system == "full":
        feature_frame = build_semantic_ablation_features(jev_features, "full")
    else:
        feature_frame = jev_features

    for holdout in holdouts:
        run_dir = output_root / protocol / system / holdout
        run_dirs.append(run_dir)
        if system == "gte_catboost":
            embedding_evaluate(
                samples, outcomes, embeddings, run_dir,
                split=protocol, holdout=holdout, seed=seed, router="catboost",
            )
        elif system == "tfidf":
            train_evaluate(
                samples, outcomes, feature_frame, run_dir,
                router="tfidf", split=protocol, holdout=holdout, seed=seed, feature_set="all",
            )
        elif system == "domain_only":
            train_evaluate(
                samples, outcomes, feature_frame, run_dir,
                router="catboost", split=protocol, holdout=holdout, seed=seed, feature_set="domain_only",
            )
        else:
            train_evaluate(
                samples, outcomes, feature_frame, run_dir,
                router="catboost", split=protocol, holdout=holdout, seed=seed, feature_set="all",
            )

    combined = _concat_predictions(run_dirs, output_root / protocol / system / "oof_predictions.csv")
    metrics = benchmark_metrics(samples, outcomes, combined)
    payload = {"protocol": protocol, "system": system, "seed": seed, **metrics}
    (output_root / protocol / system / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def reproduce_ood(
    samples: pd.DataFrame,
    outcomes: pd.DataFrame,
    jev_features: pd.DataFrame,
    embeddings: Path,
    output_root: Path,
) -> pd.DataFrame:
    systems = ("probability_mass", "full", "gte_catboost", "tfidf", "domain_only")
    rows: list[dict[str, object]] = []
    for protocol in ("dataset-ood", "domain-ood"):
        for system in systems:
            rows.append(_ood_system(
                samples, outcomes, jev_features, embeddings,
                protocol=protocol, system=system, output_root=output_root,
            ))
    frame = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "summary.csv", index=False)
    return frame


def reproduce_selective(samples: pd.DataFrame, features: pd.DataFrame, pm_oof: Path, output_root: Path) -> dict[str, object]:
    pred = pd.read_csv(pm_oof, dtype={"sample_id": str})
    feat = features.copy()
    feat["sample_id"] = feat["sample_id"].astype(str)
    feat = feat.set_index("sample_id").reindex(pred["sample_id"])
    entropy_cols = [c for c in feat.columns if c.startswith("jev_") and c.endswith("__entropy") and "task_family" not in c]
    entropy = feat[entropy_cols].mean(axis=1).to_numpy(float)
    margin_uncertainty = -pred["prediction_margin"].to_numpy(float)
    regret = pred["regret"].to_numpy(float)
    sem_curve = selective_risk(regret, entropy)
    margin_curve = selective_risk(regret, margin_uncertainty)
    output_root.mkdir(parents=True, exist_ok=True)
    sem_curve.to_csv(output_root / "semantic_entropy_curve.csv", index=False)
    margin_curve.to_csv(output_root / "router_margin_curve.csv", index=False)
    payload = {
        "semantic_entropy_AURC": aurc(sem_curve),
        "router_margin_AURC": aurc(margin_curve),
        "n_queries": int(len(pred)),
    }
    (output_root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def reproduce_calibrated_selective(
    samples: pd.DataFrame,
    outcomes: pd.DataFrame,
    features: pd.DataFrame,
    output_root: Path,
) -> dict[str, object]:
    rows = []
    for seed in SEEDS:
        result = calibrated_domain_ood(
            samples, outcomes, features, output_root / str(seed),
            calibration_ratio=0.2, min_coverage=0.5, threshold_grid_size=101,
            seed=seed, feature_set="semantic_manual",
        )
        rows.append({"seed": seed, **result})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "seed_summary.csv", index=False)
    payload = {
        "n_seeds": len(rows),
        "test_score_mean_pct": float(frame["macro_test_score"].mean() * 100),
        "test_score_std_pct": float(frame["macro_test_score"].std(ddof=1) * 100),
        "router_only_mean_pct": float(frame["macro_router_only_score"].mean() * 100),
        "router_only_std_pct": float(frame["macro_router_only_score"].std(ddof=1) * 100),
        "coverage_mean_pct": float(frame["macro_test_coverage"].mean() * 100),
        "coverage_std_pct": float(frame["macro_test_coverage"].std(ddof=1) * 100),
    }
    (output_root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def reproduce_cost(root: Path, output_root: Path) -> pd.DataFrame:
    p = default_paths(root)
    samples = pd.read_csv(p["cost_samples"], dtype={"sample_id": str})
    outcomes = pd.read_csv(p["cost_outcomes"], dtype={"sample_id": str})
    features = pd.read_csv(p["cost_features"], dtype={"sample_id": str})
    rows = []
    for seed in COST_SEEDS:
        result = paper_cost_evaluate_validated(
            samples, outcomes, features, output_root / str(seed),
            seed=seed, train_ratio=0.7, validation_ratio=0.2,
            representation="probability_mass", allow_missing=True, baseline_model="gpt-5",
        )
        rows.append(result)
    frame = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "summary.csv", index=False)
    payload = {
        "PerfGain_pct_mean": float(frame["PerfGain_pct"].mean()),
        "PerfGain_pct_std": float(frame["PerfGain_pct"].std(ddof=1)),
        "positive_PerfGain_seeds": int((frame["PerfGain_pct"] > 0).sum()),
        "strict_CostSave_reported_seeds": int(frame["CostSave_pct"].notna().sum()),
        "positive_strict_CostSave_seeds": int((frame["CostSave_pct"].dropna() > 0).sum()),
    }
    (output_root / "aggregate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return frame


def reproduce_all(root: str | Path = ".", output_dir: str | Path = "artifacts/reproduction") -> dict[str, object]:
    root = Path(root).resolve()
    out = (root / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    validation = validate_release(root, require_models=True)
    (out / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")

    p, samples, outcomes, jev, laya = _load(root)

    grouped_root = out / "performance_grouped"
    grouped = rerun_main_grouped(
        samples, outcomes, jev,
        output_root=grouped_root,
        gte_embeddings=p["gte_embeddings"],
        seeds=SEEDS,
        train_ratio=0.7,
        laya_features=laya,
        run_nested_lite=True,
    )

    stats_dir = grouped_root / "paper/stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    pm_pred = grouped_root / "paper/cv_grouped/probability_mass/oof_predictions.csv"
    hard_pred = grouped_root / "paper/cv_grouped/hard/oof_predictions.csv"
    laya_pred = grouped_root / "paper/laya_grouped/cv/oof_predictions.csv"
    pm_hard = paired_prediction_test(samples, pd.read_csv(pm_pred), pd.read_csv(hard_pred), n_bootstrap=10_000, n_permutations=20_000, seed=3407, group_duplicates=True)
    laya_jev = paired_prediction_test(samples, pd.read_csv(laya_pred), pd.read_csv(pm_pred), n_bootstrap=10_000, n_permutations=20_000, seed=3407, group_duplicates=True)
    (stats_dir / "probability_mass_vs_hard.json").write_text(json.dumps(pm_hard, indent=2), encoding="utf-8")
    (stats_dir / "laya_vs_jev.json").write_text(json.dumps(laya_jev, indent=2), encoding="utf-8")

    # probe = leave_one_probe_out(
    #     samples, outcomes, jev, grouped_root / "paper/leave_one_probe_out",
    #     seeds=SEEDS, router="catboost", split="id-grouped",
    # )

    # ood = reproduce_ood(samples, outcomes, jev, p["gte_embeddings"], out / "ood")
    selective = reproduce_selective(samples, jev, pm_pred, out / "selective")
    calibrated = reproduce_calibrated_selective(samples, outcomes, jev, out / "calibrated_domain_ood")
    # cost = reproduce_cost(root, out / "performance_cost")

    # JEV-Direct is a frozen semantic-backend experiment. Recompute aggregate metrics
    # from its published OOF predictions, but never issue new JEV requests.
    direct_pred = pd.read_csv(p["jev_direct_predictions"], dtype={"sample_id": str})
    direct_metrics = benchmark_metrics(samples, outcomes, direct_pred)
    frozen_meta = json.loads(p["jev_direct_metrics"].read_text(encoding="utf-8"))
    direct_payload = {**direct_metrics, "frozen_original_protocol": True, "published_metadata": frozen_meta}
    direct_dir = out / "frozen_jev_direct"
    direct_dir.mkdir(parents=True, exist_ok=True)
    (direct_dir / "metrics.json").write_text(json.dumps(direct_payload, indent=2), encoding="utf-8")

    # Token and observed API-latency summaries come from frozen feature records.
    profiles_dir = out / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    full_profile = profile_jev_features(jev, profiles_dir / "jev_full.json", input_price_per_million=0.042)
    compact12 = pd.read_csv(p["compact12_features"], dtype={"sample_id": str})
    lite_profile = profile_jev_features(compact12, profiles_dir / "jev_compact12.json", input_price_per_million=0.042)
    laya_latency = laya.get("meta_laya_latency_ms", pd.Series(dtype=float)).dropna().to_numpy(float)
    laya_profile = {
        "n": int(len(laya_latency)),
        "latency_ms_mean": float(np.mean(laya_latency)) if len(laya_latency) else None,
        "latency_ms_p50": float(np.quantile(laya_latency, 0.50)) if len(laya_latency) else None,
        "latency_ms_p95": float(np.quantile(laya_latency, 0.95)) if len(laya_latency) else None,
        "note": "Frozen local-Laya measurements; hardware dependent.",
    }
    (profiles_dir / "laya.json").write_text(json.dumps(laya_profile, indent=2), encoding="utf-8")

    summary = {
        "validation": validation,
        "main": grouped,
        "paired_pm_vs_hard": pm_hard,
        "paired_laya_vs_jev": laya_jev,
        "probe_ablation_path": str(grouped_root / "paper/leave_one_probe_out/leave_one_probe_out_summary.csv"),
        "ood_path": str(out / "ood/summary.csv"),
        "selective": selective,
        "calibrated_selective": calibrated,
        "cost_path": str(out / "performance_cost/summary.csv"),
        "jev_direct": direct_payload,
        "jev_profile": full_profile,
        "compact12_profile": lite_profile,
        "laya_profile": laya_profile,
        "note": "No JEV or Laya inference is performed by reproduce_all().",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
