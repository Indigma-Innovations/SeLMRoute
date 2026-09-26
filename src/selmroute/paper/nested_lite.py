import json
from pathlib import Path

import pandas as pd

from selmroute.evaluation.metrics import outcome_matrix, selected_frame
from selmroute.paper.ablation import build_semantic_ablation_features
from selmroute.paper.benchmark import benchmark_metrics
from selmroute.paper.crossval import _router, dataset_stratified_folds
from selmroute.paper.statistics import paired_noninferiority_test, paired_prediction_test
from selmroute.routing.policy import select_quality


def _probe_name(col: str) -> str:
    return col.split("__", 1)[0].removeprefix("jev_")


def _macro_avgacc(samples: pd.DataFrame, predictions: pd.DataFrame) -> float:
    s = samples[["sample_id", "dataset"]].copy()
    s["sample_id"] = s["sample_id"].astype(str)
    p = predictions[["sample_id", "selected_score"]].copy()
    p["sample_id"] = p["sample_id"].astype(str)
    merged = p.merge(s, on="sample_id", how="left", validate="one_to_one")
    if merged["dataset"].isna().any():
        raise ValueError("Missing dataset metadata while scoring nested probe selection")
    return float(merged.groupby("dataset", sort=True)["selected_score"].mean().mean() * 100.0)


def _inner_oof_avgacc(
    samples: pd.DataFrame,
    outcomes: pd.DataFrame,
    features: pd.DataFrame,
    *,
    n_folds: int,
    seed: int,
    router: str,
    group_duplicates: bool = False,
) -> float:
    s = samples.copy()
    o = outcomes.copy()
    x = features.copy()
    for frame in (s, o, x):
        frame["sample_id"] = frame["sample_id"].astype(str)

    folds = dataset_stratified_folds(s, n_folds=n_folds, seed=seed, group_duplicates=group_duplicates)
    models = sorted(o["model"].astype(str).unique())
    xidx = x.set_index("sample_id")
    preds: list[pd.DataFrame] = []

    for fold in range(n_folds):
        test_ids = folds.loc[folds.fold == fold, "sample_id"].tolist()
        train_ids = folds.loc[folds.fold != fold, "sample_id"].tolist()
        train_x = xidx.reindex(train_ids).reset_index()
        test_x = xidx.reindex(test_ids).reset_index()
        train_y = outcome_matrix(o, train_ids, models)
        test_y = outcome_matrix(o, test_ids, models)
        fitted = _router(router, seed + fold * 1009).fit(train_x, train_y)
        pred = fitted.predict(test_x)
        selected = select_quality(pred)
        preds.append(selected_frame(test_ids, models, pred, selected, test_y))

    combined = pd.concat(preds, ignore_index=True)
    if len(combined) != len(s) or combined["sample_id"].duplicated().any():
        raise ValueError("Inner OOF predictions must contain every outer-training sample exactly once")
    return _macro_avgacc(s, combined)


def nested_lite_oof(
    samples: pd.DataFrame,
    outcomes: pd.DataFrame,
    features: pd.DataFrame,
    output_dir: str | Path,
    *,
    keep_probes: int = 12,
    outer_folds: int = 5,
    inner_folds: int = 3,
    seed: int = 3407,
    router: str = "catboost",
    margin_pp: float = 0.5,
    n_bootstrap: int = 10000,
    group_duplicates: bool = False,
) -> dict[str, object]:
    """Nested confirmation of a compact semantic probe set.

    Probe selection happens exclusively inside each outer-training partition. For each
    outer fold, leave-one-probe-out inner OOF performance estimates the contribution of
    each probe. The top ``keep_probes`` probes are retained, then both the compact and
    full ProbabilityMass routers are trained on the complete outer-training partition
    and evaluated on the untouched outer-test partition.
    """
    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("outer_folds and inner_folds must both be >= 2")

    s = samples.copy()
    o = outcomes.copy()
    f = features.copy()
    for frame in (s, o, f):
        frame["sample_id"] = frame["sample_id"].astype(str)

    base = build_semantic_ablation_features(f, "probability_mass")
    probes = list(dict.fromkeys(_probe_name(c) for c in base.columns if c != "sample_id"))
    if keep_probes < 1 or keep_probes >= len(probes):
        raise ValueError(f"keep_probes must be between 1 and {len(probes)-1}; got {keep_probes}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    outer = dataset_stratified_folds(s, n_folds=outer_folds, seed=seed, group_duplicates=group_duplicates)
    outer.to_csv(out / "outer_folds.csv", index=False)

    models = sorted(o["model"].astype(str).unique())
    xidx = base.set_index("sample_id")
    full_preds: list[pd.DataFrame] = []
    lite_preds: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for outer_fold in range(outer_folds):
        outer_test_ids = outer.loc[outer.fold == outer_fold, "sample_id"].tolist()
        outer_train_ids = outer.loc[outer.fold != outer_fold, "sample_id"].tolist()
        train_samples = s[s["sample_id"].isin(outer_train_ids)].copy()
        train_outcomes = o[o["sample_id"].isin(outer_train_ids)].copy()
        train_base = xidx.reindex(outer_train_ids).reset_index()

        inner_seed = seed + (outer_fold + 1) * 100_003
        baseline_inner = _inner_oof_avgacc(
            train_samples,
            train_outcomes,
            train_base,
            n_folds=inner_folds,
            seed=inner_seed,
            router=router,
            group_duplicates=group_duplicates,
        )

        contribution: dict[str, float] = {}
        dropped_scores: dict[str, float] = {}
        for probe in probes:
            cols = [c for c in train_base.columns if c == "sample_id" or _probe_name(c) != probe]
            score = _inner_oof_avgacc(
                train_samples,
                train_outcomes,
                train_base[cols],
                n_folds=inner_folds,
                seed=inner_seed,
                router=router,
                group_duplicates=group_duplicates,
            )
            dropped_scores[probe] = score
            contribution[probe] = baseline_inner - score

        probe_pos = {probe: i for i, probe in enumerate(probes)}
        ranked = sorted(probes, key=lambda p: (-contribution[p], probe_pos[p]))
        selected_probes = ranked[:keep_probes]
        selected_set = set(selected_probes)

        for rank, probe in enumerate(ranked, start=1):
            selection_rows.append(
                {
                    "outer_fold": outer_fold,
                    "probe": probe,
                    "rank": rank,
                    "selected": probe in selected_set,
                    "inner_all_avgacc": baseline_inner,
                    "inner_drop_avgacc": dropped_scores[probe],
                    "contribution_pp": contribution[probe],
                }
            )

        train_x_full = xidx.reindex(outer_train_ids).reset_index()
        test_x_full = xidx.reindex(outer_test_ids).reset_index()
        lite_cols = [
            c for c in train_x_full.columns
            if c == "sample_id" or _probe_name(c) in selected_set
        ]
        train_x_lite = train_x_full[lite_cols]
        test_x_lite = test_x_full[lite_cols]
        train_y = outcome_matrix(o, outer_train_ids, models)
        test_y = outcome_matrix(o, outer_test_ids, models)
        fit_seed = seed + outer_fold * 1009

        full_model = _router(router, fit_seed).fit(train_x_full, train_y)
        full_raw = full_model.predict(test_x_full)
        full_selected = select_quality(full_raw)
        full_pf = selected_frame(outer_test_ids, models, full_raw, full_selected, test_y)
        full_pf["fold"] = outer_fold
        full_preds.append(full_pf)

        lite_model = _router(router, fit_seed).fit(train_x_lite, train_y)
        lite_raw = lite_model.predict(test_x_lite)
        lite_selected = select_quality(lite_raw)
        lite_pf = selected_frame(outer_test_ids, models, lite_raw, lite_selected, test_y)
        lite_pf["fold"] = outer_fold
        lite_preds.append(lite_pf)

        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "n_train": len(outer_train_ids),
                "n_test": len(outer_test_ids),
                "inner_all_avgacc": baseline_inner,
                "selected_probes": ",".join(selected_probes),
                "selected_probe_count": len(selected_probes),
                "selected_feature_count": len(lite_cols) - 1,
                "full_feature_count": len(base.columns) - 1,
            }
        )

    full_oof = pd.concat(full_preds, ignore_index=True)
    lite_oof = pd.concat(lite_preds, ignore_index=True)
    for name, frame in (("full", full_oof), ("lite", lite_oof)):
        if len(frame) != len(s) or frame["sample_id"].duplicated().any():
            raise ValueError(f"Nested {name} OOF predictions must contain each sample exactly once")

    full_oof.to_csv(out / "full16_oof_predictions.csv", index=False)
    lite_oof.to_csv(out / "nested_lite_oof_predictions.csv", index=False)

    full_metrics = benchmark_metrics(s, o, full_oof)
    lite_metrics = benchmark_metrics(s, o, lite_oof)
    paired = paired_prediction_test(
        s, lite_oof, full_oof,
        n_bootstrap=n_bootstrap,
        n_permutations=max(1000, min(20000, n_bootstrap * 2)),
        seed=seed,
        group_duplicates=group_duplicates,
    )
    noninferiority = paired_noninferiority_test(
        s, lite_oof, full_oof,
        margin_pp=margin_pp,
        n_bootstrap=n_bootstrap,
        seed=seed,
        group_duplicates=group_duplicates,
    )

    selection = pd.DataFrame(selection_rows)
    selection.to_csv(out / "probe_selection_by_fold.csv", index=False)
    fold_summary = pd.DataFrame(fold_rows)
    fold_summary.to_csv(out / "nested_fold_summary.csv", index=False)

    freq = (
        selection.groupby("probe", sort=False)
        .agg(
            selected_folds=("selected", "sum"),
            mean_rank=("rank", "mean"),
            mean_contribution_pp=("contribution_pp", "mean"),
            std_contribution_pp=("contribution_pp", "std"),
        )
        .reset_index()
    )
    freq["selection_frequency"] = freq["selected_folds"] / float(outer_folds)
    freq = freq.sort_values(
        ["selected_folds", "mean_contribution_pp", "mean_rank"],
        ascending=[False, False, True],
    )
    freq.to_csv(out / "probe_selection_frequency.csv", index=False)

    payload: dict[str, object] = {
        "method": "nested_leave_one_probe_out_top_k",
        "router": router,
        "representation": "probability_mass",
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "seed": seed,
        "keep_probes": keep_probes,
        "available_probes": len(probes),
        "full_feature_count": len(base.columns) - 1,
        "lite_feature_count_mean": float(fold_summary["selected_feature_count"].mean()),
        "full16_AvgAcc": float(full_metrics["AvgAcc"]),
        "nested_lite_AvgAcc": float(lite_metrics["AvgAcc"]),
        "nested_lite_minus_full_pp": float(lite_metrics["AvgAcc"] - full_metrics["AvgAcc"]),
        "paired_bootstrap_ci_low_pp": float(paired["bootstrap_ci_low_pp"]),
        "paired_bootstrap_ci_high_pp": float(paired["bootstrap_ci_high_pp"]),
        "paired_permutation_p_two_sided": float(paired["permutation_p_two_sided"]),
        "noninferiority_margin_pp": float(margin_pp),
        "noninferiority_lower_95_one_sided_pp": float(noninferiority["lower_95_one_sided_pp"]),
        "noninferior_to_full16": bool(noninferiority["noninferior_A_to_B"]),
        "n_queries": int(len(s)),
        "n_datasets": int(s["dataset"].nunique()),
        "group_duplicates": group_duplicates,
    }
    (out / "nested_lite_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "full16_metrics.json").write_text(json.dumps(full_metrics, indent=2), encoding="utf-8")
    (out / "nested_lite_router_metrics.json").write_text(json.dumps(lite_metrics, indent=2), encoding="utf-8")
    (out / "paired_test.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    (out / "noninferiority_test.json").write_text(json.dumps(noninferiority, indent=2), encoding="utf-8")
    return payload
