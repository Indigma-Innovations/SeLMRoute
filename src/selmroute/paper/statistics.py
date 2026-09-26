import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_group_text(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def _leakage_group_ids(samples: pd.DataFrame) -> pd.Series:
    if "query" not in samples.columns:
        raise ValueError("group-aware paired inference requires a 'query' column in samples")
    dataset = samples["dataset"].astype(str)
    query = samples["query"].map(_normalize_group_text)
    return pd.Series(
        [hashlib.sha256(f"{d}\0{q}".encode()).hexdigest() for d, q in zip(dataset, query, strict=True)],
        index=samples.index,
        name="leakage_group",
    )


def _paired_frame(
    samples: pd.DataFrame,
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    group_duplicates: bool = False,
) -> pd.DataFrame:
    s = samples.copy()
    s["sample_id"] = s["sample_id"].astype(str)
    meta_cols = ["sample_id", "dataset"]
    if group_duplicates:
        s["leakage_group"] = _leakage_group_ids(s)
        meta_cols.append("leakage_group")
    aa = a[["sample_id", "selected_score"]].copy().rename(columns={"selected_score": "score_a"})
    bb = b[["sample_id", "selected_score"]].copy().rename(columns={"selected_score": "score_b"})
    aa["sample_id"] = aa["sample_id"].astype(str)
    bb["sample_id"] = bb["sample_id"].astype(str)
    merged = aa.merge(bb, on="sample_id", how="inner", validate="one_to_one")
    if len(merged) != len(aa) or len(merged) != len(bb):
        raise ValueError("Prediction files do not contain the same sample ids")
    merged = merged.merge(s[meta_cols], on="sample_id", how="left", validate="one_to_one")
    if merged["dataset"].isna().any():
        raise ValueError("Missing dataset metadata for prediction rows")
    if group_duplicates and merged["leakage_group"].isna().any():
        raise ValueError("Missing leakage-group metadata for prediction rows")
    merged["diff"] = merged["score_a"].astype(float) - merged["score_b"].astype(float)
    return merged


def _macro_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("dataset")["sample_id"].transform("count").to_numpy(float)
    n_datasets = frame["dataset"].nunique()
    return 1.0 / (float(n_datasets) * counts)


def paired_prediction_test(
    samples: pd.DataFrame,
    predictions_a: pd.DataFrame,
    predictions_b: pd.DataFrame,
    *,
    n_bootstrap: int = 5000,
    n_permutations: int = 10000,
    seed: int = 3407,
    group_duplicates: bool = False,
) -> dict[str, object]:
    """Paired macro-AvgAcc test with optional duplicate-input clustering.

    When ``group_duplicates`` is true, bootstrap resampling and sign-flip
    permutation operate on whole ``(dataset, normalized query)`` clusters. This
    preserves the sample-level macro estimand while avoiding an independence
    assumption across repeated router inputs.
    """
    frame = _paired_frame(samples, predictions_a, predictions_b, group_duplicates=group_duplicates)
    diff = frame["diff"].to_numpy(float)
    weights = _macro_weights(frame)
    observed_macro = float(np.dot(weights, diff))
    observed_micro = float(diff.mean())
    rng = np.random.default_rng(seed)

    boot = np.empty(n_bootstrap, dtype=float)
    if not group_duplicates:
        groups = [g["diff"].to_numpy(float) for _, g in frame.groupby("dataset", sort=True)]
        for i in range(n_bootstrap):
            dataset_means = []
            for arr in groups:
                idx = rng.integers(0, len(arr), size=len(arr))
                dataset_means.append(float(arr[idx].mean()))
            boot[i] = float(np.mean(dataset_means))
        perm_contrib = diff * weights
        n_perm_units = len(diff)
        n_clusters = len(diff)
    else:
        # Cluster bootstrap within each dataset. Resampled clusters retain all
        # member rows, so the target remains the per-sample mean within dataset.
        dataset_clusters: list[list[np.ndarray]] = []
        for _, dg in frame.groupby("dataset", sort=True):
            dataset_clusters.append([g["diff"].to_numpy(float) for _, g in dg.groupby("leakage_group", sort=True)])
        for i in range(n_bootstrap):
            dataset_means = []
            for clusters in dataset_clusters:
                idx = rng.integers(0, len(clusters), size=len(clusters))
                sampled = np.concatenate([clusters[int(j)] for j in idx])
                dataset_means.append(float(sampled.mean()))
            boot[i] = float(np.mean(dataset_means))
        # Sign-flip a whole repeated-input cluster together. Summing the original
        # sample macro weights inside a cluster preserves the observed estimand.
        weighted = frame.assign(_weighted=diff * weights)
        perm_contrib = (
            weighted.groupby(["dataset", "leakage_group"], sort=True)["_weighted"]
            .sum()
            .to_numpy(float)
        )
        n_perm_units = len(perm_contrib)
        n_clusters = n_perm_units

    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])

    extreme = 0
    remaining = n_permutations
    batch_size = 256
    abs_obs = abs(observed_macro)
    while remaining > 0:
        b = min(batch_size, remaining)
        signs = rng.integers(0, 2, size=(b, n_perm_units), dtype=np.int8) * 2 - 1
        stats = signs @ perm_contrib
        extreme += int(np.sum(np.abs(stats) >= abs_obs - 1e-15))
        remaining -= b
    p_value = (extreme + 1.0) / (n_permutations + 1.0)

    wins = int(np.sum(diff > 0))
    losses = int(np.sum(diff < 0))
    ties = int(np.sum(diff == 0))
    return {
        "macro_diff": observed_macro,
        "macro_diff_pp": observed_macro * 100.0,
        "micro_diff": observed_micro,
        "micro_diff_pp": observed_micro * 100.0,
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "bootstrap_ci_low_pp": float(ci_low * 100.0),
        "bootstrap_ci_high_pp": float(ci_high * 100.0),
        "permutation_p_two_sided": float(p_value),
        "n_bootstrap": int(n_bootstrap),
        "n_permutations": int(n_permutations),
        "n_queries": int(len(frame)),
        "n_datasets": int(frame["dataset"].nunique()),
        "n_inference_clusters": int(n_clusters),
        "cluster_duplicate_inputs": bool(group_duplicates),
        "query_wins": wins,
        "query_losses": losses,
        "query_ties": ties,
    }


def paired_seed_suite(
    samples: pd.DataFrame,
    *,
    a_template: str,
    b_template: str,
    seeds: Iterable[int],
    output_dir: str | Path,
    label_a: str = "A",
    label_b: str = "B",
    n_bootstrap: int = 5000,
    n_permutations: int = 10000,
    seed: int = 3407,
) -> tuple[pd.DataFrame, dict[str, object]]:
    seed_values = [int(x) for x in seeds]
    if len(seed_values) > 1 and ("{seed}" not in a_template or "{seed}" not in b_template):
        raise ValueError("Both templates must contain {seed} when evaluating multiple seeds")
    rows: list[dict[str, object]] = []
    for pos, split_seed in enumerate(seed_values):
        pa = Path(a_template.format(seed=int(split_seed)))
        pb = Path(b_template.format(seed=int(split_seed)))
        if not pa.exists():
            raise FileNotFoundError(pa)
        if not pb.exists():
            raise FileNotFoundError(pb)
        result = paired_prediction_test(
            samples,
            pd.read_csv(pa),
            pd.read_csv(pb),
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            seed=seed + pos * 1009,
        )
        rows.append({"seed": int(split_seed), "A": label_a, "B": label_b, **result})
    per_seed = pd.DataFrame(rows)
    summary = {
        "A": label_a,
        "B": label_b,
        "n_seeds": int(len(per_seed)),
        "macro_diff_pp_mean": float(per_seed["macro_diff_pp"].mean()),
        "macro_diff_pp_std": float(per_seed["macro_diff_pp"].std(ddof=1)) if len(per_seed) > 1 else 0.0,
        "micro_diff_pp_mean": float(per_seed["micro_diff_pp"].mean()),
        "micro_diff_pp_std": float(per_seed["micro_diff_pp"].std(ddof=1)) if len(per_seed) > 1 else 0.0,
        "positive_seeds": int((per_seed["macro_diff"] > 0).sum()),
        "negative_seeds": int((per_seed["macro_diff"] < 0).sum()),
        "note": "Per-seed randomization p-values are reported separately; overlapping random splits are not treated as independent for a combined p-value.",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / "paired_tests_by_seed.csv", index=False)
    (out / "paired_tests_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return per_seed, summary


def paired_equivalence_test(
    samples: pd.DataFrame,
    predictions_a: pd.DataFrame,
    predictions_b: pd.DataFrame,
    *,
    margin_pp: float = 0.5,
    n_bootstrap: int = 10000,
    seed: int = 3407,
    group_duplicates: bool = False,
) -> dict[str, object]:
    """Bootstrap-based non-inferiority/equivalence diagnostic on macro AvgAcc difference.

    A-B is measured in percentage points. Equivalence is declared only when the
    two-sided 95% bootstrap interval is wholly inside [-margin, +margin].
    Non-inferiority of A to B requires the lower CI bound to exceed -margin.
    """
    result = paired_prediction_test(samples, predictions_a, predictions_b, n_bootstrap=n_bootstrap, n_permutations=100, seed=seed, group_duplicates=group_duplicates)
    lo=float(result["bootstrap_ci_low_pp"])
    hi=float(result["bootstrap_ci_high_pp"])
    m=float(margin_pp)
    return {
        "macro_diff_pp": float(result["macro_diff_pp"]),
        "bootstrap_ci_low_pp": lo,
        "bootstrap_ci_high_pp": hi,
        "margin_pp": m,
        "noninferior_A_to_B": bool(lo > -m),
        "equivalent_within_margin": bool(lo > -m and hi < m),
        "n_queries": int(result["n_queries"]),
        "n_datasets": int(result["n_datasets"]),
    }


def paired_noninferiority_test(
    samples: pd.DataFrame,
    predictions_a: pd.DataFrame,
    predictions_b: pd.DataFrame,
    *,
    margin_pp: float = 0.5,
    n_bootstrap: int = 10000,
    seed: int = 3407,
    confidence: float = 0.95,
    group_duplicates: bool = False,
) -> dict[str, object]:
    """Dataset-aware bootstrap non-inferiority diagnostic for macro AvgAcc.

    The estimand is A-B in percentage points. A is declared non-inferior to B
    when the one-sided lower confidence bound exceeds ``-margin_pp``.
    """
    if margin_pp <= 0:
        raise ValueError("margin_pp must be > 0")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    frame = _paired_frame(samples, predictions_a, predictions_b, group_duplicates=group_duplicates)
    observed_pp = float(frame.groupby("dataset", sort=True)["diff"].mean().mean() * 100.0)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=float)
    if not group_duplicates:
        groups = [g["diff"].to_numpy(float) for _, g in frame.groupby("dataset", sort=True)]
        for i in range(n_bootstrap):
            means = []
            for arr in groups:
                idx = rng.integers(0, len(arr), size=len(arr))
                means.append(float(arr[idx].mean()))
            boot[i] = float(np.mean(means) * 100.0)
        n_clusters = len(frame)
    else:
        dataset_clusters = [
            [g["diff"].to_numpy(float) for _, g in dg.groupby("leakage_group", sort=True)]
            for _, dg in frame.groupby("dataset", sort=True)
        ]
        for i in range(n_bootstrap):
            means = []
            for clusters in dataset_clusters:
                idx = rng.integers(0, len(clusters), size=len(clusters))
                sampled = np.concatenate([clusters[int(j)] for j in idx])
                means.append(float(sampled.mean()))
            boot[i] = float(np.mean(means) * 100.0)
        n_clusters = int(frame[["dataset", "leakage_group"]].drop_duplicates().shape[0])
    alpha = 1.0 - float(confidence)
    lower = float(np.quantile(boot, alpha))
    two_lo, two_hi = [float(x) for x in np.quantile(boot, [0.025, 0.975])]
    m = float(margin_pp)
    return {
        "macro_diff_pp": observed_pp,
        "margin_pp": m,
        "confidence": float(confidence),
        "lower_95_one_sided_pp": lower,
        "bootstrap_ci_low_pp": two_lo,
        "bootstrap_ci_high_pp": two_hi,
        "noninferior_A_to_B": bool(lower > -m),
        "n_bootstrap": int(n_bootstrap),
        "n_queries": int(len(frame)),
        "n_datasets": int(frame["dataset"].nunique()),
        "n_inference_clusters": int(n_clusters),
        "cluster_duplicate_inputs": bool(group_duplicates),
        "decision_rule": "non-inferior iff one-sided lower confidence bound > -margin_pp",
    }
