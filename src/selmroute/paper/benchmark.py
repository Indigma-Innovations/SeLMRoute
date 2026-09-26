import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_relative(numerator: pd.Series, denominator: pd.Series, *, one_minus: bool = False) -> pd.Series:
    denom = denominator.astype(float)
    valid = denom.abs() > 1e-12
    out = pd.Series(np.nan, index=denominator.index, dtype=float)
    ratio = numerator.astype(float)[valid] / denom[valid]
    out.loc[valid] = (1.0 - ratio) if one_minus else (ratio - 1.0)
    return out


def benchmark_metrics(
    samples: pd.DataFrame,
    outcomes: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, object]:
    """Compute LLMRouterBench performance-oriented headline metrics.

    Definitions follow the benchmark paper: AvgAcc is the macro mean over datasets;
    Gain@R and Gain@B are mean per-dataset relative gains over Random and Best Single;
    Gap@O is the mean per-dataset relative gap to the instance Oracle.

    Best Single is selected in hindsight by highest *macro dataset accuracy* on the
    evaluated test queries, matching the benchmark baseline definition.
    """
    s = samples.copy()
    o = outcomes.copy()
    p = predictions.copy()
    for frame in (s, o, p):
        frame["sample_id"] = frame["sample_id"].astype(str)

    if p["sample_id"].duplicated().any():
        raise ValueError("predictions contains duplicate sample_id rows")
    if "selected_score" not in p.columns:
        raise ValueError("predictions must contain selected_score")

    meta = s.set_index("sample_id")[["dataset"]]
    ids = p["sample_id"].tolist()
    missing = [sid for sid in ids if sid not in meta.index]
    if missing:
        raise ValueError(f"samples is missing {len(missing)} prediction sample ids")

    routed = p.set_index("sample_id").join(meta, how="left")
    router_acc = routed.groupby("dataset", sort=True)["selected_score"].mean()

    test_o = o[o["sample_id"].isin(ids)].copy()
    expected = len(ids) * test_o["model"].nunique()
    if len(test_o) != expected:
        raise ValueError("outcomes does not contain a complete sample/model matrix for predictions")
    if "dataset" not in test_o.columns:
        test_o = test_o.join(meta, on="sample_id", how="left")
    else:
        test_o["dataset"] = test_o["dataset"].astype(str)

    dataset_model = (
        test_o.groupby(["dataset", "model"], sort=True)["score"]
        .mean()
        .unstack("model")
        .reindex(router_acc.index)
    )
    random_acc = dataset_model.mean(axis=1)
    model_macro = dataset_model.mean(axis=0)
    best_model = str(model_macro.idxmax())
    best_acc = dataset_model[best_model]
    dataset_oracle_acc = dataset_model.max(axis=1)

    per_query_oracle = test_o.groupby("sample_id", sort=False)["score"].max()
    oracle_frame = meta.reindex(per_query_oracle.index).copy()
    oracle_frame["oracle"] = per_query_oracle
    oracle_acc = oracle_frame.groupby("dataset", sort=True)["oracle"].mean().reindex(router_acc.index)

    gain_r = _safe_relative(router_acc, random_acc)
    gain_b = _safe_relative(router_acc, best_acc)
    gap_o = _safe_relative(router_acc, oracle_acc, one_minus=True)

    return {
        "AvgAcc": float(router_acc.mean() * 100.0),
        "Gain@R": float(gain_r.mean(skipna=True) * 100.0),
        "Gain@B": float(gain_b.mean(skipna=True) * 100.0),
        "Gap@O": float(gap_o.mean(skipna=True) * 100.0),
        "BestSingleAvg": float(best_acc.mean() * 100.0),
        "DatasetOracleAvg": float(dataset_oracle_acc.mean() * 100.0),
        "OracleAvg": float(oracle_acc.mean() * 100.0),
        "best_single_model": best_model,
        "n_datasets": int(len(router_acc)),
        "n_test": int(len(ids)),
    }


def _parse_system_specs(specs: Iterable[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"System spec must be NAME=PATH_TEMPLATE: {spec}")
        name, template = spec.split("=", 1)
        name = name.strip()
        template = template.strip()
        if not name or not template:
            raise ValueError(f"Invalid system spec: {spec}")
        parsed.append((name, template))
    if not parsed:
        raise ValueError("At least one system spec is required")
    return parsed


def benchmark_seed_suite(
    samples: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    system_specs: Iterable[str],
    seeds: Iterable[int],
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    systems = _parse_system_specs(system_specs)
    seed_values = [int(x) for x in seeds]
    if len(seed_values) > 1:
        missing_template = [template for _, template in systems if "{seed}" not in template]
        if missing_template:
            raise ValueError("System path templates must contain {seed} when evaluating multiple seeds")
    rows: list[dict[str, object]] = []
    for seed in seed_values:
        for system, template in systems:
            path = Path(template.format(seed=int(seed)))
            if not path.exists():
                raise FileNotFoundError(path)
            metrics = benchmark_metrics(samples, outcomes, pd.read_csv(path))
            rows.append({"seed": int(seed), "system": system, **metrics})

    all_runs = pd.DataFrame(rows)
    numeric = ["AvgAcc", "Gain@R", "Gain@B", "Gap@O", "BestSingleAvg", "DatasetOracleAvg", "OracleAvg"]
    summary_rows: list[dict[str, object]] = []
    for system, group in all_runs.groupby("system", sort=False):
        row: dict[str, object] = {"system": system, "n_seeds": int(len(group))}
        for col in numeric:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        best_names = group["best_single_model"].dropna().astype(str)
        row["best_single_model"] = best_names.mode().iloc[0] if not best_names.empty else ""
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("AvgAcc_mean", ascending=False)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_runs.to_csv(out / "benchmark_metrics_all_seeds.csv", index=False)
    summary.to_csv(out / "benchmark_metrics_seed_summary.csv", index=False)
    if len(seed_values) == 5:
        summary.to_csv(out / "benchmark_metrics_5seed_summary.csv", index=False)
    (out / "benchmark_metadata.json").write_text(
        json.dumps({"seeds": seed_values, "systems": [x[0] for x in systems]}, indent=2),
        encoding="utf-8",
    )
    return all_runs, summary
