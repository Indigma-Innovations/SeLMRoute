import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from selmroute.paper.benchmark import benchmark_metrics

DEFAULT_SYSTEM_PATTERNS: dict[str, tuple[str, ...]] = {
    "probability_mass": (
        "runs/paper/ablations/{seed}/probability_mass/predictions.csv",
        "paper/ablations/{seed}/probability_mass/predictions.csv",
        "runs/ablations/{seed}/probability_mass/predictions.csv",
        "ablations/{seed}/probability_mass/predictions.csv",
    ),
    "full": (
        "runs/paper/ablations/{seed}/full/predictions.csv",
        "paper/ablations/{seed}/full/predictions.csv",
        "runs/ablations/{seed}/full/predictions.csv",
        "ablations/{seed}/full/predictions.csv",
    ),
    "hard": (
        "runs/paper/ablations/{seed}/hard/predictions.csv",
        "paper/ablations/{seed}/hard/predictions.csv",
        "runs/ablations/{seed}/hard/predictions.csv",
        "ablations/{seed}/hard/predictions.csv",
    ),
    "domain_only": (
        "runs/seeds/{seed}/domain_only/predictions.csv",
        "runs/paper/seeds/{seed}/domain_only/predictions.csv",
        "runs/{seed}/domain_only/predictions.csv",
        "paper/seeds/{seed}/domain_only/predictions.csv",
    ),
    "tfidf": (
        "runs/seeds/{seed}/tfidf/predictions.csv",
        "runs/paper/seeds/{seed}/tfidf/predictions.csv",
        "runs/{seed}/tfidf/predictions.csv",
        "paper/seeds/{seed}/tfidf/predictions.csv",
    ),
    "gte_catboost": (
        "runs/paper/gte_catboost/{seed}/predictions.csv",
        "paper/gte_catboost/{seed}/predictions.csv",
        "runs/gte_catboost/{seed}/predictions.csv",
        "gte_catboost/{seed}/predictions.csv",
    ),
}


def _parse_overrides(specs: Iterable[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"--system must be NAME=PATH_TEMPLATE, got: {spec}")
        name, template = spec.split("=", 1)
        name = name.strip()
        template = template.strip()
        if not name or not template:
            raise ValueError(f"Invalid --system specification: {spec}")
        if "{seed}" not in template:
            raise ValueError(f"System template must contain {{seed}}: {spec}")
        out[name] = template
    return out


def _resolve_one(root: Path, system: str, seed: int, override: str | None) -> Path:
    if override is not None:
        p = Path(override.format(seed=seed))
        if not p.is_absolute():
            p = root / p
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    patterns = DEFAULT_SYSTEM_PATTERNS.get(system)
    if patterns is None:
        raise ValueError(f"No default artifact pattern for system={system!r}; pass --system {system}=...")
    candidates = [root / pattern.format(seed=seed) for pattern in patterns]
    found = [p for p in candidates if p.exists()]
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise ValueError(
            f"Ambiguous prediction artifacts for system={system}, seed={seed}: "
            f"{[str(p) for p in found]}. Pass an explicit --system override."
        )
    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find predictions for system={system}, seed={seed}. Tried:\n  {tried}\n"
        f"Use --system {system}=RELATIVE_OR_ABSOLUTE_TEMPLATE_WITH_{{seed}} to override."
    )


def _id_digest(ids: list[str]) -> str:
    payload = "\n".join(sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_complete_table1(
    samples: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    artifact_root: str | Path,
    output_dir: str | Path,
    seeds: Iterable[int] = (42, 999, 2024, 2025, 3407),
    systems: Iterable[str] = (
        "probability_mass", "full", "hard", "domain_only", "tfidf", "gte_catboost"
    ),
    system_specs: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a fully populated, apples-to-apples main benchmark table.

    The function refuses to compare systems unless, within each seed, every prediction
    file contains exactly the same test sample IDs. This prevents silent metric drift
    from mismatched train/test splits.
    """
    s = samples.copy()
    o = outcomes.copy()
    s["sample_id"] = s["sample_id"].astype(str)
    o["sample_id"] = o["sample_id"].astype(str)
    o["model"] = o["model"].astype(str)
    if s["sample_id"].duplicated().any():
        raise ValueError("samples contains duplicate sample_id rows")
    if o[["sample_id", "model"]].duplicated().any():
        raise ValueError("outcomes contains duplicate sample_id/model rows")

    root = Path(artifact_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_values = [int(x) for x in seeds]
    system_values = [str(x) for x in systems]
    overrides = _parse_overrides(system_specs)

    rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    baseline_rows_by_seed: list[dict[str, object]] = []

    for seed in seed_values:
        resolved: dict[str, Path] = {
            system: _resolve_one(root, system, seed, overrides.get(system))
            for system in system_values
        }
        predictions: dict[str, pd.DataFrame] = {}
        reference_ids: set[str] | None = None
        reference_system: str | None = None
        for system, path in resolved.items():
            p = pd.read_csv(path, dtype={"sample_id": str})
            required = {"sample_id", "selected_score"}
            missing = required - set(p.columns)
            if missing:
                raise ValueError(f"{path} missing required columns {sorted(missing)}")
            if p["sample_id"].duplicated().any():
                raise ValueError(f"{path} contains duplicate sample_id rows")
            if p["selected_score"].isna().any():
                raise ValueError(f"{path} contains missing selected_score values")
            ids = set(p["sample_id"].astype(str))
            if reference_ids is None:
                reference_ids = ids
                reference_system = system
            elif ids != reference_ids:
                only_ref = sorted(reference_ids - ids)[:10]
                only_cur = sorted(ids - reference_ids)[:10]
                raise ValueError(
                    f"Table-1 support mismatch for seed={seed}: {system} != {reference_system}. "
                    f"missing_from_{system}={only_ref}, extra_in_{system}={only_cur}"
                )
            predictions[system] = p
            support_rows.append({
                "seed": seed,
                "system": system,
                "prediction_path": str(path),
                "n_test": len(p),
                "sample_id_digest": _id_digest(p["sample_id"].astype(str).tolist()),
            })

        assert reference_ids is not None
        for system in system_values:
            metrics = benchmark_metrics(s, o, predictions[system])
            rows.append({"seed": seed, "system": system, **metrics})
        first = rows[-len(system_values)]
        baseline_rows_by_seed.extend([
            {"seed": seed, "baseline": "Best single", "AvgAcc": first["BestSingleAvg"]},
            {"seed": seed, "baseline": "Dataset oracle", "AvgAcc": first["DatasetOracleAvg"]},
            {"seed": seed, "baseline": "Instance oracle", "AvgAcc": first["OracleAvg"]},
        ])

    runs = pd.DataFrame(rows)
    numeric = ["AvgAcc", "Gain@R", "Gain@B", "Gap@O", "BestSingleAvg", "DatasetOracleAvg", "OracleAvg"]
    summary_rows: list[dict[str, object]] = []
    for system, group in runs.groupby("system", sort=False):
        row: dict[str, object] = {"system": system, "n_seeds": int(len(group))}
        for col in numeric:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
        best_names = group["best_single_model"].dropna().astype(str)
        row["best_single_model"] = best_names.mode().iloc[0] if not best_names.empty else ""
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("AvgAcc_mean", ascending=False).reset_index(drop=True)

    baseline_runs = pd.DataFrame(baseline_rows_by_seed)
    baselines = (
        baseline_runs.groupby("baseline", sort=False)["AvgAcc"]
        .agg([("AvgAcc_mean", "mean"), ("AvgAcc_std", "std")])
        .reset_index()
    )
    definitions = {
        "Best single": "One fixed candidate selected by highest macro Dataset-Avg on the evaluated benchmark support.",
        "Dataset oracle": "Best candidate selected separately for each dataset.",
        "Instance oracle": "Best observed candidate selected separately for each query.",
    }
    baselines["definition"] = baselines["baseline"].map(definitions)

    support = pd.DataFrame(support_rows)
    runs.to_csv(out / "table1_metrics_all_seeds.csv", index=False)
    summary.to_csv(out / "table1_complete.csv", index=False)
    baselines.to_csv(out / "table1_baselines.csv", index=False)
    support.to_csv(out / "table1_support_audit.csv", index=False)
    (out / "table1_metadata.json").write_text(
        json.dumps({
            "seeds": seed_values,
            "systems": system_values,
            "support_rule": "all systems must have identical sample_id sets within each seed",
            "artifact_root": str(root),
        }, indent=2),
        encoding="utf-8",
    )
    return runs, summary, baselines
