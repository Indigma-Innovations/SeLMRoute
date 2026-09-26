import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from selmroute.evaluation.run import train_evaluate

ABLATION_MODES = ("full", "probability_mass", "no_entropy", "primary", "primary_entropy", "hard")


def _fine_semantic_columns(features: pd.DataFrame, semantic_prefix: str = "jev") -> list[str]:
    # Preserve the exact source-column order used by V1 semantic_manual.
    # CatBoost can use feature position in deterministic tie-breaking, so sorting
    # here would make the supposedly identical `full` control a different run.
    prefix = semantic_prefix.rstrip("_") + "_"
    return [
        c for c in features.columns
        if c.startswith(prefix) and "task_family" not in c and not c.startswith("meta_")
    ]


def _probe_prefix(col: str) -> str:
    return col.split("__", 1)[0]


def build_semantic_ablation_features(
    features: pd.DataFrame, mode: str, *, semantic_prefix: str = "jev"
) -> pd.DataFrame:
    if mode not in ABLATION_MODES:
        raise ValueError(f"Unknown ablation mode={mode}; expected one of {ABLATION_MODES}")
    src = features.copy()
    src["sample_id"] = src["sample_id"].astype(str)
    fine = _fine_semantic_columns(src, semantic_prefix=semantic_prefix)
    if mode == "full":
        return src[["sample_id", *fine]].copy()
    if mode == "probability_mass":
        cols = [c for c in fine if c.endswith("__noul") or "__p__" in c]
        return src[["sample_id", *cols]].copy()
    if mode == "no_entropy":
        cols = [c for c in fine if not c.endswith("__entropy")]
        return src[["sample_id", *cols]].copy()
    if mode == "primary":
        cols = [c for c in fine if c.endswith("__noul") or c.endswith("__expected")]
        return src[["sample_id", *cols]].copy()
    if mode == "primary_entropy":
        cols = [
            c for c in fine
            if c.endswith("__noul") or c.endswith("__expected") or c.endswith("__entropy")
        ]
        return src[["sample_id", *cols]].copy()

    # Hard semantic decisions: Noul -> {0,1}; Score -> argmax level from the
    # returned probability mass. This tests whether retaining probabilities matters.
    out = pd.DataFrame({"sample_id": src["sample_id"]})
    prefixes = list(dict.fromkeys(_probe_prefix(c) for c in fine))
    for prefix in prefixes:
        noul = f"{prefix}__noul"
        if noul in src.columns:
            out[f"{prefix}__hard"] = (src[noul].astype(float) >= 0.5).astype(float)
            continue
        pcols = sorted(c for c in fine if c.startswith(prefix + "__p__"))
        if not pcols:
            expected = f"{prefix}__expected"
            if expected in src.columns:
                out[f"{prefix}__hard"] = np.rint(src[expected].astype(float))
            continue
        coords: list[float] = []
        for pos, col in enumerate(pcols):
            raw = col.split("__p__", 1)[1]
            try:
                coords.append(float(raw))
            except ValueError:
                coords.append(float(pos))
        arr = src[pcols].to_numpy(float)
        winners = np.nanargmax(arr, axis=1)
        out[f"{prefix}__hard"] = np.asarray(coords, dtype=float)[winners]
    return out


def semantic_ablation_suite(
    samples: pd.DataFrame,
    outcomes: pd.DataFrame,
    features: pd.DataFrame,
    output_root: str | Path,
    *,
    modes: Iterable[str] = ABLATION_MODES,
    split: str = "id",
    holdout: str | None = None,
    train_ratio: float = 0.7,
    seed: int = 3407,
    router: str = "catboost",
) -> pd.DataFrame:
    root = Path(output_root)
    rows: list[dict[str, object]] = []
    for mode in modes:
        ablated = build_semantic_ablation_features(features, mode)
        run_dir = root / mode
        result = train_evaluate(
            samples,
            outcomes,
            ablated,
            run_dir,
            router=router,
            split=split,
            holdout=holdout,
            train_ratio=train_ratio,
            seed=seed,
            feature_set="all",
        )
        result["ablation_mode"] = mode
        result["ablation_feature_count"] = int(len(ablated.columns) - 1)
        (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        rows.append({"mode": mode, **result})
    summary = pd.DataFrame(rows).sort_values("avg_score", ascending=False)
    root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(root / "ablation_summary.csv", index=False)
    return summary
