from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from selmroute.evaluation.run import train_evaluate
from selmroute.paper.ablation import build_semantic_ablation_features
from selmroute.paper.benchmark import benchmark_seed_suite

MODEL_FAMILIES = ("catboost", "ols", "ridge", "random_forest", "mlp")


def model_family_ablation(
    samples: pd.DataFrame,
    outcomes: pd.DataFrame,
    features: pd.DataFrame,
    output_root: str | Path,
    *,
    routers: Iterable[str] = MODEL_FAMILIES,
    seeds: Iterable[int] = (42, 999, 2024, 2025, 3407),
    representation: str = "probability_mass",
    train_ratio: float = 0.7,
    split: str = "id",
) -> pd.DataFrame:
    root = Path(output_root)
    compact = build_semantic_ablation_features(features, representation)
    rows: list[dict[str, object]] = []
    router_values = [str(x) for x in routers]
    seed_values = [int(x) for x in seeds]
    for seed in seed_values:
        for router in router_values:
            out = root / str(seed) / router
            metrics = train_evaluate(
                samples, outcomes, compact, out,
                router=router, split=split, train_ratio=train_ratio, seed=seed, feature_set="all",
            )
            rows.append({"seed": seed, "router": router, **metrics})
    raw = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    raw.to_csv(root / "model_ablation_runs.csv", index=False)

    specs = [f"{router}={root}/{seed_placeholder}/{router}/predictions.csv" for router in router_values for seed_placeholder in ["{seed}"]]
    _, summary = benchmark_seed_suite(
        samples, outcomes, system_specs=specs, seeds=seed_values, output_dir=root / "benchmark",
    )
    summary = summary.rename(columns={"system": "router"})
    summary.to_csv(root / "model_ablation_5seed_summary.csv", index=False)
    return summary
