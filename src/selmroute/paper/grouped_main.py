import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from selmroute.evaluation.embedding import embedding_evaluate
from selmroute.evaluation.run import train_evaluate
from selmroute.paper.ablation import ABLATION_MODES, semantic_ablation_suite
from selmroute.paper.benchmark import benchmark_seed_suite
from selmroute.paper.crossval import semantic_oof
from selmroute.paper.nested_lite import nested_lite_oof
from selmroute.paper.table1 import build_complete_table1

# from selmroute.paper.model_ablation import model_family_ablation


def rerun_main_grouped(
    samples: pd.DataFrame,
    outcomes: pd.DataFrame,
    features: pd.DataFrame,
    *,
    output_root: str | Path,
    gte_embeddings: str | Path,
    seeds: Iterable[int] = (42, 999, 2024, 2025, 3407),
    train_ratio: float = 0.7,
    laya_features: pd.DataFrame | None = None,
    run_nested_lite: bool = False,
) -> dict[str, object]:
    """Re-run paper ID experiments with duplicate router inputs kept together.

    Uses frozen semantic features/embeddings only; performs no semantic-backend calls.
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    seed_values = [int(x) for x in seeds]

    # Six semantic representation modes, including the three used in Table 1.
    for seed in seed_values:
        semantic_ablation_suite(
            samples, outcomes, features,
            root / f"runs/paper/ablations/{seed}",
            modes=ABLATION_MODES,
            split="id-grouped",
            train_ratio=train_ratio,
            seed=seed,
            router="catboost",
        )
        train_evaluate(
            samples, outcomes, features,
            root / f"runs/seeds/{seed}/domain_only",
            router="catboost", split="id-grouped", train_ratio=train_ratio,
            seed=seed, feature_set="domain_only",
        )
        train_evaluate(
            samples, outcomes, features,
            root / f"runs/seeds/{seed}/tfidf",
            router="tfidf", split="id-grouped", train_ratio=train_ratio,
            seed=seed, feature_set="all",
        )
        embedding_evaluate(
            samples, outcomes, gte_embeddings,
            root / f"runs/paper/gte_catboost/{seed}",
            split="id-grouped", train_ratio=train_ratio, seed=seed,
            router="catboost",
        )

    # Fully populated main table.
    _, table1, baselines = build_complete_table1(
        samples, outcomes, artifact_root=root,
        output_dir=root / "paper/main_table",
        seeds=seed_values,
    )

    # Representation summary including modes not present in Table 1.
    specs = [
        f"{mode}={root}/runs/paper/ablations/{{seed}}/{mode}/predictions.csv"
        for mode in ABLATION_MODES
    ]
    _, rep_summary = benchmark_seed_suite(
        samples, outcomes, system_specs=specs, seeds=seed_values,
        output_dir=root / "paper/representation_grouped",
    )
    rep_summary.to_csv(root / "paper/representation_grouped/summary.csv", index=False)

    # Downstream model-family ablation with the same grouped split policy.
    # model_summary = model_family_ablation(
    #     samples, outcomes, features,
    #     root / "paper/model_ablation_grouped",
    #     seeds=seed_values, representation="probability_mass",
    #     train_ratio=train_ratio, split="id-grouped",
    # )

    # Group-safe OOF for the central statistical comparison.
    pm_oof = semantic_oof(
        samples, outcomes, features,
        root / "paper/cv_grouped/probability_mass",
        representation="probability_mass", router="catboost",
        n_folds=5, seed=3407, semantic_prefix="jev", group_duplicates=True,
    )
    hard_oof = semantic_oof(
        samples, outcomes, features,
        root / "paper/cv_grouped/hard",
        representation="hard", router="catboost",
        n_folds=5, seed=3407, semantic_prefix="jev", group_duplicates=True,
    )

    laya_oof = None
    if laya_features is not None:
        laya_oof = semantic_oof(
            samples, outcomes, laya_features,
            root / "paper/laya_grouped/cv",
            representation="probability_mass", router="catboost",
            n_folds=5, seed=3407, semantic_prefix="laya", group_duplicates=True,
        )

    nested = None
    if run_nested_lite:
        nested = nested_lite_oof(
            samples, outcomes, features,
            root / "paper/nested_lite12_grouped",
            keep_probes=12, outer_folds=5, inner_folds=3,
            seed=3407, router="catboost", margin_pp=0.5,
            n_bootstrap=10000, group_duplicates=True,
        )

    payload = {
        "protocol": "duplicate-router-input-safe ID evaluation",
        "split": "id-grouped",
        "grouping": "dataset + normalized query",
        "seeds": seed_values,
        "table1_path": str(root / "paper/main_table/table1_complete.csv"),
        "probability_mass_oof": pm_oof,
        "hard_oof": hard_oof,
        "laya_oof": laya_oof,
        "nested_lite": nested,
        "note": "Uses frozen features/embeddings only; no JEV/Laya inference is performed.",
    }
    (root / "paper/grouped_rerun_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
