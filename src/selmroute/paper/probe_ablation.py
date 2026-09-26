from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from selmroute.evaluation.run import train_evaluate
from selmroute.paper.ablation import build_semantic_ablation_features
from selmroute.paper.benchmark import benchmark_seed_suite


def _probe_name(col: str) -> str:
    return col.split("__", 1)[0].removeprefix("jev_")


def leave_one_probe_out(
    samples: pd.DataFrame,
    outcomes: pd.DataFrame,
    features: pd.DataFrame,
    output_root: str | Path,
    *, seeds: Iterable[int] = (42,999,2024,2025,3407), router: str = "catboost", split: str = "id",
) -> pd.DataFrame:
    root=Path(output_root)
    base=build_semantic_ablation_features(features,"probability_mass")
    probes=list(dict.fromkeys(_probe_name(c) for c in base.columns if c!="sample_id"))
    systems=["all_probes", *[f"drop_{p}" for p in probes]]
    for seed in [int(x) for x in seeds]:
        train_evaluate(samples,outcomes,base,root/str(seed)/"all_probes",router=router,split=split,seed=seed,feature_set="all")
        for p in probes:
            cols=[c for c in base.columns if c=="sample_id" or _probe_name(c)!=p]
            train_evaluate(samples,outcomes,base[cols],root/str(seed)/f"drop_{p}",router=router,split=split,seed=seed,feature_set="all")
    specs=[f"{name}={root}/{{seed}}/{name}/predictions.csv" for name in systems]
    _, summary=benchmark_seed_suite(samples,outcomes,system_specs=specs,seeds=seeds,output_dir=root/"benchmark")
    baseline=float(summary.loc[summary.system=="all_probes","AvgAcc_mean"].iloc[0])
    summary["drop_vs_all_pp"]=baseline-summary["AvgAcc_mean"]
    summary.to_csv(root/"leave_one_probe_out_summary.csv",index=False)
    return summary.sort_values("drop_vs_all_pp",ascending=False)
