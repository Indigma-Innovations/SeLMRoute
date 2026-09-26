import numpy as np
import pandas as pd


def selective_risk(regret: np.ndarray, uncertainty: np.ndarray, coverages: list[float] | None = None) -> pd.DataFrame:
    regret = np.asarray(regret, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    if len(regret) != len(uncertainty):
        raise ValueError("regret and uncertainty lengths differ")
    coverages = coverages or [1.0, .95, .9, .85, .8, .7, .6, .5]
    order = np.argsort(uncertainty)  # most certain first
    rows=[]
    n=len(regret)
    for coverage in sorted(set(coverages), reverse=True):
        k=max(1, int(round(n*coverage)))
        kept=order[:k]
        rows.append({"coverage": k/n, "risk": float(regret[kept].mean()), "n": k})
    return pd.DataFrame(rows).sort_values("coverage")


def aurc(curve: pd.DataFrame) -> float:
    x=curve["coverage"].to_numpy(float)
    y=curve["risk"].to_numpy(float)
    return float(np.trapezoid(y, x))
