import hashlib
import re

import numpy as np
import pandas as pd


def _normalize_group_text(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def _query_group_ids(samples: pd.DataFrame) -> pd.Series:
    """Stable leakage groups for the actual router input.

    Identical query text within a dataset is assigned to the same group so an
    identical semantic/router input can never appear on both sides of an ID split.
    Dataset is included because the benchmark stratifies within dataset and the same
    short question in two unrelated datasets need not denote the same task instance.
    """
    if "query" not in samples.columns:
        raise ValueError("group-aware splitting requires a 'query' column in samples")
    dataset = samples["dataset"].astype(str)
    query = samples["query"].map(_normalize_group_text)
    return pd.Series(
        [hashlib.sha256(f"{d}\0{q}".encode()).hexdigest() for d, q in zip(dataset, query, strict=True)],
        index=samples.index,
        name="leakage_group",
    )


def id_split(
    samples: pd.DataFrame,
    train_ratio: float = 0.7,
    seed: int = 3407,
    *,
    group_duplicates: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    rng = np.random.default_rng(seed)
    train_ids: list[str] = []
    test_ids: list[str] = []
    s = samples.copy()
    s["sample_id"] = s["sample_id"].astype(str)
    if group_duplicates:
        s["_leakage_group"] = _query_group_ids(s)

    for _, group in s.groupby("dataset", sort=True):
        if not group_duplicates:
            ids = group["sample_id"].to_numpy(copy=True)
            rng.shuffle(ids)
            n_train = max(1, min(len(ids) - 1, int(round(len(ids) * train_ratio)))) if len(ids) > 1 else 1
            train_ids.extend(ids[:n_train])
            test_ids.extend(ids[n_train:])
            continue

        grouped = [g["sample_id"].astype(str).tolist() for _, g in group.groupby("_leakage_group", sort=True)]
        order = np.arange(len(grouped))
        rng.shuffle(order)
        if len(grouped) <= 1:
            # Degenerate datasets are not present in the paper benchmark, but keep
            # the historical behavior of assigning the only sample to train.
            train_groups = set(order.tolist())
        else:
            n_train_groups = max(1, min(len(grouped) - 1, int(round(len(grouped) * train_ratio))))
            train_groups = set(order[:n_train_groups].tolist())
        for i, ids in enumerate(grouped):
            (train_ids if i in train_groups else test_ids).extend(ids)

    return np.array(train_ids, dtype=object), np.array(test_ids, dtype=object)


def dataset_ood_split(samples: pd.DataFrame, holdout: str) -> tuple[np.ndarray, np.ndarray]:
    mask = samples["dataset"].astype(str).str.lower() == holdout.lower()
    if not mask.any():
        raise ValueError(f"Unknown holdout dataset: {holdout}")
    return samples.loc[~mask, "sample_id"].to_numpy(), samples.loc[mask, "sample_id"].to_numpy()


def domain_ood_split(samples: pd.DataFrame, holdout: str) -> tuple[np.ndarray, np.ndarray]:
    mask = samples["domain"].astype(str).str.lower() == holdout.lower()
    if not mask.any():
        raise ValueError(f"Unknown holdout domain: {holdout}")
    return samples.loc[~mask, "sample_id"].to_numpy(), samples.loc[mask, "sample_id"].to_numpy()
