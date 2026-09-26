import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _entropy(probabilities: list[float]) -> float:
    arr = np.asarray(probabilities, dtype=float)
    arr = arr[arr > 0]
    if len(arr) == 0:
        return 0.0
    value = -float(np.sum(arr * np.log(arr)))
    denom = math.log(len(probabilities)) if len(probabilities) > 1 else 1.0
    return value / denom if denom > 0 else 0.0


def _numeric_probability_items(probabilities: Any) -> list[tuple[str, float]]:
    if isinstance(probabilities, dict):
        return [(str(k), float(v)) for k, v in probabilities.items()]
    if isinstance(probabilities, list):
        return [(str(i), float(v)) for i, v in enumerate(probabilities)]
    return []


def encode_answer(prefix: str, answer: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    atype = answer.get("type")
    if atype == "noul" or "noul" in answer:
        p = float(answer.get("noul", 0.5))
        out[f"{prefix}__noul"] = p
        out[f"{prefix}__entropy"] = _entropy([p, 1.0 - p])
        return out

    items = _numeric_probability_items(answer.get("probabilities"))
    probs = [v for _, v in items]
    if "confidence" in answer and answer["confidence"] is not None:
        out[f"{prefix}__confidence"] = float(answer["confidence"])
    if probs:
        out[f"{prefix}__entropy"] = _entropy(probs)
        for key, value in items:
            safe = key.replace(" ", "_").replace("/", "_")
            out[f"{prefix}__p__{safe}"] = value

    if atype == "score" or "score" in answer:
        score = float(answer.get("score", 0.0))
        out[f"{prefix}__score"] = score
        if items:
            # Prefer numeric keys; otherwise use criterion order as returned.
            coords = []
            for pos, (key, p) in enumerate(items):
                try:
                    level = float(key)
                except ValueError:
                    level = float(pos)
                coords.append((level, p))
            mean = sum(level * p for level, p in coords)
            second = sum(level * level * p for level, p in coords)
            out[f"{prefix}__expected"] = mean
            out[f"{prefix}__spread"] = math.sqrt(max(0.0, second - mean * mean))
    return out


def decision_features(path: str | Path, *, prefix: str) -> pd.DataFrame:
    """Encode typed-decision JSONL into SeLMRoute feature columns.

    ``prefix`` names the semantic backend (for example ``jev`` or ``laya``). The
    answer schema is shared across supported typed-decision backends, which lets us
    compare backends while keeping the downstream representation identical.
    """
    clean_prefix = prefix.rstrip("_")
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            row: dict[str, Any] = {"sample_id": str(record["sample_id"])}
            for qid, answer in record.get("answers", {}).items():
                row.update(encode_answer(f"{clean_prefix}_{qid}", answer))
            usage = record.get("usage", {}) or {}
            if usage.get("input_tokens") is not None:
                row[f"meta_{clean_prefix}_input_tokens"] = float(usage["input_tokens"])
            if record.get("latency_ms") is not None:
                row[f"meta_{clean_prefix}_latency_ms"] = float(record["latency_ms"])
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["sample_id"])
    return pd.DataFrame(rows).drop_duplicates("sample_id", keep="last")


def jev_features(path: str | Path) -> pd.DataFrame:
    return decision_features(path, prefix="jev")


def laya_features(path: str | Path) -> pd.DataFrame:
    return decision_features(path, prefix="laya")


def _build_feature_frame(
    samples: pd.DataFrame,
    semantic_results: str | Path | None,
    *,
    semantic_prefix: str,
) -> pd.DataFrame:
    from .deterministic import deterministic_features

    out = deterministic_features(samples)
    if semantic_results is not None:
        semantic = decision_features(semantic_results, prefix=semantic_prefix)
        out = out.merge(semantic, on="sample_id", how="left", validate="one_to_one")
    return out


def build_feature_frame(samples: pd.DataFrame, jev_results: str | Path | None = None) -> pd.DataFrame:
    return _build_feature_frame(samples, jev_results, semantic_prefix="jev")


def build_laya_feature_frame(samples: pd.DataFrame, laya_results: str | Path | None = None) -> pd.DataFrame:
    return _build_feature_frame(samples, laya_results, semantic_prefix="laya")
