import json
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from selmroute.routing.catboost_router import CatBoostQualityRouter


def _quantiles(arr: np.ndarray, prefix: str) -> dict[str, float]:
    if arr.size == 0:
        return {}
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_p50": float(np.quantile(arr, 0.50)),
        f"{prefix}_p95": float(np.quantile(arr, 0.95)),
    }


def profile_jev_features(
    features: pd.DataFrame,
    output: str | Path,
    *,
    input_price_per_million: float = 0.0,
) -> dict[str, object]:
    result: dict[str, object] = {"n_samples": int(len(features))}
    if "meta_jev_input_tokens" in features.columns:
        tokens = features["meta_jev_input_tokens"].dropna().to_numpy(float)
        result.update(_quantiles(tokens, "input_tokens"))
        result["input_tokens_total"] = float(tokens.sum())
        if input_price_per_million > 0:
            result["input_price_per_million"] = float(input_price_per_million)
            result["estimated_input_cost_total"] = float(tokens.sum() / 1_000_000.0 * input_price_per_million)
            result["estimated_input_cost_per_query"] = float(tokens.mean() / 1_000_000.0 * input_price_per_million)
    if "meta_jev_latency_ms" in features.columns:
        latency = features["meta_jev_latency_ms"].dropna().to_numpy(float)
        result.update(_quantiles(latency, "latency_ms"))
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def profile_router_model(
    features: pd.DataFrame,
    model_dir: str | Path,
    output: str | Path,
    *,
    batch_sizes: Iterable[int] = (1, 32, 256),
    repeats: int = 100,
    seed: int = 3407,
) -> pd.DataFrame:
    router = CatBoostQualityRouter.load(model_dir)
    f = features.copy()
    f["sample_id"] = f["sample_id"].astype(str)
    cols = router.feature_names
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    # Warm up once.
    warm = f.iloc[: min(8, len(f))][["sample_id", *cols]].copy()
    router.predict(warm)
    for requested in batch_sizes:
        batch = min(int(requested), len(f))
        if batch <= 0:
            continue
        durations: list[float] = []
        for _ in range(repeats):
            idx = rng.choice(len(f), size=batch, replace=False)
            X = f.iloc[idx][["sample_id", *cols]].copy()
            t0 = time.perf_counter_ns()
            pred = router.predict(X)
            np.argmax(pred, axis=1)
            t1 = time.perf_counter_ns()
            durations.append((t1 - t0) / 1_000_000.0)
        arr = np.asarray(durations)
        rows.append({
            "batch_size": batch,
            "repeats": repeats,
            "batch_ms_mean": float(arr.mean()),
            "batch_ms_p50": float(np.quantile(arr, 0.50)),
            "batch_ms_p95": float(np.quantile(arr, 0.95)),
            "query_ms_mean": float(arr.mean() / batch),
            "query_ms_p50_equiv": float(np.quantile(arr, 0.50) / batch),
            "query_ms_p95_equiv": float(np.quantile(arr, 0.95) / batch),
            "n_features": len(cols),
            "n_candidates": len(router.model_names),
        })
    result = pd.DataFrame(rows)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False)
    return result


def profile_embedding_encoder(
    samples: pd.DataFrame,
    output: str | Path,
    *,
    model_name: str,
    revision: str | None = None,
    n_samples: int = 256,
    batch_size: int = 8,
    trust_remote_code: bool = False,
    device: str | None = None,
    seed: int = 3407,
) -> dict[str, object]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install with `uv sync --extra embeddings`") from exc

    kwargs: dict[str, object] = {"trust_remote_code": trust_remote_code}
    if revision:
        kwargs["revision"] = revision
    if device:
        kwargs["device"] = device
    load_start = time.perf_counter()
    encoder = SentenceTransformer(model_name, **kwargs)
    load_seconds = time.perf_counter() - load_start

    frame = samples.copy()
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(frame))
    idx = rng.choice(len(frame), size=n, replace=False)
    texts = frame.iloc[idx]["query"].astype(str).tolist()
    # Warmup outside measurement.
    encoder.encode(texts[: min(batch_size, len(texts))], batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)

    cuda_peak = None
    try:
        import torch
        if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except Exception:
        torch = None  # type: ignore[assignment]

    t0 = time.perf_counter()
    emb = encoder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    try:
        if torch is not None and torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
            torch.cuda.synchronize()
            cuda_peak = int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    elapsed = time.perf_counter() - t0
    arr = np.asarray(emb)
    result: dict[str, object] = {
        "model_name": model_name,
        "revision": revision,
        "device": device,
        "n_samples": n,
        "batch_size": batch_size,
        "dimension": int(arr.shape[1]),
        "model_load_seconds": float(load_seconds),
        "encode_seconds": float(elapsed),
        "queries_per_second": float(n / elapsed),
        "ms_per_query": float(elapsed * 1000.0 / n),
    }
    if cuda_peak is not None:
        result["cuda_peak_memory_bytes"] = cuda_peak
        result["cuda_peak_memory_gib"] = float(cuda_peak / (1024 ** 3))
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
