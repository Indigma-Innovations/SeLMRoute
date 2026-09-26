import os
from pathlib import Path

import numpy as np
import pandas as pd


def build_embeddings(
    samples: pd.DataFrame,
    output: str | Path,
    *,
    model_name: str | None = None,
    revision: str | None = None,
    batch_size: int = 8,
    trust_remote_code: bool | None = None,
) -> dict[str, object]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install with `uv sync --extra embeddings`") from exc

    model_name = model_name or os.getenv("SELMROUTE_EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
    revision = revision or os.getenv("SELMROUTE_EMBED_REVISION")
    if trust_remote_code is None:
        trust_remote_code = os.getenv("SELMROUTE_EMBED_TRUST_REMOTE_CODE", "0").lower() in {"1", "true", "yes"}

    kwargs: dict[str, object] = {"trust_remote_code": bool(trust_remote_code)}
    if revision:
        kwargs["revision"] = revision
    encoder = SentenceTransformer(model_name, **kwargs)

    frame = samples.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    emb = encoder.encode(
        frame["query"].astype(str).tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    emb = np.asarray(emb, dtype=np.float32)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        sample_ids=np.asarray(frame["sample_id"].astype(str).tolist(), dtype="U"),
        embeddings=emb,
        model_name=np.array(model_name),
        revision=np.array(revision or ""),
    )
    return {
        "n_samples": len(frame),
        "dimension": int(emb.shape[1]),
        "model_name": model_name,
        "revision": revision,
        "output": str(target),
    }


def load_embeddings(path: str | Path) -> tuple[list[str], np.ndarray, dict[str, str]]:
    with np.load(Path(path), allow_pickle=False) as data:
        ids = [str(x) for x in data["sample_ids"].tolist()]
        emb = np.asarray(data["embeddings"], dtype=np.float32)
        meta = {
            "model_name": str(data["model_name"].item()),
            "revision": str(data["revision"].item()),
        }
    return ids, emb, meta
