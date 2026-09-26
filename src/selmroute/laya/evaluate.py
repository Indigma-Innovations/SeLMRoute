import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from selmroute.schema import ProbeSet
from selmroute.util import stable_json


def _snapshot_revision(path: str | Path) -> str | None:
    parts = Path(path).parts
    if "snapshots" in parts:
        i = parts.index("snapshots")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def _resolve_laya_model(model: str, revision: str | None, subfolder: str | None) -> tuple[str, str | None]:
    local = Path(model)
    if local.exists():
        return str(local), revision

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Laya support requires the optional dependency set. Run: uv sync --extra laya"
        ) from exc

    prefix = f"{subfolder.strip('/')}/" if subfolder else ""
    patterns = [
        prefix + "rl_agent_config.json",
        prefix + "model.safetensors",
        prefix + "tokenizer/*",
        prefix + "encoder/*",
    ]
    path = snapshot_download(
        model,
        revision=revision,
        token=os.getenv("HF_TOKEN"),
        allow_patterns=patterns,
    )
    return str(path), _snapshot_revision(path) or revision


def _questions_from_probes(probes: ProbeSet) -> dict[str, dict[str, Any]]:
    return {
        qid: q.model_dump(mode="json", exclude_none=True)
        for qid, q in probes.questions.items()
    }


def evaluate_laya_samples(
    samples: pd.DataFrame,
    probes: ProbeSet,
    output: str | Path,
    *,
    model: str = "convaiinnovations/laya",
    revision: str | None = None,
    subfolder: str | None = None,
    device: str | None = None,
    limit: int | None = None,
) -> dict[str, object]:
    """Evaluate the SeLMRoute probe set with the open-weight Laya backend.

    Results intentionally mirror the JEV JSONL schema so that downstream feature
    extraction and routing evaluation are identical apart from the backend prefix.
    """
    required = {"sample_id", "query"}
    if missing := required - set(samples.columns):
        raise ValueError(f"samples missing columns: {sorted(missing)}")

    try:
        import laya
    except ImportError as exc:
        raise RuntimeError(
            "Laya support is optional. Install it with: uv sync --extra laya"
        ) from exc

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if target.exists():
        with target.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        done.add(str(json.loads(line)["sample_id"]))
                    except Exception:
                        continue

    pending = samples[~samples["sample_id"].astype(str).isin(done)].copy()
    if limit is not None:
        pending = pending.head(limit)

    model_path, resolved_revision = _resolve_laya_model(model, revision, subfolder)
    agent = laya.load(model_path, device=device, subfolder=subfolder)
    questions = _questions_from_probes(probes)
    stats: dict[str, object] = {
        "requested": int(len(pending)),
        "completed": 0,
        "failed": 0,
        "model": model,
        "resolved_revision": resolved_revision,
        "subfolder": subfolder,
        "device": str(getattr(agent, "device", device or "auto")),
    }

    for row in pending.itertuples(index=False):
        try:
            start = time.perf_counter()
            response = agent.predict({"query": str(row.query)}, questions)
            latency_ms = (time.perf_counter() - start) * 1000.0
            record = {
                "sample_id": str(row.sample_id),
                "query_hash": getattr(row, "query_hash", None),
                "probe_set_version": probes.version,
                "backend": "laya",
                "requested_model": model,
                "resolved_model": response.get("model", "laya-rl-agent"),
                "resolved_revision": resolved_revision,
                "answers": response.get("answers", {}),
                "usage": response.get("usage", {}) or {},
                "latency_ms": latency_ms,
                "cached": False,
            }
            with target.open("a", encoding="utf-8") as fh:
                fh.write(stable_json(record) + "\n")
            stats["completed"] = int(stats["completed"]) + 1
        except Exception as exc:
            stats["failed"] = int(stats["failed"]) + 1
            err = {"sample_id": str(row.sample_id), "error": f"{type(exc).__name__}: {exc}"}
            with target.with_suffix(target.suffix + ".errors.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(stable_json(err) + "\n")

    metadata = {
        "backend": "laya",
        "model": model,
        "resolved_revision": resolved_revision,
        "subfolder": subfolder,
        "device": stats["device"],
        "probe_set_version": probes.version,
        "n_questions": len(probes.questions),
    }
    target.with_suffix(target.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return stats
