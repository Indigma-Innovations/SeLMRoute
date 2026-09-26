import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def query_hash(query: str) -> str:
    normalized = " ".join(query.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(stable_json(row) + "\n")


def stable_seed(*parts: object, modulo: int = 2_147_483_647) -> int:
    """Derive a process-stable non-negative RNG seed from semantic identifiers.

    Unlike Python's built-in ``hash()``, this is stable across processes and
    machines. Experiments should key randomness by identities such as
    ``(global_seed, heldout_model, k, repeat, stage)`` rather than loop order.
    """

    if modulo <= 1:
        raise ValueError("modulo must be > 1")
    payload = stable_json(list(parts)).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return int(value % modulo)
