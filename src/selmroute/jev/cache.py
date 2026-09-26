import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class SQLiteCache:
    def __init__(self, path: str | Path, *, store_requests: bool = False):
        self.path = Path(path)
        self.store_requests = store_requests
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jev_cache (
                cache_key TEXT PRIMARY KEY,
                request_json TEXT NOT NULL,
                response_json TEXT NOT NULL,
                requested_model TEXT NOT NULL,
                resolved_model TEXT,
                input_tokens INTEGER,
                latency_ms REAL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT response_json FROM jev_cache WHERE cache_key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, request: dict[str, Any], response: dict[str, Any], latency_ms: float) -> None:
        usage = response.get("usage", {}) or {}
        with self._lock:
            self._conn.execute(
                """INSERT OR IGNORE INTO jev_cache
                (cache_key, request_json, response_json, requested_model, resolved_model, input_tokens, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    key,
                    json.dumps(request if self.store_requests else {"redacted": True}, ensure_ascii=False, sort_keys=True),
                    json.dumps(response, ensure_ascii=False, sort_keys=True),
                    request.get("model", ""),
                    response.get("model"),
                    usage.get("input_tokens"),
                    latency_ms,
                ),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()
