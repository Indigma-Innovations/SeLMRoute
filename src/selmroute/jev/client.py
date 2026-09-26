import asyncio
import email.utils
import os
import random
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from selmroute.jev.cache import SQLiteCache
from selmroute.schema import JevResponse, ProbeSet
from selmroute.util import sha256_json


class TypeSafeAPIError(RuntimeError):
    pass


def _retry_delay(response: httpx.Response | None, attempt: int, base: float = 0.5, cap: float = 20.0) -> float:
    if response is not None:
        ms = response.headers.get("retry-after-ms")
        if ms:
            try:
                return min(cap, max(0.0, float(ms) / 1000.0))
            except ValueError:
                pass
        value = response.headers.get("retry-after")
        if value:
            try:
                return min(cap, max(0.0, float(value)))
            except ValueError:
                try:
                    dt = email.utils.parsedate_to_datetime(value)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=UTC)
                    return min(cap, max(0.0, (dt - datetime.now(UTC)).total_seconds()))
                except Exception:
                    pass
    return min(cap, base * (2**attempt)) * random.uniform(0.8, 1.2)


class AsyncJevClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.typesafe.ai",
        timeout_s: float = 60.0,
        max_retries: int = 6,
        concurrency: int = 16,
        cache: SQLiteCache | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ValueError("TYPESAFE_API_KEY is required for real JEV evaluation")
        self.max_retries = max_retries
        self._sem = asyncio.Semaphore(concurrency)
        self.cache = cache
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s),
            limits=httpx.Limits(max_connections=max(concurrency * 2, 20), max_keepalive_connections=max(concurrency, 10)),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            transport=transport,
        )

    async def __aenter__(self) -> "AsyncJevClient":
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def build_request(query: str, probes: ProbeSet) -> dict[str, Any]:
        questions = {qid: q.model_dump(mode="json", exclude_none=True) for qid, q in probes.questions.items()}
        return {"state": {"query": query}, "model": probes.model, "questions": questions}

    async def evaluate(self, query: str, probes: ProbeSet) -> tuple[JevResponse, bool, float]:
        request = self.build_request(query, probes)
        key = sha256_json(request)
        if self.cache:
            cached = self.cache.get(key)
            if cached is not None:
                return JevResponse.model_validate(cached), True, 0.0

        async with self._sem:
            start = time.perf_counter()
            last_error: Exception | None = None
            for attempt in range(self.max_retries + 1):
                response: httpx.Response | None = None
                try:
                    response = await self._client.post("/v1/systemone", json=request)
                    if response.status_code < 400:
                        payload = response.json()
                        parsed = JevResponse.model_validate(payload)
                        latency_ms = (time.perf_counter() - start) * 1000.0
                        if self.cache:
                            self.cache.put(key, request, payload, latency_ms)
                        return parsed, False, latency_ms
                    if response.status_code not in {408, 409, 425, 429, 500, 502, 503, 504, 529}:
                        raise TypeSafeAPIError(f"TypeSafe HTTP {response.status_code}: {response.text[:1000]}")
                    last_error = TypeSafeAPIError(f"retryable TypeSafe HTTP {response.status_code}")
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    last_error = exc
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(_retry_delay(response, attempt))
            raise TypeSafeAPIError(f"TypeSafe request failed after retries: {last_error}")
