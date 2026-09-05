"""
Client for external/mock data sources (Theme 3: "query multiple sources").

Uses JSONPlaceholder (https://jsonplaceholder.typicode.com) as a stand-in for
a real support/CRM backend:
- ``/users/{id}``  -> user profile
- ``/users/{id}/posts`` -> treated as the user's past support "threads"
- ``/posts/{id}/comments`` -> treated as replies/history on a thread

Reliability measures:
- Every call has an explicit timeout (``EXTERNAL_API_TIMEOUT_SECONDS``).
- Results are cached in-memory (reusing the template's ``LRUCache``) to
  reduce load and latency for repeated lookups.
- A minimal circuit breaker stops hammering a failing upstream: after
  ``EXTERNAL_API_FAILURE_THRESHOLD`` consecutive failures, calls short-circuit
  to a fallback (empty result) for ``EXTERNAL_API_COOLDOWN_SECONDS`` instead
  of blocking the request pipeline.
- Failures never raise into the caller - they degrade to "no data from this
  source" so a flaky external API cannot crash the agent, it only lowers
  confidence (fewer corroborating sources).
"""

import time
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import settings
from app.utils.lru_cache import LRUCache

_cache = LRUCache(capacity=settings.EXTERNAL_API_CACHE_SIZE)


class _CircuitBreaker:
    def __init__(self, failure_threshold: int, cooldown_seconds: float):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._failures = 0
        self._opened_at: Optional[float] = None

    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.cooldown_seconds:
            # Half-open: allow a trial request through.
            self._opened_at = None
            self._failures = 0
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()


_breaker = _CircuitBreaker(
    settings.EXTERNAL_API_FAILURE_THRESHOLD, settings.EXTERNAL_API_COOLDOWN_SECONDS
)


async def _get_json(path: str) -> Optional[Any]:
    cache_key = f"external:{path}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached.get("payload")

    if _breaker.is_open():
        return None

    url = f"{settings.EXTERNAL_API_BASE_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=settings.EXTERNAL_API_TIMEOUT_SECONDS) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError):
        _breaker.record_failure()
        return None

    _breaker.record_success()
    _cache.put(cache_key, {"payload": payload})
    return payload


async def get_user_profile(external_user_id: int) -> Optional[Dict[str, Any]]:
    """Fetch the user's external profile (name, email, company, address)."""
    data = await _get_json(f"/users/{external_user_id}")
    if not isinstance(data, dict):
        return None
    return data


async def get_user_history(external_user_id: int) -> List[Dict[str, Any]]:
    """
    Fetch the user's past "threads" (JSONPlaceholder posts), used as a proxy
    for prior support interactions / order history.
    """
    data = await _get_json(f"/users/{external_user_id}/posts")
    if not isinstance(data, list):
        return []
    return data
