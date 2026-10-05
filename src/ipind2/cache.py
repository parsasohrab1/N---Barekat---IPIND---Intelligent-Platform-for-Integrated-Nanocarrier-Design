"""
Data-layer cache (SRS §3.1 architecture: "Cache (Redis)").

Default ``MemoryCache`` (thread-safe with TTL and a size cap) with no external dependency;
``RedisCache`` if ``IPIND_REDIS_URL`` is set and ``redis`` is installed. Both share one interface and
hold only JSON-serializable values (keys are strings).
"""

import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Optional, Protocol


class Cache(Protocol):
    def get(self, key: str) -> Optional[Any]: ...
    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None: ...
    def delete(self, key: str) -> None: ...


class MemoryCache:
    def __init__(self, max_items: int = 1024, default_ttl: float = 3600.0):
        self._data: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = threading.Lock()
        self.max_items = max_items
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires, value = item
            if expires < time.monotonic():
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        ttl = self.default_ttl if ttl_seconds is None else ttl_seconds
        with self._lock:
            self._data[key] = (time.monotonic() + ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)


class RedisCache:
    def __init__(self, url: str, prefix: str = "ipind2:", default_ttl: int = 3600):
        import redis

        self._client = redis.Redis.from_url(url, decode_responses=True)
        self.prefix = prefix
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        raw = self._client.get(self.prefix + key)
        return None if raw is None else json.loads(raw)

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        ttl = int(self.default_ttl if ttl_seconds is None else ttl_seconds)
        self._client.set(self.prefix + key, json.dumps(value), ex=max(ttl, 1))

    def delete(self, key: str) -> None:
        self._client.delete(self.prefix + key)


def make_cache() -> Cache:
    """Redis if ``IPIND_REDIS_URL`` is set and reachable; otherwise local memory."""
    url = os.environ.get("IPIND_REDIS_URL")
    if url:
        try:
            cache = RedisCache(url)
            cache._client.ping()
            return cache
        except Exception:  # Redis unavailable; fall back to local memory (logged by the caller)
            pass
    return MemoryCache()
