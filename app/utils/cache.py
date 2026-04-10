"""
app/utils/cache.py
Cache hasil terjemahan dengan TTL — menghindari inference ulang untuk teks yang sama.
"""

import hashlib
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TranslationCache:
    """
    In-memory LRU cache dengan TTL untuk hasil terjemahan.
    Thread-safe untuk single-process deployment.
    Untuk multi-process, ganti dengan Redis.
    """

    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        self._cache: dict = {}
        self._max_size = max_size
        self._ttl = ttl_seconds

    def _make_key(self, text: str, src: str, tgt: str, num_beams: int) -> str:
        raw = f"{src}:{tgt}:{num_beams}:{text}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, text: str, src: str, tgt: str, num_beams: int = 4) -> Optional[dict]:
        key = self._make_key(text, src, tgt, num_beams)
        entry = self._cache.get(key)
        if entry is None:
            return None
        if (time.time() - entry["ts"]) > self._ttl:
            del self._cache[key]
            return None
        logger.debug(f"Cache HIT: {src}->{tgt} ({len(text)} chars)")
        return entry["data"]

    def set(self, text: str, src: str, tgt: str, result: dict, num_beams: int = 4):
        # Evict oldest entry saat cache penuh
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k]["ts"])
            del self._cache[oldest_key]

        key = self._make_key(text, src, tgt, num_beams)
        self._cache[key] = {"data": result, "ts": time.time()}

    def stats(self) -> dict:
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
        }

    def clear(self):
        self._cache.clear()


# Singleton — dipakai di seluruh aplikasi
translation_cache = TranslationCache(max_size=500, ttl_seconds=3600)
