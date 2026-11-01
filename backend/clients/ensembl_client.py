"""
Ensembl client (Section 22).

Endpoint usage: REST `/lookup/id`, `/xrefs`.
Auth: None.
Request strategy: ID cross-reference only, never primary evidence.
Rate-limit handling: self-throttle ~15 req/sec.
Caching: cache by Ensembl ID.
Failure handling: retry x3.
"""

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import requests

ENSEMBL_BASE = "https://rest.ensembl.org"
SELF_THROTTLE_SECONDS = 1.0 / 15.0  # ~15 req/sec
CACHE_TTL_SECONDS = 30 * 24 * 60 * 60
MAX_RETRIES = 3

CACHE_DIR = os.environ.get("ENSEMBL_CACHE_DIR", "/tmp/ensembl_cache")


class SelfThrottle:
    """Self-throttle: at most one request per SELF_THROTTLE_SECONDS (~15 req/sec)."""

    def __init__(self, min_interval_seconds: float):
        self._min_interval = min_interval_seconds
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


class EnsemblIDCache:
    """Cache raw responses by Ensembl ID."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self._dir = cache_dir
        os.makedirs(self._dir, exist_ok=True)

    def _path(self, key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return os.path.join(self._dir, f"{digest}.json")

    def get(self, key: str) -> Optional[str]:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        age = time.time() - os.path.getmtime(path)
        if age > CACHE_TTL_SECONDS:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def set(self, key: str, raw: str) -> None:
        path = self._path(key)
        with open(path, "w", encoding="utf-8") as f:
            f.write(raw)


@dataclass
class EnsemblRecord:
    ensembl_id: str
    raw_content: str
    source_database: str
    retrieval_method: str


class EnsemblClient:
    """
    Ensembl client — used ONLY for ID cross-reference resolution
    (e.g. Ensembl ID -> UniProt accession, HGNC ID), never as a source of
    primary evidence. Self-throttled to ~15 req/sec.
    """

    def __init__(self):
        self._throttle = SelfThrottle(SELF_THROTTLE_SECONDS)
        self._cache = EnsemblIDCache()

    def _request_with_retry(self, url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
        backoff = 1.0
        headers = {"Content-Type": "application/json"}
        for attempt in range(1, MAX_RETRIES + 1):
            self._throttle.wait()
            try:
                response = requests.get(url, params=params or {}, headers=headers, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt == MAX_RETRIES:
                    return None
                time.sleep(backoff)
                backoff *= 2
        return None

    def lookup(self, ensembl_id: str) -> Optional[EnsemblRecord]:
        """`/lookup/id` — resolve basic feature info for an Ensembl ID, cached by ID."""
        cache_key = f"lookup:{ensembl_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            raw_content = cached
        else:
            url = f"{ENSEMBL_BASE}/lookup/id/{ensembl_id}"
            response = self._request_with_retry(url)
            if response is None:
                return None
            raw_content = response.text
            self._cache.set(cache_key, raw_content)

        return EnsemblRecord(
            ensembl_id=ensembl_id,
            raw_content=raw_content,
            source_database="Ensembl",
            retrieval_method="Ensembl REST /lookup/id",
        )

    def xrefs(self, ensembl_id: str) -> Optional[EnsemblRecord]:
        """`/xrefs` — cross-reference an Ensembl ID to external identifiers, cached by ID."""
        cache_key = f"xrefs:{ensembl_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            raw_content = cached
        else:
            url = f"{ENSEMBL_BASE}/xrefs/id/{ensembl_id}"
            response = self._request_with_retry(url)
            if response is None:
                return None
            raw_content = response.text
            self._cache.set(cache_key, raw_content)

        return EnsemblRecord(
            ensembl_id=ensembl_id,
            raw_content=raw_content,
            source_database="Ensembl",
            retrieval_method="Ensembl REST /xrefs",
        )
