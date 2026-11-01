"""
ChEMBL client (Section 22).

Endpoint usage: REST `/chembl/api/data/target`, `/activity`.
Auth: None.
Request strategy: query by UniProt-mapped ChEMBL target ID.
Rate-limit handling: self-throttle 5 req/sec.
Caching: cache by ChEMBL ID.
Failure handling: retry x3.
"""

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import requests

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
SELF_THROTTLE_SECONDS = 1.0 / 5.0  # 5 req/sec
CACHE_TTL_SECONDS = 30 * 24 * 60 * 60
MAX_RETRIES = 3

CACHE_DIR = os.environ.get("CHEMBL_CACHE_DIR", "/tmp/chembl_cache")


class SelfThrottle:
    """Self-throttle: at most one request per SELF_THROTTLE_SECONDS (5 req/sec)."""

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


class ChEMBLIDCache:
    """Cache raw responses by ChEMBL ID."""

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
class ChEMBLRecord:
    chembl_target_id: str
    raw_content: str
    source_database: str
    retrieval_method: str


class ChEMBLClient:
    """
    ChEMBL client — queries `/target` and `/activity` by a ChEMBL target ID
    that has already been resolved from a UniProt accession. Self-throttled
    to 5 req/sec.
    """

    def __init__(self):
        self._throttle = SelfThrottle(SELF_THROTTLE_SECONDS)
        self._cache = ChEMBLIDCache()

    def _request_with_retry(self, url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
        backoff = 1.0
        for attempt in range(1, MAX_RETRIES + 1):
            self._throttle.wait()
            try:
                response = requests.get(url, params=params or {"format": "json"}, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt == MAX_RETRIES:
                    return None
                time.sleep(backoff)
                backoff *= 2
        return None

    def fetch_target(self, chembl_target_id: str) -> Optional[ChEMBLRecord]:
        """`/chembl/api/data/target` — cached by ChEMBL target ID."""
        cache_key = f"target:{chembl_target_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            raw_content = cached
        else:
            url = f"{CHEMBL_BASE}/target/{chembl_target_id}"
            response = self._request_with_retry(url)
            if response is None:
                return None
            raw_content = response.text
            self._cache.set(cache_key, raw_content)

        return ChEMBLRecord(
            chembl_target_id=chembl_target_id,
            raw_content=raw_content,
            source_database="ChEMBL",
            retrieval_method="ChEMBL REST /chembl/api/data/target",
        )

    def fetch_activity(self, chembl_target_id: str) -> Optional[ChEMBLRecord]:
        """`/chembl/api/data/activity` — cached by ChEMBL target ID."""
        cache_key = f"activity:{chembl_target_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            raw_content = cached
        else:
            url = f"{CHEMBL_BASE}/activity"
            response = self._request_with_retry(url, params={"target_chembl_id": chembl_target_id, "format": "json"})
            if response is None:
                return None
            raw_content = response.text
            self._cache.set(cache_key, raw_content)

        return ChEMBLRecord(
            chembl_target_id=chembl_target_id,
            raw_content=raw_content,
            source_database="ChEMBL",
            retrieval_method="ChEMBL REST /chembl/api/data/activity",
        )
