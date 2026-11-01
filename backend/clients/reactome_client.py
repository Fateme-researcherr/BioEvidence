"""
Reactome client (Section 22).

Endpoint usage: Content Service `/data/pathways/low/entity/{id}`.
Auth: None.
Request strategy: query by UniProt accession.
Rate-limit handling: self-throttle 5 req/sec.
Caching: cache by pathway ID.
Failure handling: retry x3.
"""

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import requests

REACTOME_BASE = "https://reactome.org/ContentService/data"
SELF_THROTTLE_SECONDS = 1.0 / 5.0  # 5 req/sec
CACHE_TTL_SECONDS = 30 * 24 * 60 * 60
MAX_RETRIES = 3

CACHE_DIR = os.environ.get("REACTOME_CACHE_DIR", "/tmp/reactome_cache")


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


class PathwayIDCache:
    """Cache raw responses by pathway ID."""

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
class ReactomePathwayRecord:
    entity_id: str
    raw_content: str
    source_database: str
    retrieval_method: str


class ReactomeClient:
    """
    Reactome client — queries the Content Service for low-level pathway
    entities by UniProt accession. Self-throttled to 5 req/sec.
    """

    def __init__(self):
        self._throttle = SelfThrottle(SELF_THROTTLE_SECONDS)
        self._cache = PathwayIDCache()

    def _request_with_retry(self, entity_id: str) -> Optional[requests.Response]:
        url = f"{REACTOME_BASE}/pathways/low/entity/{entity_id}"
        backoff = 1.0
        for attempt in range(1, MAX_RETRIES + 1):
            self._throttle.wait()
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt == MAX_RETRIES:
                    return None
                time.sleep(backoff)
                backoff *= 2
        return None

    def fetch_pathways_for_uniprot_accession(
        self, uniprot_accession: str
    ) -> Optional[ReactomePathwayRecord]:
        """Query low-level pathways for a given UniProt accession, cached by ID."""
        cached = self._cache.get(uniprot_accession)
        if cached is not None:
            raw_content = cached
        else:
            response = self._request_with_retry(uniprot_accession)
            if response is None:
                return None
            raw_content = response.text
            self._cache.set(uniprot_accession, raw_content)

        return ReactomePathwayRecord(
            entity_id=uniprot_accession,
            raw_content=raw_content,
            source_database="Reactome",
            retrieval_method="Reactome Content Service /data/pathways/low/entity/{id}",
        )
