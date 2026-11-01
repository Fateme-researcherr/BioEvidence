"""
Europe PMC client (Section 22).

Endpoint usage: REST `/search`, `/fullTextXML`.
Auth: None.
Request strategy: query by disease/gene terms.
Rate-limit handling: self-throttle 1 req/sec.
Caching: cache by PMCID.
Failure handling: retry x3 exponential backoff, then log/skip (same retry pattern as NCBI).
"""

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import requests

EUROPE_PMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
SELF_THROTTLE_SECONDS = 1.0  # 1 req/sec
CACHE_TTL_SECONDS = 30 * 24 * 60 * 60  # consistent 30-day TTL convention
MAX_RETRIES = 3

CACHE_DIR = os.environ.get("EUROPEPMC_CACHE_DIR", "/tmp/europepmc_cache")


class SelfThrottle:
    """Simple self-throttle: at most one request per SELF_THROTTLE_SECONDS."""

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


class PMCIDCache:
    """Cache raw responses by PMCID."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self._dir = cache_dir
        os.makedirs(self._dir, exist_ok=True)

    def _path(self, key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return os.path.join(self._dir, f"{digest}.xml")

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
class EuropePMCRecord:
    pmcid: str
    raw_content: str
    source_database: str
    retrieval_method: str


class EuropePMCClient:
    """
    Europe PMC client.

    Query by disease/gene terms via `/search`; fetch full text via
    `/fullTextXML`. Self-throttled to 1 req/sec.
    """

    def __init__(self):
        self._throttle = SelfThrottle(SELF_THROTTLE_SECONDS)
        self._cache = PMCIDCache()

    def _request_with_retry(self, url: str, params: dict) -> Optional[requests.Response]:
        backoff = 1.0
        for attempt in range(1, MAX_RETRIES + 1):
            self._throttle.wait()
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt == MAX_RETRIES:
                    # log/skip after exhausting retries
                    return None
                time.sleep(backoff)
                backoff *= 2
        return None

    def search(self, query: str, page_size: int = 25) -> List[str]:
        """Query by disease/gene terms, returning a list of PMCIDs."""
        url = f"{EUROPE_PMC_BASE}/search"
        params = {"query": query, "format": "json", "pageSize": page_size}
        response = self._request_with_retry(url, params)
        if response is None:
            return []
        data = response.json()
        results = data.get("resultList", {}).get("result", [])
        return [r["pmcid"] for r in results if "pmcid" in r]

    def fetch_full_text(self, pmcid: str) -> Optional[EuropePMCRecord]:
        """Fetch full-text XML for a single PMCID, cached by PMCID."""
        cached = self._cache.get(pmcid)
        if cached is not None:
            raw_content = cached
        else:
            url = f"{EUROPE_PMC_BASE}/{pmcid}/fullTextXML"
            response = self._request_with_retry(url, {})
            if response is None:
                return None
            raw_content = response.text
            self._cache.set(pmcid, raw_content)

        return EuropePMCRecord(
            pmcid=pmcid,
            raw_content=raw_content,
            source_database="EuropePMC",
            retrieval_method="Europe PMC REST /fullTextXML",
        )
