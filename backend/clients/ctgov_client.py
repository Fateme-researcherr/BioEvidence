"""
ClinicalTrials.gov API v2 client (Section 22).

Endpoint usage: `/studies` with `query.cond`, `query.intr`, `pageToken`/`pageSize` explicit.
Auth: None (open).
Request strategy: paginate via `nextPageToken`.
Rate-limit handling: self-throttle 5 req/sec.
Caching: cache by NCT ID.
Failure handling: retry, then skip with manifest note.
"""

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import requests

CTGOV_BASE = "https://clinicaltrials.gov/api/v2"
SELF_THROTTLE_SECONDS = 1.0 / 5.0  # 5 req/sec
CACHE_TTL_SECONDS = 30 * 24 * 60 * 60
MAX_RETRIES = 3

CACHE_DIR = os.environ.get("CTGOV_CACHE_DIR", "/tmp/ctgov_cache")


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


class NCTIDCache:
    """Cache raw responses by NCT ID."""

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


class Manifest:
    """Records skip notes when retries are exhausted."""

    def __init__(self, manifest_dir: str = CACHE_DIR):
        self._path = os.path.join(manifest_dir, "manifest.log")
        os.makedirs(manifest_dir, exist_ok=True)

    def note_skip(self, identifier: str, reason: str) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} SKIP {identifier} {reason}\n")


@dataclass
class ClinicalTrialRecord:
    nct_id: str
    raw_content: str
    source_database: str
    retrieval_method: str


class ClinicalTrialsGovClient:
    """
    ClinicalTrials.gov API v2 client.

    Paginates `/studies` via `query.cond`, `query.intr`, and `nextPageToken`.
    Self-throttled to 5 req/sec.
    """

    def __init__(self):
        self._throttle = SelfThrottle(SELF_THROTTLE_SECONDS)
        self._cache = NCTIDCache()
        self._manifest = Manifest()

    def _request_with_retry(self, params: dict) -> Optional[requests.Response]:
        url = f"{CTGOV_BASE}/studies"
        backoff = 1.0
        for attempt in range(1, MAX_RETRIES + 1):
            self._throttle.wait()
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt == MAX_RETRIES:
                    return None
                time.sleep(backoff)
                backoff *= 2
        return None

    def search(
        self,
        condition: Optional[str] = None,
        intervention: Optional[str] = None,
        page_size: int = 50,
    ) -> List[dict]:
        """Paginate through /studies using nextPageToken until exhausted."""
        all_studies: List[dict] = []
        page_token = None

        while True:
            params = {"pageSize": page_size}
            if condition:
                params["query.cond"] = condition
            if intervention:
                params["query.intr"] = intervention
            if page_token:
                params["pageToken"] = page_token

            response = self._request_with_retry(params)
            if response is None:
                self._manifest.note_skip(
                    identifier=f"search(cond={condition}, intr={intervention})",
                    reason="retries exhausted",
                )
                break

            data = response.json()
            studies = data.get("studies", [])
            all_studies.extend(studies)

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return all_studies

    def fetch(self, nct_id: str) -> Optional[ClinicalTrialRecord]:
        """Fetch a single trial by NCT ID, cached by NCT ID."""
        cached = self._cache.get(nct_id)
        if cached is not None:
            raw_content = cached
        else:
            params = {"query.term": nct_id}
            response = self._request_with_retry(params)
            if response is None:
                self._manifest.note_skip(identifier=nct_id, reason="retries exhausted")
                return None
            raw_content = response.text
            self._cache.set(nct_id, raw_content)

        return ClinicalTrialRecord(
            nct_id=nct_id,
            raw_content=raw_content,
            source_database="ClinicalTrialsGov",
            retrieval_method="ClinicalTrials.gov API v2 /studies",
        )
