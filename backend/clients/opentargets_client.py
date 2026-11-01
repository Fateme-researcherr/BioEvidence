"""
Open Targets client (Section 22).

Endpoint usage: GraphQL Platform API.
Auth: None.
Request strategy: single query per target-disease pair, minimal fields.
Rate-limit handling: self-throttle per guidance.
Caching: cache by (target, disease) pair.
Failure handling: retry, then mark score unavailable.
"""

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import requests

OPEN_TARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
SELF_THROTTLE_SECONDS = 1.0  # self-throttle per Open Targets usage guidance
CACHE_TTL_SECONDS = 30 * 24 * 60 * 60
MAX_RETRIES = 3

CACHE_DIR = os.environ.get("OPENTARGETS_CACHE_DIR", "/tmp/opentargets_cache")

ASSOCIATION_QUERY = """
query TargetDiseaseAssociation($ensemblId: String!, $efoId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    associatedDiseases(efoIds: [$efoId]) {
      rows {
        disease {
          id
        }
        score
      }
    }
  }
}
"""


class SelfThrottle:
    """Self-throttle per Open Targets usage guidance."""

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


class TargetDiseasePairCache:
    """Cache raw responses by (target, disease) pair."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self._dir = cache_dir
        os.makedirs(self._dir, exist_ok=True)

    def _path(self, target_id: str, disease_id: str) -> str:
        key = f"{target_id}|{disease_id}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return os.path.join(self._dir, f"{digest}.json")

    def get(self, target_id: str, disease_id: str) -> Optional[str]:
        path = self._path(target_id, disease_id)
        if not os.path.exists(path):
            return None
        age = time.time() - os.path.getmtime(path)
        if age > CACHE_TTL_SECONDS:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def set(self, target_id: str, disease_id: str, raw: str) -> None:
        path = self._path(target_id, disease_id)
        with open(path, "w", encoding="utf-8") as f:
            f.write(raw)


@dataclass
class OpenTargetsAssociationRecord:
    ensembl_id: str
    efo_id: str
    raw_content: str
    source_database: str
    retrieval_method: str
    score_available: bool


class OpenTargetsClient:
    """
    Open Targets client — single GraphQL query per target-disease pair,
    requesting minimal fields. On failure after retries, the association
    score is marked unavailable rather than defaulted.
    """

    def __init__(self):
        self._throttle = SelfThrottle(SELF_THROTTLE_SECONDS)
        self._cache = TargetDiseasePairCache()

    def _request_with_retry(self, variables: dict) -> Optional[requests.Response]:
        backoff = 1.0
        payload = {"query": ASSOCIATION_QUERY, "variables": variables}
        for attempt in range(1, MAX_RETRIES + 1):
            self._throttle.wait()
            try:
                response = requests.post(OPEN_TARGETS_GRAPHQL_URL, json=payload, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt == MAX_RETRIES:
                    return None
                time.sleep(backoff)
                backoff *= 2
        return None

    def fetch_target_disease_association(
        self, ensembl_id: str, efo_id: str
    ) -> OpenTargetsAssociationRecord:
        """Single query per (target, disease) pair, cached by that pair."""
        cached = self._cache.get(ensembl_id, efo_id)
        if cached is not None:
            return OpenTargetsAssociationRecord(
                ensembl_id=ensembl_id,
                efo_id=efo_id,
                raw_content=cached,
                source_database="OpenTargets",
                retrieval_method="Open Targets GraphQL Platform API",
                score_available=True,
            )

        response = self._request_with_retry({"ensemblId": ensembl_id, "efoId": efo_id})
        if response is None:
            # retries exhausted — mark score unavailable, not defaulted
            return OpenTargetsAssociationRecord(
                ensembl_id=ensembl_id,
                efo_id=efo_id,
                raw_content="",
                source_database="OpenTargets",
                retrieval_method="Open Targets GraphQL Platform API",
                score_available=False,
            )

        raw_content = response.text
        self._cache.set(ensembl_id, efo_id, raw_content)

        return OpenTargetsAssociationRecord(
            ensembl_id=ensembl_id,
            efo_id=efo_id,
            raw_content=raw_content,
            source_database="OpenTargets",
            retrieval_method="Open Targets GraphQL Platform API",
            score_available=True,
        )
