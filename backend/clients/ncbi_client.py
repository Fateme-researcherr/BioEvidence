"""
NCBI client (Section 22, Patch 6).

Exposes two clearly separated interfaces wrapping the same underlying
E-utilities service:

  - PubMedClient  (db=pubmed) — consumed ONLY by the Evidence Agent.
  - NCBIGeneClient (db=gene)  — consumed ONLY by the Ingestion & Curation Agent.

Both interfaces share a single internal E-utilities transport
(_EUtilitiesTransport) but are never invoked interchangeably by the wrong
agent (Section 22, Patch 6).

Request strategy: esearch -> efetch, batched PMIDs (<=200/request).
Rate-limit handling: token-bucket, 3/sec (no key) or 10/sec (with key).
Caching: raw XML by PMID, 30-day TTL.
Failure handling: retry x3 exponential backoff, then log/skip.
"""

import hashlib
import os
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional

import requests

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MAX_BATCH_SIZE = 200
CACHE_TTL_SECONDS = 30 * 24 * 60 * 60  # 30-day TTL
MAX_RETRIES = 3

CACHE_DIR = os.environ.get("NCBI_CACHE_DIR", "/tmp/ncbi_cache")


class TokenBucket:
    """Token-bucket rate limiter: 3/sec (no key) or 10/sec (with key)."""

    def __init__(self, rate_per_sec: float):
        self._rate = rate_per_sec
        self._tokens = rate_per_sec
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
            if self._tokens < 1:
                wait = (1 - self._tokens) / self._rate
                time.sleep(wait)
                self._tokens = 0
            else:
                self._tokens -= 1


class RawXMLCache:
    """Cache raw XML by PMID (or NCBI Gene ID), 30-day TTL."""

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

    def set(self, key: str, raw_xml: str) -> None:
        path = self._path(key)
        with open(path, "w", encoding="utf-8") as f:
            f.write(raw_xml)


@dataclass
class NCBIRecord:
    identifier: str
    raw_xml: str
    source_database: str
    retrieval_method: str


class _EUtilitiesTransport:
    """
    Internal shared transport for E-utilities esearch/efetch.

    Not exposed directly — only via PubMedClient / NCBIGeneClient below.
    """

    def __init__(self, db: str, api_key: Optional[str] = None):
        self._db = db
        self._api_key = api_key or os.environ.get("NCBI_API_KEY")
        rate = 10.0 if self._api_key else 3.0
        self._bucket = TokenBucket(rate)
        self._cache = RawXMLCache()

    def _params(self, extra: dict) -> dict:
        params = dict(extra)
        params["db"] = self._db
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    def _request_with_retry(self, path: str, params: dict) -> Optional[requests.Response]:
        url = f"{EUTILS_BASE}/{path}"
        backoff = 1.0
        for attempt in range(1, MAX_RETRIES + 1):
            self._bucket.acquire()
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

    def esearch(self, term: str, retmax: int = MAX_BATCH_SIZE) -> List[str]:
        params = self._params({"term": term, "retmax": retmax, "retmode": "json"})
        response = self._request_with_retry("esearch.fcgi", params)
        if response is None:
            return []
        data = response.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def efetch_batch(self, ids: List[str], retmode: str = "xml") -> List[NCBIRecord]:
        records: List[NCBIRecord] = []
        for i in range(0, len(ids), MAX_BATCH_SIZE):
            batch = ids[i : i + MAX_BATCH_SIZE]
            batch_key = ",".join(batch)

            cached = self._cache.get(batch_key)
            if cached is not None:
                raw_xml = cached
            else:
                params = self._params({"id": batch_key, "retmode": retmode})
                response = self._request_with_retry("efetch.fcgi", params)
                if response is None:
                    # retry exhausted, log/skip this batch
                    continue
                raw_xml = response.text
                self._cache.set(batch_key, raw_xml)

            records.extend(self._split_records(raw_xml, batch))
        return records

    def _split_records(self, raw_xml: str, ids: List[str]) -> List[NCBIRecord]:
        """Split a batched efetch XML response into per-ID cached raw XML entries."""
        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError:
            return []

        results: List[NCBIRecord] = []
        # Individual per-record XML is cached independently by ID as well,
        # so downstream lookups by a single PMID/GeneID hit the cache.
        for identifier, elem in zip(ids, list(root)):
            record_xml = ET.tostring(elem, encoding="unicode")
            self._cache.set(identifier, record_xml)
            results.append(
                NCBIRecord(
                    identifier=identifier,
                    raw_xml=record_xml,
                    source_database=self._source_database_tag(),
                    retrieval_method=self._retrieval_method_tag(),
                )
            )
        return results

    def _source_database_tag(self) -> str:
        return "PubMed" if self._db == "pubmed" else "NCBIGene"

    def _retrieval_method_tag(self) -> str:
        return f"NCBI E-utilities efetch via requests (db={self._db})"


class PubMedClient:
    """
    PubMed-facing interface (db=pubmed).

    Consumed ONLY by the Evidence Agent (Section 22, Patch 6).
    """

    def __init__(self, api_key: Optional[str] = None):
        self._transport = _EUtilitiesTransport(db="pubmed", api_key=api_key)

    def search(self, term: str, retmax: int = MAX_BATCH_SIZE) -> List[str]:
        return self._transport.esearch(term, retmax=retmax)

    def fetch(self, pmids: List[str]) -> List[NCBIRecord]:
        return self._transport.efetch_batch(pmids, retmode="xml")


class NCBIGeneClient:
    """
    NCBI Gene-facing interface (db=gene).

    Consumed ONLY by the Ingestion & Curation Agent (Section 22, Patch 6).
    """

    def __init__(self, api_key: Optional[str] = None):
        self._transport = _EUtilitiesTransport(db="gene", api_key=api_key)

    def search(self, term: str, retmax: int = MAX_BATCH_SIZE) -> List[str]:
        return self._transport.esearch(term, retmax=retmax)

    def fetch(self, gene_ids: List[str]) -> List[NCBIRecord]:
        return self._transport.efetch_batch(gene_ids, retmode="xml")
