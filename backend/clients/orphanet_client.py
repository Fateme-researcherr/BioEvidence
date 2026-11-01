"""
Orphanet client (Section 22).

Endpoint usage: Orphadata bulk XML/RDF exports.
Auth: None.
Request strategy: scheduled weekly bulk refresh, not live per-query.
Rate-limit handling: N/A.
Caching: local normalized copy, weekly refresh.
Failure handling: fallback to last successful bulk load.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

import requests

ORPHADATA_BULK_URL = os.environ.get(
    "ORPHADATA_BULK_URL",
    "http://www.orphadata.org/data/xml/en_product1.xml",
)
REFRESH_INTERVAL_SECONDS = 7 * 24 * 60 * 60  # weekly refresh
MAX_RETRIES = 3

LOCAL_STORE_DIR = os.environ.get("ORPHANET_STORE_DIR", "/tmp/orphanet_store")
CURRENT_FILE = "current.xml"
LAST_GOOD_FILE = "last_good.xml"


@dataclass
class OrphanetBulkLoad:
    raw_content: str
    source_database: str
    retrieval_method: str
    loaded_at: float


class OrphanetClient:
    """
    Orphanet client — scheduled weekly bulk refresh from Orphadata exports,
    with fallback to the last successful bulk load on failure.
    """

    def __init__(self, store_dir: str = LOCAL_STORE_DIR):
        self._store_dir = store_dir
        os.makedirs(self._store_dir, exist_ok=True)

    def _path(self, filename: str) -> str:
        return os.path.join(self._store_dir, filename)

    def _needs_refresh(self) -> bool:
        current_path = self._path(CURRENT_FILE)
        if not os.path.exists(current_path):
            return True
        age = time.time() - os.path.getmtime(current_path)
        return age > REFRESH_INTERVAL_SECONDS

    def _download_bulk(self) -> Optional[str]:
    """
    Download the Orphadata bulk XML using HTTP Range requests.

    The source remains the official Orphadata endpoint, but the file is
    downloaded in chunks so a temporary connection reset does not force
    the entire 54+ MB file to restart.
    """
    chunk_size = 4 * 1024 * 1024  # 4 MB
    total_size = None
    chunks = []

    # First request: determine the total file size and verify that
    # the server supports HTTP Range requests.
    try:
        response = requests.get(
            ORPHADATA_BULK_URL,
            headers={"Range": "bytes=0-0"},
            timeout=30,
        )
        response.raise_for_status()

        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith("bytes 0-0/"):
            return None

        total_size = int(content_range.split("/")[-1])
        chunks.append(response.content)

    except (requests.RequestException, ValueError):
        return None

    # Download the remaining bytes in 4 MB ranges.
    start = 1

    while start < total_size:
        end = min(start + chunk_size - 1, total_size - 1)

        downloaded = False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.get(
                    ORPHADATA_BULK_URL,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=120,
                )
                response.raise_for_status()

                content_range = response.headers.get("Content-Range", "")

                expected_prefix = f"bytes {start}-{end}/"

                if not content_range.startswith(expected_prefix):
                    raise requests.RequestException(
                        f"Unexpected Content-Range: {content_range}"
                    )

                chunks.append(response.content)
                downloaded = True
                break

            except requests.RequestException:
                if attempt == MAX_RETRIES:
                    return None

                time.sleep(2 ** (attempt - 1))

        if not downloaded:
            return None

        start = end + 1

    raw_bytes = b"".join(chunks)

    if len(raw_bytes) != total_size:
        return None

    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None

    
        
        
            
            
            
            
                
                
                

    def refresh_if_needed(self) -> None:
        """Perform the scheduled weekly bulk refresh if due."""
        if not self._needs_refresh():
            return

        raw_content = self._download_bulk()
        if raw_content is None:
            # Fallback to last successful bulk load — no update this cycle.
            return

        current_path = self._path(CURRENT_FILE)
        with open(current_path, "w", encoding="utf-8") as f:
            f.write(raw_content)

        # Promote to last-known-good copy on successful bulk load.
        last_good_path = self._path(LAST_GOOD_FILE)
        with open(last_good_path, "w", encoding="utf-8") as f:
            f.write(raw_content)

    def load(self) -> Optional[OrphanetBulkLoad]:
        """
        Return the local normalized bulk copy, refreshing weekly if due.
        Falls back to the last successful bulk load if refresh fails
        or no current copy exists yet.
        """
        self.refresh_if_needed()

        current_path = self._path(CURRENT_FILE)
        last_good_path = self._path(LAST_GOOD_FILE)

        path_to_read = current_path if os.path.exists(current_path) else last_good_path
        if not os.path.exists(path_to_read):
            return None

        with open(path_to_read, "r", encoding="utf-8") as f:
            raw_content = f.read()

        return OrphanetBulkLoad(
            raw_content=raw_content,
            source_database="Orphanet",
            retrieval_method="Orphadata bulk XML export (scheduled weekly refresh)",
            loaded_at=os.path.getmtime(path_to_read),
        )
