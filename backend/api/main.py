"""
FastAPI application (Section 27).

- POST /query        — submits a query; enqueues a `job` row (Section 20.1)
                        rather than blocking on external API latency; the
                        `worker` service (Patch 13, Section 28) picks it up.
- GET  /query/{job_id} — polls job status/result; no message queue needed
                        at MVP scale.

Authentication: API-key header auth backed by the `api_key` table
(Section 20.1) — sufficient for an MVP serving a small researcher group;
no OAuth needed yet.

Logging format: structured JSON per pipeline stage —
{timestamp, job_id, stage, status, source_manifest_delta, duration_ms}.

Internal agent APIs are not exposed via HTTP at MVP — agents are called
in-process by the Orchestration workflow controller (via the worker), not
from this module directly.
"""

import hashlib
import json
import logging
import os
import time
import uuid
from typing import Optional, Union

import psycopg2
import psycopg2.extras
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("api")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://app@postgres/biomed_evidence")

app = FastAPI(title="Biomedical Evidence Reasoning API")


def log_stage(job_id: str, stage: str, status: str, source_manifest_delta: int, duration_ms: float) -> None:
    """Structured JSON log per pipeline stage (Section 27)."""
    logger.info(
        json.dumps(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "job_id": job_id,
                "stage": stage,
                "status": status,
                "source_manifest_delta": source_manifest_delta,
                "duration_ms": duration_ms,
            }
        )
    )


def get_connection():
    connection = psycopg2.connect(DATABASE_URL)
    try:
        yield connection
    finally:
        connection.close()


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def verify_api_key(
    x_api_key: Optional[str] = Header(default=None),
    connection=Depends(get_connection),
) -> str:
    """
    API-key header auth backed by the `api_key` table (Section 20.1,
    Section 27). Rejects missing, unknown, or revoked keys.
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    key_hash = hash_api_key(x_api_key)
    with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT key_id, revoked_at FROM api_key WHERE key_hash = %(key_hash)s",
            {"key_hash": key_hash},
        )
        record = cur.fetchone()

    if record is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if record["revoked_at"] is not None:
        raise HTTPException(status_code=401, detail="API key has been revoked")

    return str(record["key_id"])


class QueryRequest(BaseModel):
    disease: str
    gene_target: str
    therapeutic_approach: str


class JobSubmittedResponse(BaseModel):
    job_id: str
    status: str


class BriefResponse(BaseModel):
    job_id: str
    status: str
    result: dict


class GapNoticeResponse(BaseModel):
    job_id: str
    status: str
    reason: str
    failed_sections: list


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[dict] = None


@app.post("/query", response_model=JobSubmittedResponse)
async def submit_query(
    request: QueryRequest,
    api_key_id: str = Depends(verify_api_key),
    connection=Depends(get_connection),
):
    """
    Submits a query. Runs as an async job (Section 27) rather than
    blocking on external API latency — enqueues a `job` row that the
    worker service (Patch 13) picks up.
    """
    job_id = uuid.uuid4()
    params = {
        "disease": request.disease,
        "gene": request.gene_target,
        "therapy": request.therapeutic_approach,
    }

    start = time.monotonic()
    with connection:
        with connection.cursor() as cur:
            cur.execute(
                """
                INSERT INTO job (job_id, status, params)
                VALUES (%(job_id)s, 'pending', %(params)s)
                """,
                {"job_id": str(job_id), "params": json.dumps(params)},
            )
    duration_ms = (time.monotonic() - start) * 1000

    log_stage(
        job_id=str(job_id),
        stage="submit_query",
        status="pending",
        source_manifest_delta=0,
        duration_ms=duration_ms,
    )

    return JobSubmittedResponse(job_id=str(job_id), status="pending")


@app.get("/query/{job_id}", response_model=Union[JobStatusResponse, BriefResponse, GapNoticeResponse])
async def get_query_status(
    job_id: str,
    api_key_id: str = Depends(verify_api_key),
    connection=Depends(get_connection),
):
    """GET /query/{job_id} polling endpoint (Section 27)."""
    with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT job_id, status, result FROM job WHERE job_id = %(job_id)s",
            {"job_id": job_id},
        )
        record = cur.fetchone()

    if record is None:
        raise HTTPException(status_code=404, detail="job_id not found")

    result = record["result"]
    if isinstance(result, str):
        result = json.loads(result)

    if record["status"] == "complete" and result is not None:
        return BriefResponse(job_id=str(record["job_id"]), status=record["status"], result=result)

    if record["status"] == "failed" and result is not None and "reason" in result:
        return GapNoticeResponse(
            job_id=str(record["job_id"]),
            status=record["status"],
            reason=result.get("reason", ""),
            failed_sections=result.get("failed_sections", []),
        )

    return JobStatusResponse(job_id=str(record["job_id"]), status=record["status"], result=result)
