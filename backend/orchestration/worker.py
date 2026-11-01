"""
orchestration/worker.py (Patch 13).

A thin process entrypoint that polls the `job` table for pending jobs and
invokes the existing Orchestration workflow controller
(`WorkflowController.receive_query`, Section 21.5) — no logic beyond
job-queue polling and delegation to the existing controller.

Matches the `worker` service defined in docker-compose.yml (Section 28):
    worker:
      build: ./backend
      command: python -m orchestration.worker
"""

import dataclasses
import json
import logging
import os
import time
import uuid

import psycopg2
import psycopg2.extras

from orchestration.workflow_controller import IntelligenceBrief, StructuralError
from agents.explainability_agent import GapNotice

logger = logging.getLogger("orchestration.worker")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://app@postgres/biomed_evidence")
POLL_INTERVAL_SECONDS = float(os.environ.get("WORKER_POLL_INTERVAL_SECONDS", "2.0"))


def _to_jsonable(result) -> dict:
    """Convert a WorkflowController result (dataclass) into a JSON-serializable dict."""
    if dataclasses.is_dataclass(result):
        return dataclasses.asdict(result)
    return {"value": str(result)}


def _result_status(result) -> str:
    if isinstance(result, IntelligenceBrief):
        return "complete"
    if isinstance(result, (StructuralError, GapNotice)):
        return "failed"
    return "failed"


class Worker:
    """
    Polls the `job` table for pending jobs and delegates each one to the
    Orchestration workflow controller. No query logic lives here — this
    module is job-queue plumbing only.
    """

    def __init__(self, controller, connection):
        self._controller = controller
        self._conn = connection

    def run_forever(self) -> None:
        while True:
            job = self._claim_next_pending_job()
            if job is None:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            self._process_job(job)

    def _claim_next_pending_job(self):
        with self._conn:
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    UPDATE job
                    SET status = 'running', updated_at = now()
                    WHERE job_id = (
                        SELECT job_id FROM job
                        WHERE status = 'pending'
                        ORDER BY created_at ASC
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING job_id, params
                    """
                )
                return cur.fetchone()

    def _process_job(self, job) -> None:
        job_id = job["job_id"]
        params = job["params"]
        if isinstance(params, str):
            params = json.loads(params)

        try:
            result = self._controller.receive_query(
                disease=params.get("disease"),
                gene=params.get("gene"),
                therapy=params.get("therapy"),
            )
            status = _result_status(result)
            result_payload = _to_jsonable(result)
        except Exception as exc:  # job-level failure isolation only
            logger.exception("Job %s failed with an unhandled exception", job_id)
            status = "failed"
            result_payload = {"error": str(exc)}

        self._update_job(job_id, status, result_payload)

    def _update_job(self, job_id: uuid.UUID, status: str, result_payload: dict) -> None:
        with self._conn:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE job
                    SET status = %(status)s, result = %(result)s, updated_at = now()
                    WHERE job_id = %(job_id)s
                    """,
                    {
                        "status": status,
                        "result": json.dumps(result_payload),
                        "job_id": str(job_id),
                    },
                )


def main() -> None:
    from orchestration.workflow_controller import WorkflowController
    from orchestration.gap_rules import GapRules
    from agents.evidence_agent import EvidenceAgent
    from agents.explainability_agent import ExplainabilityAgent
    from agents.ingestion_curation_agent import IngestionCurationAgent
    from agents.reasoning_agent import ReasoningAgent
    from repositories.evidence_object_repository import EvidenceObjectRepository

    connection = psycopg2.connect(DATABASE_URL)
    evidence_repository = EvidenceObjectRepository(connection)

    controller = WorkflowController(
        ingestion_agent=IngestionCurationAgent(),
        evidence_agent=EvidenceAgent(evidence_repository=evidence_repository),
        reasoning_agent=ReasoningAgent(evidence_repository=evidence_repository),
        explainability_agent=ExplainabilityAgent(pg_repo=evidence_repository),
        evidence_repository=evidence_repository,
        gap_rules=GapRules(),
    )

    worker = Worker(controller=controller, connection=connection)
    logger.info("orchestration.worker starting, polling every %.1fs", POLL_INTERVAL_SECONDS)
    worker.run_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
