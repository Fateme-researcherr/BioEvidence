"""
Evidence Agent v0 (Section 15.1, Section 15.3, Section 21.2).

INPUT:  { targets: [{target_type, target_id}], query_context: {disease, gene, therapy} }
OUTPUT: { evidence_objects: [EvidenceObject] }

Data sources (Section 15.3): PubMed/NCBI, Europe PMC, ClinicalTrials.gov,
Open Targets, ChEMBL.

This module is pure orchestration glue over the already-existing pieces —
no new components are introduced and no existing interface is changed:

- `clients.ncbi_client.PubMedClient`, `clients.europepmc_client.EuropePMCClient`,
  `clients.ctgov_client.ClinicalTrialsGovClient`,
  `clients.opentargets_client.OpenTargetsClient`,
  `clients.chembl_client.ChEMBLClient`             — retrieval (Section 22)
- `pipeline.claim_extraction_pipeline.ClaimExtractionPipeline` — claim
  extraction (Section 23)
- `engine.confidence_engine.ConfidenceEngine`      — confidence calculation
  (Section 24)
- `schemas.evidence_object_schema.EvidenceObject`  — schema/validation
  gates (Section 8.1 / Section 21.2)
- `repositories.evidence_object_repository.EvidenceObjectRepository` —
  storage (Section 20.1)

Validation gate (Section 21.2): `source_database` enum check is enforced
by `EvidenceObject`'s pydantic validation; `confidence_components` must be
fully populated — guaranteed here because `ConfidenceEngine.compute()`
never returns a partial result, and the schema re-checks completeness
independently (an incomplete or invalid Evidence Object is rejected, not
defaulted — the record is simply dropped and logged).

Error handling (Section 21.2): each source client call is wrapped
individually — a failure logs and continues rather than aborting the
whole evidence-gathering pass. Extraction failures already retry once
with a schema-repair prompt inside `ClaimExtractionPipeline`; when that is
exhausted (`ClaimExtractionDropped`) the record is dropped with a logged
reason here, not defaulted.

LLM responsibility boundary (Section 23, restated): `source_authority_level`
is never LLM-derived — it is computed here, downstream, from
`source_database` metadata (and, for literature sources, from the
extracted `evidence_type`), never from the claim-extraction LLM call.
"""

import logging
from typing import List, Optional

from clients.chembl_client import ChEMBLClient
from clients.ctgov_client import ClinicalTrialsGovClient
from clients.europepmc_client import EuropePMCClient
from clients.ncbi_client import PubMedClient
from clients.opentargets_client import OpenTargetsClient
from engine.confidence_engine import ConfidenceEngine, ConfidenceInputs
from pipeline.claim_extraction_pipeline import ClaimExtractionDropped, ClaimExtractionPipeline
from repositories.evidence_object_repository import EvidenceLinkTarget, EvidenceObjectRepository
from schemas.evidence_object_schema import EvidenceObject, EvidenceObjectValidationError

logger = logging.getLogger("evidence_agent")

# Source authority level is computed downstream from source metadata
# (Section 23: "computed downstream from configuration and metadata,
# never from the LLM call") — never LLM-derived. Defaults per
# source_database; literature sources (PubMed/EuropePMC) are refined
# per-record below using the extracted `evidence_type`
# (review_meta_analysis -> secondary tier).
SOURCE_AUTHORITY_BY_DATABASE = {
    "ClinicalTrialsGov": "registered_trial_record",
    "OpenTargets": "database_aggregated_score",
    "ChEMBL": "curated_database",
    "PubMed": "peer_reviewed_primary",
    "EuropePMC": "peer_reviewed_primary",
}

LITERATURE_SOURCES = ("PubMed", "EuropePMC")


def _primary_or_secondary(evidence_type: str) -> str:
    return "secondary" if evidence_type == "review_meta_analysis" else "primary"


def _authority_level(source_database: str, evidence_type: str) -> str:
    if source_database in LITERATURE_SOURCES and evidence_type == "review_meta_analysis":
        return "peer_reviewed_secondary"
    return SOURCE_AUTHORITY_BY_DATABASE.get(source_database, "peer_reviewed_secondary")


class EvidenceAgent:
    """
    Evidence Agent (Section 15.3 / Section 21.2): retrieves candidate
    source records for the targets/query context assigned by
    Orchestration, runs each through the existing claim-extraction
    pipeline, computes confidence via the existing Confidence Engine, and
    stores the resulting Evidence Objects via the existing
    EvidenceObjectRepository.
    """

    def __init__(
        self,
        evidence_repository: EvidenceObjectRepository,
        claim_extraction_pipeline: Optional[ClaimExtractionPipeline] = None,
        confidence_engine: Optional[ConfidenceEngine] = None,
        pubmed_client: Optional[PubMedClient] = None,
        europepmc_client: Optional[EuropePMCClient] = None,
        ctgov_client: Optional[ClinicalTrialsGovClient] = None,
        opentargets_client: Optional[OpenTargetsClient] = None,
        chembl_client: Optional[ChEMBLClient] = None,
        max_records_per_source: int = 5,
    ):
        self._evidence_repository = evidence_repository
        self._claim_extraction_pipeline = claim_extraction_pipeline or ClaimExtractionPipeline()
        self._confidence_engine = confidence_engine or ConfidenceEngine()
        self._pubmed_client = pubmed_client or PubMedClient()
        self._europepmc_client = europepmc_client or EuropePMCClient()
        self._ctgov_client = ctgov_client or ClinicalTrialsGovClient()
        self._opentargets_client = opentargets_client or OpenTargetsClient()
        self._chembl_client = chembl_client or ChEMBLClient()
        self._max_records_per_source = max_records_per_source

    def run(self, targets: List[dict], query_context: dict) -> dict:
        """
        Section 21.2 contract: INPUT {targets, query_context} -> OUTPUT
        {evidence_objects}. `targets` assigned by Orchestration based on
        new Ingestion writes or gaps found by Orchestration's
        gap-detection rules (Section 21.5 step 4).
        """
        if not targets:
            return {"evidence_objects": []}

        target_description = self._build_target_description(query_context)
        search_term = self._build_search_term(query_context)

        raw_records = []
        raw_records.extend(self._retrieve_pubmed(search_term))
        raw_records.extend(self._retrieve_europepmc(search_term))
        raw_records.extend(self._retrieve_ctgov(query_context))
        raw_records.extend(self._retrieve_opentargets(query_context))
        raw_records.extend(self._retrieve_chembl(query_context))

        evidence_objects: List[EvidenceObject] = []
        for source_id, raw_text, source_database, retrieval_method in raw_records:
            evidence = self._process_record(
                source_id=source_id,
                raw_text=raw_text,
                source_database=source_database,
                retrieval_method=retrieval_method,
                target_description=target_description,
                targets=targets,
            )
            if evidence is not None:
                evidence_objects.append(evidence)

        return {"evidence_objects": evidence_objects}

    # -- query-context helpers ----------------------------------------------

    def _build_target_description(self, query_context: dict) -> str:
        return (
            f"disease={query_context.get('disease')}, "
            f"gene/target={query_context.get('gene')}, "
            f"therapeutic_approach={query_context.get('therapy')}"
        )

    def _build_search_term(self, query_context: dict) -> str:
        parts = [
            str(value)
            for value in (
                query_context.get("gene"),
                query_context.get("disease"),
                query_context.get("therapy"),
            )
            if value
        ]
        return " ".join(parts)

    # -- per-source retrieval (each wrapped individually, Section 21.2) -----

    def _retrieve_pubmed(self, search_term: str):
        records = []
        if not search_term:
            return records
        try:
            pmids = self._pubmed_client.search(search_term, retmax=self._max_records_per_source)
            fetched = self._pubmed_client.fetch(pmids[: self._max_records_per_source])
            for rec in fetched:
                records.append((rec.identifier, rec.raw_xml, rec.source_database, rec.retrieval_method))
        except Exception as exc:  # per-source failure logs and continues
            logger.warning("PubMed retrieval failed, skipping: %s", exc)
        return records

    def _retrieve_europepmc(self, search_term: str):
        records = []
        if not search_term:
            return records
        try:
            pmcids = self._europepmc_client.search(search_term, page_size=self._max_records_per_source)
            for pmcid in pmcids[: self._max_records_per_source]:
                rec = self._europepmc_client.fetch_full_text(pmcid)
                if rec is not None:
                    records.append((rec.pmcid, rec.raw_content, rec.source_database, rec.retrieval_method))
        except Exception as exc:
            logger.warning("Europe PMC retrieval failed, skipping: %s", exc)
        return records

    def _retrieve_ctgov(self, query_context: dict):
        records = []
        condition = query_context.get("disease")
        intervention = query_context.get("therapy")
        if not condition and not intervention:
            return records
        try:
            studies = self._ctgov_client.search(
                condition=condition,
                intervention=intervention,
                page_size=self._max_records_per_source,
            )
            for study in studies[: self._max_records_per_source]:
                nct_id = (
                    study.get("protocolSection", {})
                    .get("identificationModule", {})
                    .get("nctId")
                )
                if not nct_id:
                    continue
                records.append(
                    (nct_id, str(study), "ClinicalTrialsGov", "ClinicalTrials.gov API v2 /studies")
                )
        except Exception as exc:
            logger.warning("ClinicalTrials.gov retrieval failed, skipping: %s", exc)
        return records

    def _retrieve_opentargets(self, query_context: dict):
        records = []
        ensembl_id = query_context.get("ensembl_id")
        efo_id = query_context.get("efo_id")
        if not ensembl_id or not efo_id:
            return records
        try:
            rec = self._opentargets_client.fetch_target_disease_association(ensembl_id, efo_id)
            if rec.score_available:
                records.append(
                    (f"{ensembl_id}-{efo_id}", rec.raw_content, rec.source_database, rec.retrieval_method)
                )
        except Exception as exc:
            logger.warning("Open Targets retrieval failed, skipping: %s", exc)
        return records

    def _retrieve_chembl(self, query_context: dict):
        records = []
        chembl_target_id = query_context.get("chembl_target_id")
        if not chembl_target_id:
            return records
        try:
            rec = self._chembl_client.fetch_activity(chembl_target_id)
            if rec is not None:
                records.append(
                    (rec.chembl_target_id, rec.raw_content, rec.source_database, rec.retrieval_method)
                )
        except Exception as exc:
            logger.warning("ChEMBL retrieval failed, skipping: %s", exc)
        return records

    # -- extraction + confidence + storage -----------------------------------

    def _process_record(
        self,
        source_id: str,
        raw_text: str,
        source_database: str,
        retrieval_method: str,
        target_description: str,
        targets: List[dict],
    ) -> Optional[EvidenceObject]:
        try:
            extraction = self._claim_extraction_pipeline.extract(
                raw_text=raw_text, target_description=target_description, source_id=source_id
            )
        except ClaimExtractionDropped as exc:
            logger.warning("Dropping extraction for %s: %s", source_id, exc.reason)
            return None

        primary_target = targets[0]
        additional_targets = [
            EvidenceLinkTarget(target_type=t["target_type"], target_id=t["target_id"])
            for t in targets[1:]
        ]

        replication_count = self._evidence_repository.count_supporting(
            primary_target["target_type"], primary_target["target_id"]
        )
        conflicting_count = len(
            self._evidence_repository.find_conflicting(
                primary_target["target_type"], primary_target["target_id"]
            )
        )

        source_authority_level = _authority_level(source_database, extraction.evidence_type)

        confidence_inputs = ConfidenceInputs(
            evidence_type=extraction.evidence_type,
            study_design=extraction.study_design,
            source_authority_level=source_authority_level,
            extraction_confidence=extraction.extraction_confidence,
            replication_count=replication_count,
            conflicting_count=conflicting_count,
        )

        try:
            confidence_result = self._confidence_engine.compute(confidence_inputs)
        except Exception as exc:
            logger.warning("Confidence computation failed for %s, skipping: %s", source_id, exc)
            return None

        try:
            evidence = EvidenceObject(
                claim_summary=extraction.claim_summary,
                source_database=source_database,
                source_id=source_id,
                retrieval_method=retrieval_method,
                source_authority_level=source_authority_level,
                primary_or_secondary=_primary_or_secondary(extraction.evidence_type),
                evidence_type=extraction.evidence_type,
                study_design=extraction.study_design,
                supports_or_conflicts=extraction.supports_or_conflicts,
                target_type=primary_target["target_type"],
                target_id=primary_target["target_id"],
                confidence_score=confidence_result.confidence_score,
                confidence_components={
                    "evidence_type_weight": confidence_result.evidence_type_weight,
                    "study_design_weight": confidence_result.study_design_weight,
                    "replication_count": replication_count,
                    "source_authority_weight": confidence_result.source_authority_weight,
                    "consistency_conflict_adjustment": confidence_result.consistency_conflict_adjustment,
                    "extraction_confidence": confidence_result.extraction_confidence,
                },
            )
        except EvidenceObjectValidationError as exc:
            # Validation gate (Section 21.2): rejected, not defaulted.
            logger.warning("Evidence Object rejected for %s: %s", source_id, exc)
            return None
        except Exception as exc:
            logger.warning("Evidence Object construction failed for %s, skipping: %s", source_id, exc)
            return None

        try:
            self._evidence_repository.store(evidence, additional_targets=additional_targets)
        except Exception as exc:
            logger.warning("Evidence Object store failed for %s, skipping: %s", source_id, exc)
            return None

        return evidence
