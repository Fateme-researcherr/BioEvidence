"""
Evidence Object schema and validators (Section 8, Section 20.1, Section 21.2).

Evidence Objects are independent, first-class entities — not annotations
on graph edges, and not owned by any single graph (Correction 1).

Validation gates enforced here:
- `source_database` must be one of the ten primary scientific sources
  (Section 8.3) — libraries/wrappers (Biopython, pyobo, bioregistry,
  mygene.py, sentence-transformers, LangChain, etc.) are never a valid
  `source_database` value and are rejected at write time.
- `confidence_components` must be fully populated (Section 24) — an
  Evidence Object missing any component is rejected, not defaulted
  (Section 21.2).
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Section 8.3 — the only valid origins of Evidence Objects.
PRIMARY_SOURCE_DATABASES = {
    "PubMed",
    "EuropePMC",
    "ClinicalTrialsGov",
    "Orphanet",
    "MONDO",
    "HPO",
    "NCBIGene",
    "UniProt",
    "Reactome",
    "OpenTargets",
    "ChEMBL",
}

# Section 8.3 — libraries/wrappers, implementation accelerators only,
# never valid `source_database` values. Checked explicitly so a
# library name produces a clear rejection reason rather than a generic
# "not in enum" error.
LIBRARY_WRAPPER_NAMES = {
    "Biopython",
    "pyobo",
    "bioregistry",
    "mygene.py",
    "sentence-transformers",
    "LangChain",
    "LlamaIndex",
}


class SourceDatabase(str, Enum):
    PUBMED = "PubMed"
    EUROPE_PMC = "EuropePMC"
    CLINICAL_TRIALS_GOV = "ClinicalTrialsGov"
    ORPHANET = "Orphanet"
    MONDO = "MONDO"
    HPO = "HPO"
    NCBI_GENE = "NCBIGene"
    UNIPROT = "UniProt"
    REACTOME = "Reactome"
    OPEN_TARGETS = "OpenTargets"
    CHEMBL = "ChEMBL"


class SourceAuthorityLevel(str, Enum):
    CURATED_DATABASE = "curated_database"
    PEER_REVIEWED_PRIMARY = "peer_reviewed_primary"
    PEER_REVIEWED_SECONDARY = "peer_reviewed_secondary"
    REGISTERED_TRIAL_RECORD = "registered_trial_record"
    PREPRINT = "preprint"
    DATABASE_AGGREGATED_SCORE = "database_aggregated_score"


class PrimaryOrSecondary(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"


class EvidenceType(str, Enum):
    GENETIC_ASSOCIATION = "genetic_association"
    FUNCTIONAL_MECHANISTIC = "functional_mechanistic"
    CLINICAL_TRIAL_RESULT = "clinical_trial_result"
    CASE_REPORT = "case_report"
    TARGET_DISEASE_ASSOCIATION_SCORE = "target_disease_association_score"
    REVIEW_META_ANALYSIS = "review_meta_analysis"


class StudyDesign(str, Enum):
    RCT = "RCT"
    OBSERVATIONAL = "observational"
    PRECLINICAL_IN_VITRO = "preclinical_in_vitro"
    PRECLINICAL_IN_VIVO = "preclinical_in_vivo"
    CASE_SERIES = "case_series"
    NOT_APPLICABLE = "not_applicable"


class SupportsOrConflicts(str, Enum):
    SUPPORTS = "supports"
    CONFLICTS = "conflicts"
    NEUTRAL = "neutral"


class TargetType(str, Enum):
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    HYPOTHESIS = "hypothesis"
    EXPERIMENT = "experiment"
    QUESTION = "question"
    CLAIM = "claim"


class EvidenceObjectValidationError(ValueError):
    """Raised when an Evidence Object fails a validation gate and must be rejected."""


class ConfidenceComponents(BaseModel):
    """
    Section 24 — six components, only one (`extraction_confidence`) is
    LLM-derived. All six are required; none may be defaulted.
    """

    evidence_type_weight: float
    study_design_weight: float
    replication_count: int
    source_authority_weight: float
    consistency_conflict_adjustment: float
    extraction_confidence: float  # the ONLY LLM-derived component

    @field_validator(
        "evidence_type_weight",
        "study_design_weight",
        "source_authority_weight",
        "extraction_confidence",
    )
    @classmethod
    def _weight_in_unit_range(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise EvidenceObjectValidationError(
                f"confidence_components weight must be in [0.0, 1.0], got {value}"
            )
        return value

    @field_validator("replication_count")
    @classmethod
    def _replication_count_non_negative(cls, value: int) -> int:
        if value < 0:
            raise EvidenceObjectValidationError("replication_count must be >= 0")
        return value


class EvidenceObject(BaseModel):
    """
    Section 8.1 schema. `confidence_score` and `source_authority_level` are
    never supplied directly by extraction logic — they are computed
    downstream from configuration and metadata (Section 23 LLM
    responsibility boundary); this schema still declares them as required
    fields on the persisted object, since by the time an EvidenceObject is
    constructed for storage those values must already be resolved.
    """

    evidence_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    claim_summary: str

    source_database: SourceDatabase
    source_id: str
    retrieval_method: str
    source_authority_level: SourceAuthorityLevel
    primary_or_secondary: PrimaryOrSecondary

    evidence_type: EvidenceType
    study_design: StudyDesign

    supports_or_conflicts: SupportsOrConflicts
    target_type: TargetType
    target_id: uuid.UUID

    confidence_score: float
    confidence_components: ConfidenceComponents

    retrieved_at: datetime = Field(default_factory=datetime.utcnow)
    claim_asserted_date: Optional[datetime] = None

    @field_validator("source_database", mode="before")
    @classmethod
    def _reject_library_wrapper_names(cls, value):
        """
        Section 8.3 enforcement: a record whose `source_database` would
        resolve to a library or repository name rather than a primary
        source is rejected at write time — this is a validation gate in
        the write path, not a documentation convention.
        """
        if isinstance(value, str) and value in LIBRARY_WRAPPER_NAMES:
            raise EvidenceObjectValidationError(
                f"'{value}' is a library/wrapper, not a primary source "
                f"(Section 8.3) — rejected as source_database"
            )
        return value

    @field_validator("claim_summary")
    @classmethod
    def _claim_summary_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise EvidenceObjectValidationError("claim_summary must be non-empty")
        return value

    @field_validator("confidence_score")
    @classmethod
    def _confidence_score_in_range(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise EvidenceObjectValidationError(
                f"confidence_score must be in [0.0, 1.0], got {value}"
            )
        return value

    @model_validator(mode="after")
    def _confidence_components_fully_populated(self) -> "EvidenceObject":
        """
        Section 21.2 validation gate: `confidence_components` must be
        fully populated — an Evidence Object missing any component is
        rejected, not defaulted. Pydantic's required-field enforcement on
        ConfidenceComponents already guarantees presence; this re-checks
        explicitly so the gate is visible and enforced independent of
        model-definition changes.
        """
        required_fields = (
            "evidence_type_weight",
            "study_design_weight",
            "replication_count",
            "source_authority_weight",
            "consistency_conflict_adjustment",
            "extraction_confidence",
        )
        for field_name in required_fields:
            if getattr(self.confidence_components, field_name, None) is None:
                raise EvidenceObjectValidationError(
                    f"confidence_components.{field_name} is missing — "
                    f"Evidence Object rejected, not defaulted"
                )
        return self


def validate_source_database_enum(source_database: str) -> None:
    """
    Standalone enum gate (Section 8.3 / Section 29 unit test:
    'source validation (source_database enum gate)').

    Raises EvidenceObjectValidationError if `source_database` is not one
    of the ten primary scientific sources.
    """
    if source_database in LIBRARY_WRAPPER_NAMES:
        raise EvidenceObjectValidationError(
            f"'{source_database}' is a library/wrapper, not a primary source "
            f"(Section 8.3) — rejected as source_database"
        )
    if source_database not in PRIMARY_SOURCE_DATABASES:
        raise EvidenceObjectValidationError(
            f"'{source_database}' is not one of the ten primary scientific "
            f"sources (Section 8.3): {sorted(PRIMARY_SOURCE_DATABASES)}"
        )
