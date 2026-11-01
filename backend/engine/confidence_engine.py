"""
Confidence Engine (Section 24).

Confidence is a composite score, never a single LLM output (Correction 4).
It combines six components, only one of which touches the LLM
(`extraction_confidence`).

Configurability scope (Patch 2, final): per-category weight tables
(`evidence_type`, `study_design`, `authority`) are configurable via
`configs/confidence_weights.yaml`. The outer combination formula and its
coefficients are fixed constants, versioned with the Confidence Engine's
code, not exposed as configuration.
"""

import os
from dataclasses import dataclass

import yaml

DEFAULT_WEIGHTS_PATH = os.environ.get(
    "CONFIDENCE_WEIGHTS_PATH",
    os.path.join(os.path.dirname(__file__), "..", "configs", "confidence_weights.yaml"),
)

# Fixed outer-formula coefficients — versioned with code, never configurable.
EVIDENCE_TYPE_COEFFICIENT = 0.30
STUDY_DESIGN_COEFFICIENT = 0.25
SOURCE_AUTHORITY_COEFFICIENT = 0.20
EXTRACTION_CONFIDENCE_COEFFICIENT = 0.15

REPLICATION_BOOST_PER_UNIT = 0.05
REPLICATION_BOOST_CAP = 0.15

CONFLICT_PENALTY_PER_UNIT = 0.10
CONFLICT_PENALTY_CAP = 0.30


@dataclass
class ConfidenceInputs:
    evidence_type: str
    study_design: str
    source_authority_level: str
    extraction_confidence: float
    replication_count: int = 0
    conflicting_count: int = 0


@dataclass
class ConfidenceResult:
    confidence_score: float
    evidence_type_weight: float
    study_design_weight: float
    source_authority_weight: float
    extraction_confidence: float
    replication_boost: float
    conflict_penalty: float
    consistency_conflict_adjustment: float


class ConfidenceWeightsError(ValueError):
    """Raised when a category/key is missing from the weight configuration."""


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class ConfidenceEngine:
    """
    Computes the composite confidence_score for an Evidence Object,
    per the Section 24 formula:

        confidence_score = clamp(0.0, 1.0,
            0.30 * evidence_type_weight +
            0.25 * study_design_weight +
            0.20 * source_authority_weight +
            0.15 * extraction_confidence +
            replication_boost -
            conflict_penalty
        )
        replication_boost = min(0.15, 0.05 * replication_count)
        conflict_penalty  = min(0.30, 0.10 * conflicting_count)
    """

    def __init__(self, weights_path: str = DEFAULT_WEIGHTS_PATH):
        with open(weights_path, "r", encoding="utf-8") as f:
            self._weights = yaml.safe_load(f)

    def _lookup_weight(self, category: str, key: str) -> float:
        category_table = self._weights.get(category)
        if category_table is None:
            raise ConfidenceWeightsError(f"Unknown weight category: '{category}'")
        weight = category_table.get(key)
        if weight is None:
            raise ConfidenceWeightsError(
                f"No configured weight for {category}.{key} in confidence_weights.yaml"
            )
        return float(weight)

    def compute(self, inputs: ConfidenceInputs) -> ConfidenceResult:
        evidence_type_weight = self._lookup_weight("evidence_type", inputs.evidence_type)
        study_design_weight = self._lookup_weight("study_design", inputs.study_design)
        source_authority_weight = self._lookup_weight(
            "authority", inputs.source_authority_level
        )

        replication_boost = min(
            REPLICATION_BOOST_CAP, REPLICATION_BOOST_PER_UNIT * inputs.replication_count
        )
        conflict_penalty = min(
            CONFLICT_PENALTY_CAP, CONFLICT_PENALTY_PER_UNIT * inputs.conflicting_count
        )
        consistency_conflict_adjustment = replication_boost - conflict_penalty

        raw_score = (
            EVIDENCE_TYPE_COEFFICIENT * evidence_type_weight
            + STUDY_DESIGN_COEFFICIENT * study_design_weight
            + SOURCE_AUTHORITY_COEFFICIENT * source_authority_weight
            + EXTRACTION_CONFIDENCE_COEFFICIENT * inputs.extraction_confidence
            + replication_boost
            - conflict_penalty
        )
        confidence_score = _clamp(raw_score, 0.0, 1.0)

        return ConfidenceResult(
            confidence_score=confidence_score,
            evidence_type_weight=evidence_type_weight,
            study_design_weight=study_design_weight,
            source_authority_weight=source_authority_weight,
            extraction_confidence=inputs.extraction_confidence,
            replication_boost=replication_boost,
            conflict_penalty=conflict_penalty,
            consistency_conflict_adjustment=consistency_conflict_adjustment,
        )
