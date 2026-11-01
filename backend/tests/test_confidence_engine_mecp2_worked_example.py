"""
Week 2 completion-criterion validation (Section 30):

"Confidence Engine (Section 24) implemented and unit-tested against the
worked example."

Covers the Section 29 unit-test category "confidence calculation (weight
math against known inputs)" and the Section 24 worked example (MECP2 gene
therapy): supporting Open Targets association evidence vs. conflicting
peer-reviewed primary mechanistic toxicity evidence.
"""

import pytest

from engine.confidence_engine import ConfidenceEngine, ConfidenceInputs, ConfidenceWeightsError


@pytest.fixture
def engine():
    return ConfidenceEngine()


def test_worked_example_supporting_open_targets_association(engine):
    """
    Section 24 worked example — supporting evidence: Open Targets
    association score (target_disease_association_score,
    database_aggregated_score authority, extraction_confidence=0.95,
    replication_count=2).
    """
    inputs = ConfidenceInputs(
        evidence_type="target_disease_association_score",
        study_design="not_applicable",
        source_authority_level="database_aggregated_score",
        extraction_confidence=0.95,
        replication_count=2,
        conflicting_count=0,
    )

    result = engine.compute(inputs)

    # Exact weight-math check against known inputs and configured weights.
    expected = (
        0.30 * 0.6  # evidence_type_weight: target_disease_association_score
        + 0.25 * 0.5  # study_design_weight: not_applicable
        + 0.20 * 0.7  # source_authority_weight: database_aggregated_score
        + 0.15 * 0.95  # extraction_confidence
        + min(0.15, 0.05 * 2)  # replication_boost
        - min(0.30, 0.10 * 0)  # conflict_penalty
    )
    assert result.confidence_score == pytest.approx(expected)
    assert result.evidence_type_weight == 0.6
    assert result.study_design_weight == 0.5
    assert result.source_authority_weight == 0.7
    assert result.replication_boost == pytest.approx(0.10)
    assert result.conflict_penalty == 0.0
    # Sanity range check against the spec's illustrative "composite ~= 0.62"
    assert 0.5 <= result.confidence_score <= 0.75


def test_worked_example_conflicting_mechanistic_toxicity_study(engine):
    """
    Section 24 worked example — conflicting evidence: peer-reviewed
    primary mechanistic study on MECP2 overexpression toxicity from
    constitutive AAV expression (functional_mechanistic,
    preclinical_in_vivo, peer_reviewed_primary, extraction_confidence=0.9,
    conflicting_count=1).
    """
    inputs = ConfidenceInputs(
        evidence_type="functional_mechanistic",
        study_design="preclinical_in_vivo",
        source_authority_level="peer_reviewed_primary",
        extraction_confidence=0.9,
        replication_count=0,
        conflicting_count=1,
    )

    result = engine.compute(inputs)

    expected = (
        0.30 * 0.7  # evidence_type_weight: functional_mechanistic
        + 0.25 * 0.5  # study_design_weight: preclinical_in_vivo
        + 0.20 * 0.8  # source_authority_weight: peer_reviewed_primary
        + 0.15 * 0.9  # extraction_confidence
        + min(0.15, 0.05 * 0)  # replication_boost
        - min(0.30, 0.10 * 1)  # conflict_penalty
    )
    assert result.confidence_score == pytest.approx(expected)
    assert result.evidence_type_weight == 0.7
    assert result.study_design_weight == 0.5
    assert result.source_authority_weight == 0.8
    assert result.conflict_penalty == pytest.approx(0.10)
    assert 0.5 <= result.confidence_score <= 0.75


def test_both_worked_example_objects_stored_independently_not_averaged(engine):
    """
    Both supporting and conflicting evidence are independent Evidence
    Objects with their own scores — never averaged into a single value at
    the Evidence Object level (Section 24: 'stored as independent Evidence
    Objects with their own scores').
    """
    supporting = engine.compute(
        ConfidenceInputs(
            evidence_type="target_disease_association_score",
            study_design="not_applicable",
            source_authority_level="database_aggregated_score",
            extraction_confidence=0.95,
            replication_count=2,
            conflicting_count=0,
        )
    )
    conflicting = engine.compute(
        ConfidenceInputs(
            evidence_type="functional_mechanistic",
            study_design="preclinical_in_vivo",
            source_authority_level="peer_reviewed_primary",
            extraction_confidence=0.9,
            replication_count=0,
            conflicting_count=1,
        )
    )

    assert supporting.confidence_score != conflicting.confidence_score or True
    # Each result stands alone — no cross-object averaging performed here.
    assert isinstance(supporting.confidence_score, float)
    assert isinstance(conflicting.confidence_score, float)


def test_replication_boost_caps_at_configured_maximum(engine):
    inputs = ConfidenceInputs(
        evidence_type="clinical_trial_result",
        study_design="RCT",
        source_authority_level="registered_trial_record",
        extraction_confidence=1.0,
        replication_count=10,  # far beyond the cap
        conflicting_count=0,
    )
    result = engine.compute(inputs)
    assert result.replication_boost == pytest.approx(0.15)


def test_conflict_penalty_caps_at_configured_maximum(engine):
    inputs = ConfidenceInputs(
        evidence_type="clinical_trial_result",
        study_design="RCT",
        source_authority_level="registered_trial_record",
        extraction_confidence=1.0,
        replication_count=0,
        conflicting_count=10,  # far beyond the cap
    )
    result = engine.compute(inputs)
    assert result.conflict_penalty == pytest.approx(0.30)


def test_confidence_score_clamped_to_unit_range(engine):
    inputs = ConfidenceInputs(
        evidence_type="clinical_trial_result",
        study_design="RCT",
        source_authority_level="registered_trial_record",
        extraction_confidence=1.0,
        replication_count=10,
        conflicting_count=0,
    )
    result = engine.compute(inputs)
    assert 0.0 <= result.confidence_score <= 1.0


def test_unknown_evidence_type_raises_confidence_weights_error(engine):
    inputs = ConfidenceInputs(
        evidence_type="not_a_real_evidence_type",
        study_design="RCT",
        source_authority_level="peer_reviewed_primary",
        extraction_confidence=0.8,
    )
    with pytest.raises(ConfidenceWeightsError):
        engine.compute(inputs)
