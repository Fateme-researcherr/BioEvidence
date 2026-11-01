ALTER TABLE evidence_object
    ADD CONSTRAINT chk_confidence_components_shape
    CHECK (
        confidence_components ? 'evidence_type_weight' AND
        confidence_components ? 'study_design_weight' AND
        confidence_components ? 'replication_count' AND
        confidence_components ? 'source_authority_weight' AND
        confidence_components ? 'consistency_conflict_adjustment' AND
        confidence_components ? 'extraction_confidence'
    );
