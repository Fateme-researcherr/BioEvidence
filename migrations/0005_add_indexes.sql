CREATE INDEX idx_evidence_object_source ON evidence_object(source_database, source_id);
CREATE INDEX idx_evidence_link_target ON evidence_link(target_type, target_id);
CREATE INDEX idx_evidence_link_evidence ON evidence_link(evidence_id);
CREATE INDEX idx_hypothesis_question ON hypothesis(question_id);
CREATE INDEX idx_question_disease ON question(disease_kg_id);
-- idx_job_status moved to 0007_create_job_table.sql: the `job` table does
-- not exist until that migration, so creating this index here would fail.
