CREATE TABLE question (
    question_id UUID PRIMARY KEY,
    disease_kg_id TEXT NOT NULL,
    gene_target_kg_id TEXT,
    therapeutic_approach TEXT NOT NULL,
    framework_selected TEXT NOT NULL
        CHECK (framework_selected IN ('PICO','MechanisticBiology','BradfordHill','TargetValidation')),
    framework_rationale TEXT NOT NULL,
    missing_information_flags JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
