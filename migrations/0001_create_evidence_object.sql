CREATE TABLE evidence_object (
    evidence_id UUID PRIMARY KEY,
    claim_summary TEXT NOT NULL,
    source_database TEXT NOT NULL
        CHECK (source_database IN (
            'PubMed','EuropePMC','ClinicalTrialsGov','Orphanet',
            'MONDO','HPO','NCBIGene','UniProt','Reactome',
            'OpenTargets','ChEMBL')),
    source_id TEXT NOT NULL,
    retrieval_method TEXT NOT NULL,
    source_authority_level TEXT NOT NULL,
    primary_or_secondary TEXT NOT NULL CHECK (primary_or_secondary IN ('primary','secondary')),
    evidence_type TEXT NOT NULL,
    study_design TEXT NOT NULL
        CHECK (study_design IN ('RCT','observational','preclinical_in_vivo',
            'preclinical_in_vitro','case_series','not_applicable')),
    supports_or_conflicts TEXT NOT NULL CHECK (supports_or_conflicts IN ('supports','conflicts','neutral')),
    confidence_score NUMERIC(4,3) NOT NULL CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    confidence_components JSONB NOT NULL,
    retrieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    claim_asserted_date TIMESTAMPTZ
);
