CREATE TABLE evidence_link (
    link_id UUID PRIMARY KEY,
    evidence_id UUID NOT NULL REFERENCES evidence_object(evidence_id),
    target_type TEXT NOT NULL CHECK (target_type IN
        ('entity','relationship','hypothesis','experiment','question','claim')),
    target_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- evidence_link is what makes Evidence Objects independent and polymorphic:
-- one Evidence Object can have multiple rows here, pointing at different
-- target types simultaneously. When target_type='relationship', target_id
-- resolves against the stable `id` property required on every Neo4j
-- relationship (Section 31, Patch 9).
