CREATE TABLE job (
    job_id UUID PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','running','complete','failed')),
    params JSONB NOT NULL,
    result JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Moved from 0005_add_indexes.sql: this index requires the job table,
-- which does not exist until this migration.
CREATE INDEX idx_job_status ON job(status);
