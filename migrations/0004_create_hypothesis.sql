CREATE TABLE hypothesis (
    hypothesis_id UUID PRIMARY KEY,
    question_id UUID NOT NULL REFERENCES question(question_id),
    statement TEXT NOT NULL,
    confidence_summary NUMERIC(4,3),
    reframed_from_failure BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
