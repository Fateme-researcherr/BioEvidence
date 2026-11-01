CREATE TABLE api_key (
    key_id UUID PRIMARY KEY,
    key_hash TEXT NOT NULL UNIQUE,
    owner_label TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at TIMESTAMPTZ
);
