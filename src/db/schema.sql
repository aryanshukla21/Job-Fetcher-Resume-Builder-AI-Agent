CREATE TABLE IF NOT EXISTS jobs (
    job_id UUID PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    company VARCHAR(255) NOT NULL,
    location VARCHAR(255),
    salary_raw TEXT,
    salary_min_usd INTEGER,
    work_type VARCHAR(50),
    jd_text TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    source_platform VARCHAR(100) NOT NULL,
    relevance_score NUMERIC(4, 2) DEFAULT 0.0,
    keyword_matches JSONB DEFAULT '[]'::jsonb,
    status VARCHAR(50) DEFAULT 'pending',
    fetched_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_relevance ON jobs(relevance_score DESC);
-- GIN index allows fast querying inside the JSONB array if needed later
CREATE INDEX IF NOT EXISTS idx_jobs_keywords ON jobs USING GIN (keyword_matches);