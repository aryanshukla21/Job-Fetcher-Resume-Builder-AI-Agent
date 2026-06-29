CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY,
    github_username VARCHAR(255) NOT NULL,
    base_resume_path TEXT NOT NULL, 
    target_roles JSONB NOT NULL DEFAULT '[]'::jsonb,
    manual_skills JSONB NOT NULL DEFAULT '[]'::jsonb,
    salary_min_usd INTEGER NOT NULL,
    salary_max_usd INTEGER,
    work_type VARCHAR(50) NOT NULL,
    location VARCHAR(255) NOT NULL,
    max_iterations INTEGER DEFAULT 5,
    ats_threshold NUMERIC(3, 2) DEFAULT 0.80,
    platforms JSONB NOT NULL DEFAULT '["linkedin", "indeed"]'::jsonb,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
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

-- Optimized for the HITL queue: filtering pending jobs by specific user
CREATE INDEX IF NOT EXISTS idx_jobs_user_status ON jobs(user_id, status);

-- Optimized for relevance sorting within a user's queue
CREATE INDEX IF NOT EXISTS idx_jobs_user_relevance ON jobs(user_id, relevance_score DESC);

-- GIN index allows fast element-level querying inside the JSONB keyword array
CREATE INDEX IF NOT EXISTS idx_jobs_keywords ON jobs USING GIN (keyword_matches);