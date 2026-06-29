"""
PostgreSQL interface for the jobs database.
Handles translation between JobRecord Pydantic models and PostgreSQL rows.
"""
from __future__ import annotations

import json
from pathlib import Path

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

from src.config import settings
from src.schemas.models import JobRecord, JobStatus, WorkType

# Global pool instance. 
# In a distributed deployment, initialize this on app startup/worker boot.
db_pool = ConnectionPool(
    conninfo=settings.database_url,
    min_size=2,
    max_size=20,
    kwargs={"row_factory": dict_row}
)

def init_db() -> None:
    """Initializes the database schema."""
    schema_path = Path(__file__).parent / "schema.sql"
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(schema_path.read_text(encoding="utf-8"))
        conn.commit()

def upsert_jobs(jobs: list[JobRecord]) -> None:
    """
    Inserts new jobs or updates existing ones. 
    Maintains the status of jobs already in the database.
    """
    if not jobs:
        return

    query = """
        INSERT INTO jobs (
            job_id, title, company, location, salary_raw, salary_min_usd, 
            work_type, jd_text, url, source_platform, relevance_score, 
            keyword_matches, status, fetched_at
        ) VALUES (
            %(job_id)s, %(title)s, %(company)s, %(location)s, %(salary_raw)s, 
            %(salary_min_usd)s, %(work_type)s, %(jd_text)s, %(url)s, 
            %(source_platform)s, %(relevance_score)s, %(keyword_matches)s, 
            %(status)s, %(fetched_at)s
        )
        ON CONFLICT(url) DO UPDATE SET
            relevance_score = EXCLUDED.relevance_score,
            keyword_matches = EXCLUDED.keyword_matches,
            status = EXCLUDED.status;
    """

    data = [
        {
            "job_id": job.job_id,
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "salary_raw": job.salary_raw,
            "salary_min_usd": job.salary_min_usd,
            "work_type": job.work_type.value if job.work_type else None,
            "jd_text": job.jd_text,
            "url": job.url,
            "source_platform": job.source_platform,
            "relevance_score": job.relevance_score,
            "keyword_matches": json.dumps(job.keyword_matches),
            "status": job.status.value,
            "fetched_at": job.fetched_at
        }
        for job in jobs
    ]

    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(query, data)
        conn.commit()

def get_pending_jobs(min_score: float = 3.0) -> list[JobRecord]:
    """Retrieves pending jobs above a certain relevance score for HITL review."""
    query = """
        SELECT * FROM jobs 
        WHERE status = 'pending' AND relevance_score >= %s
        ORDER BY relevance_score DESC
    """
    
    with db_pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (min_score,))
            rows = cur.fetchall()

    jobs = []
    for row in rows:
        jobs.append(
            JobRecord(
                job_id=str(row["job_id"]),
                title=row["title"],
                company=row["company"],
                location=row["location"],
                salary_raw=row["salary_raw"],
                salary_min_usd=row["salary_min_usd"],
                work_type=WorkType(row["work_type"]) if row["work_type"] else None,
                jd_text=row["jd_text"],
                url=row["url"],
                source_platform=row["source_platform"],
                relevance_score=float(row["relevance_score"]),
                keyword_matches=row["keyword_matches"] if row["keyword_matches"] else [],
                status=JobStatus(row["status"]),
                fetched_at=row["fetched_at"]
            )
        )
    return jobs