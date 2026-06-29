"""
Apify scraping integration for job fetching.
Defensively parses actor payloads and maps them to clean JobRecord schemas.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from apify_client import ApifyClient
from src.config import settings
from src.schemas.models import JobRecord, JobStatus, WorkType

def _normalize_work_type(raw_type: str | None) -> WorkType | None:
    """Safely maps arbitrary scraped text to standard WorkType enums."""
    if not raw_type:
        return None
    val = raw_type.lower()
    if "remote" in val or "wfh" in val:
        return WorkType.REMOTE
    if "hybrid" in val:
        return WorkType.HYBRID
    if "onsite" in val or "on-site" in val or "office" in val:
        return WorkType.ONSITE
    return None

def fetch_jobs_from_apify(actor_id: str, run_input: dict[str, Any]) -> list[JobRecord]:
    """
    Executes an Apify actor and extracts job listings.
    Guarantees structural conformity by executing hard validation boundaries.
    """
    client = ApifyClient(settings.apify_api_token)
    valid_records: list[JobRecord] = []

    try:
        # Run the actor synchronously with a hard timeout protection configured via client
        run = client.actor(actor_id).call(run_input=run_input, timeout_secs=300)
        if not run:
            return []
            
        dataset_items = client.dataset(run["defaultDatasetId"]).list_items().items
    except Exception as e:
        print(f"Apify execution failure for actor {actor_id}: {e}")
        return []

    for item in dataset_items:
        try:
            # Enforce clean target URLs to avoid duplicate tracking anomalies
            url = item.get("url") or item.get("jobUrl")
            if not url:
                continue

            # Target required core business fields with strong structural fallbacks
            title = item.get("title") or item.get("jobTitle")
            company = item.get("company") or item.get("companyName")
            description = item.get("description") or item.get("jobDescription") or item.get("text")

            if not all([title, company, description]):
                continue  # Silently skip items missing foundational domain data

            record = JobRecord(
                job_id=str(uuid.uuid4()),
                title=title.strip(),
                company=company.strip(),
                location=item.get("location", "Unknown").strip(),
                salary_raw=item.get("salary") or item.get("salaryText"),
                salary_min_usd=int(item["salaryMin"]) if item.get("salaryMin") else None,
                work_type=_normalize_work_type(item.get("workType")),
                jd_text=description.strip(),
                url=url.strip(),
                source_platform=item.get("source", "Apify Scraper"),
                relevance_score=0.0,
                keyword_matches=[],
                status=JobStatus.PENDING,
                fetched_at=datetime.now(timezone.utc)
            )
            valid_records.append(record)
            
        except Exception as parse_error:
            # Isolate bad rows so a single malformed payload entry won't ruin the batch
            print(f"Skipping malformed scraped payload item due to parsing error: {parse_error}")
            continue

    return valid_records