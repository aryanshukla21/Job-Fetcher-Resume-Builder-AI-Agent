"""
Node: job_fetcher
Fetches new job postings via Apify and appends them to the graph state.
"""
from __future__ import annotations

from src.schemas.state import AgentState
from src.tools.apify_tools import fetch_jobs_from_apify

def job_fetcher(state: AgentState) -> dict:
    """
    Constructs search queries from user_config and executes Apify actors.
    Returns appended jobs to trigger jobs_append_reducer.
    """
    config = state["user_config"]
    
    # Constructing deterministic queries for Apify actor
    queries = [f"{role} {config.location}" for role in config.target_roles]
    
    run_input = {
        "queries": queries,
        "maxItemsPerQuery": 10,
        "country": "US" if "us" in config.location.lower() else "IN",
        "hasSalary": True if config.salary_min > 0 else False
    }
    
    actor_id = "cjl8jswvzxfhzl18o" 
    
    try:
        new_jobs = fetch_jobs_from_apify(actor_id, run_input)
    except Exception as e:
        print(f"[ERROR] Job fetcher failed: {e}")
        new_jobs = []

    # Apply hard constraints before jobs enter the LLM scoring queue
    filtered_jobs = []
    for job in new_jobs:
        if config.salary_min and job.salary_min_usd and job.salary_min_usd < config.salary_min:
            continue
        if config.work_type and job.work_type and job.work_type != config.work_type:
            continue
        filtered_jobs.append(job)

    # Returning 'jobs' array triggers jobs_append_reducer (upserts by URL)
    return {"jobs": filtered_jobs}