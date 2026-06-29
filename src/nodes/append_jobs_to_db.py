"""
Node: append_jobs_to_db
Synchronizes the LangGraph state job array with the persistent PostgreSQL database.
"""
from __future__ import annotations

from src.schemas.state import AgentState
from src.db.job_store import upsert_jobs

def append_jobs_to_db(state: AgentState) -> dict:
    """
    Reads the current jobs array from the state and commits them to the database.
    Does not mutate LangGraph state; acts strictly as a side-effect boundary.
    """
    jobs = state.get("jobs", [])
    
    if jobs:
        try:
            upsert_jobs(jobs)
        except Exception as e:
            print(f"[ERROR] Database sync failed: {e}")
            return {"last_error": f"DB Sync Error: {str(e)}"}

    return {}