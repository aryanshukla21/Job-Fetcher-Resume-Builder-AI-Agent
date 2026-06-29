"""
Node: archive_job
Transitions a job to the rejected state and cleans up the routing decision.
"""
from __future__ import annotations

from src.schemas.state import AgentState
from src.schemas.models import JobStatus

def archive_job(state: AgentState) -> dict:
    """
    Reads the highest-relevance pending job, marks it as rejected,
    and clears the user decision.
    """
    jobs = state.get("jobs", [])
    
    pending_jobs = [j for j in jobs if j.status == JobStatus.PENDING]
    
    if not pending_jobs:
        return {} # Safe exit if graph was triggered erroneously

    target_job = pending_jobs[0]
    target_job.status = JobStatus.REJECTED

    return {
        # Update job status (upserted via jobs_append_reducer)
        "jobs": [target_job],
        
        # Clear the HITL decision to prevent routing loops
        "user_decision": None
    }