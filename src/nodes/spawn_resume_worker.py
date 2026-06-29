"""
Node: spawn_resume_worker
Transitions a job to the approved state and resets the resume generation loop environment.
"""
from __future__ import annotations

from src.schemas.state import AgentState
from src.schemas.models import JobStatus

def spawn_resume_worker(state: AgentState) -> dict:
    """
    Reads the highest-relevance pending job, marks it as approved, 
    and clears all transient state variables to prepare for a fresh resume loop.
    """
    jobs = state.get("jobs", [])
    
    # Identify the job that the user just approved in HITL (highest scoring pending)
    pending_jobs = [j for j in jobs if j.status == JobStatus.PENDING]
    
    if not pending_jobs:
        raise ValueError("spawn_resume_worker triggered, but no pending jobs exist.")

    target_job = pending_jobs[0]
    target_job.status = JobStatus.APPROVED

    return {
        # Update job status (upserted via jobs_append_reducer)
        "jobs": [target_job],
        
        # Lock the pointer for downstream generation nodes
        "current_job_id": target_job.job_id,
        
        # Reset all loop variables for the new job
        "draft_pool": [],
        "iteration": 0,
        "retry_context": None,
        "top2_drafts": None,
        "approved_draft": None,
        "rejection_feedback": None,
        
        # Clear the HITL decision to prevent routing loops
        "user_decision": None
    }