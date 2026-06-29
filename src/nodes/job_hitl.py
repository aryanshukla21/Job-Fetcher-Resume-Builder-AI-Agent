"""
Node: job_hitl
Entry point for job approval queue. Active Webhook push.
"""
from __future__ import annotations

import os
import httpx
from src.schemas.state import AgentState

def job_hitl(state: AgentState) -> dict:
    """
    Pushes the highest-relevance pending job to the user via chat support webhook.
    Executes immediately before LangGraph suspends the thread.
    """
    pending_jobs = [j for j in state.get("jobs", []) if j.status.value == "pending"]
    if not pending_jobs:
        return {}
        
    top_job = pending_jobs[0]
    
    # Active dispatch to unblock HITL polling
    webhook_url = os.getenv("FRESHCHAT_API_URL")
    if webhook_url:
        try:
            httpx.post(
                webhook_url, 
                json={
                    "message": f"Job Review Required: {top_job.title} at {top_job.company}",
                    "job_id": top_job.job_id,
                    "url": top_job.url,
                    "score": top_job.relevance_score
                },
                timeout=3.0
            )
        except Exception as e:
            print(f"[ERROR] Freshchat webhook failed: {e}")
            
    return {}