"""
Node: resume_hitl
Entry point for resume approval queue. Active Webhook push.
"""
from __future__ import annotations

import os
import httpx
from src.schemas.state import AgentState

def resume_hitl(state: AgentState) -> dict:
    """
    Pushes the top 2 generated drafts for human review.
    """
    top2 = state.get("top2_drafts", [])
    job_id = state.get("current_job_id")
    
    webhook_url = os.getenv("FRESHCHAT_API_URL")
    if webhook_url:
        try:
            httpx.post(
                webhook_url, 
                json={
                    "message": f"Resume Drafts Ready for Job ID {job_id}. Please review.",
                    "drafts": [{"id": d.draft_id, "score": d.ats_score} for d in top2]
                },
                timeout=3.0
            )
        except Exception as e:
            print(f"[ERROR] Freshchat webhook failed: {e}")
            
    return {}