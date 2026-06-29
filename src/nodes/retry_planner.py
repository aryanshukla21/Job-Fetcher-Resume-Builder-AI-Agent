"""
Node: retry_planner
Analyzes rejected drafts and user feedback to formulate improvement directives.
"""
from __future__ import annotations

import json
from src.schemas.state import AgentState
from src.schemas.structured_outputs import retry_planner_chain
from src.schemas.models import RetryPlannerOutput

def retry_planner(state: AgentState) -> dict:
    """
    Produces strategic retry context and resets the loop state variables.
    """
    drafts = state.get("top2_drafts")
    if not drafts:
        raise ValueError("retry_planner triggered without top2_drafts.")
        
    job = next(j for j in state["jobs"] if j.job_id == state["current_job_id"])
    
    # Handle edge case where only 1 draft was generated but rejected
    a = drafts[0]
    b = drafts[1] if len(drafts) > 1 else drafts[0]

    try:
        result: RetryPlannerOutput = retry_planner_chain.invoke({
            "score_a": f"{a.ats_score:.0%}",
            "draft_a_json": json.dumps(a.content.model_dump()),
            "breakdown_a": json.dumps(a.ats_breakdown.model_dump()) if a.ats_breakdown else "{}",
            "score_b": f"{b.ats_score:.0%}",
            "draft_b_json": json.dumps(b.content.model_dump()),
            "breakdown_b": json.dumps(b.ats_breakdown.model_dump()) if b.ats_breakdown else "{}",
            "user_feedback": state.get("rejection_feedback") or "No feedback provided.",
            "jd_text": job.jd_text,
        })
    except Exception as e:
        raise RuntimeError(f"Retry planner failed: {e}")

    # Reset the loop environment but retain the retry context
    return {
        "retry_context": result,
        "iteration": 0,         
        "draft_pool": [],       
        "user_decision": None,  
        "top2_drafts": None,
        "rejection_feedback": None
    }