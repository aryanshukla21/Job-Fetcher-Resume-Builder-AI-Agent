"""
Node: ats_scorer
Executes hybrid keyword and semantic scoring against the current draft.
"""
from __future__ import annotations

import json
from src.schemas.state import AgentState
from src.schemas.models import ATSScoreBreakdown
from src.services.ats_engine import evaluate_ats_compliance, ats_evaluation_chain

def ats_scorer(state: AgentState) -> dict:
    """
    Pulls the latest draft, evaluates it against the JD, and updates the score.
    """
    if not state.get("draft_pool"):
        raise ValueError("ats_scorer triggered with an empty draft_pool.")
        
    current_job_id = state["current_job_id"]
    job = next((j for j in state["jobs"] if j.job_id == current_job_id), None)
    if not job:
        raise ValueError(f"Job {current_job_id} not found in state.")

    # Operate on the most recently generated draft
    latest_draft = state["draft_pool"][-1]
    
    # Convert structured draft back to raw text for deterministic evaluation
    draft_text = json.dumps(latest_draft.content.model_dump(), indent=2)
    
    # Extract JD keywords compiled by context_assembler
    assembler_output = state.get("_assembler_output")
    jd_keywords = assembler_output.jd_keywords if assembler_output else []

    # 1. Run deterministic token match
    hybrid_score, matched_kws = evaluate_ats_compliance(draft_text, job.jd_text, jd_keywords)
    
    # 2. Extract full breakdown via Tier 1 LLM
    try:
        breakdown: ATSScoreBreakdown = ats_evaluation_chain.invoke({
            "jd": job.jd_text,
            "resume": draft_text
        })
        # Override LLM keyword math with strict deterministic arrays
        breakdown.matched_keywords = matched_kws
        breakdown.missing_keywords = [kw for kw in jd_keywords if kw not in matched_kws]
        
    except Exception as e:
        print(f"[WARN] ATS Breakdown LLM failed: {e}")
        breakdown = ATSScoreBreakdown(
            keyword_score=0.0, section_score=0.0, format_score=0.0, density_score=0.0,
            total_score=0.0, matched_keywords=matched_kws, missing_keywords=[],
            missing_sections=[], format_violations=[], word_count=len(draft_text.split())
        )

    # Mutate draft metrics
    latest_draft.ats_score = hybrid_score / 100.0  # Normalize to 0.0 - 1.0
    latest_draft.ats_breakdown = breakdown
    latest_draft.keyword_gap = breakdown.missing_keywords

    # Returning array triggers drafts_reducer to upsert the draft in place by draft_id
    return {"draft_pool": [latest_draft]}