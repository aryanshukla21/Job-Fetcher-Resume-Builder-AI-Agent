"""
Node: select_top2
Slices the top 2 generated drafts for human review.
"""
from __future__ import annotations
from src.schemas.state import AgentState

def select_top2(state: AgentState) -> dict:
    """Extracts the top 2 highest-scoring drafts from the sorted draft pool."""
    drafts = state.get("draft_pool", [])
    if not drafts:
        raise ValueError("select_top2 triggered with an empty draft pool.")
    
    # drafts_reducer already maintains descending sort by ats_score
    return {"top2_drafts": drafts[:2]}