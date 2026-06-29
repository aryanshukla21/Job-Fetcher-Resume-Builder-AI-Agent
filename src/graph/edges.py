"""
Conditional edge routing logic for the StateGraph.
"""
from __future__ import annotations
from src.schemas.state import AgentState

def route_job_decision(state: AgentState) -> str:
    """Routes execution based on the user's HITL decision for a job."""
    decision = state.get("user_decision")
    if decision == "approved":
        return "spawn_resume_worker"
    elif decision == "rejected":
        return "archive_job"
    return "end"

def route_ats(state: AgentState) -> str:
    """Evaluates if the ATS generation subgraph should exit or loop."""
    latest_draft = state["draft_pool"][-1]
    threshold = state["user_config"].ats_threshold
    max_iters = state["user_config"].max_iterations
    iteration = state.get("iteration", 1)
    
    if latest_draft.ats_score >= threshold or iteration >= max_iters:
        return "exit_loop"
    return "context_assembler"

def route_resume_decision(state: AgentState) -> str:
    """Routes execution based on the user's HITL decision for the generated resumes."""
    decision = state.get("user_decision")
    if decision == "approved":
        return "render_resume"
    elif decision == "retry":
        return "retry_planner"
    return "end"