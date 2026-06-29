"""
Node: context_assembler
Analyzes the JD against the verified profile and plans the drafting strategy.
"""
from __future__ import annotations

from src.schemas.state import AgentState
from src.schemas.structured_outputs import context_assembler_chain
from src.schemas.models import ContextAssemblerOutput

def context_assembler(state: AgentState) -> dict:
    """
    Assembles the keyword gaps, priority skills, and focus instructions 
    for the current generation iteration.
    """
    ctx = state["verified_context"]
    job = next(j for j in state["jobs"] if j.job_id == state["current_job_id"])
    iteration = state.get("iteration", 0)

    # Extract previous draft gaps
    prev_score = 0.0
    prev_missing = []
    if state.get("draft_pool"):
        last_draft = state["draft_pool"][-1]
        prev_score = last_draft.ats_score
        prev_missing = last_draft.keyword_gap

    # Extract retry directives if this loop was triggered by a user rejection
    retry_directives = "none"
    if state.get("retry_context"):
        retry_directives = "\n".join(f"- {d}" for d in state["retry_context"].improvement_directives)

    try:
        result: ContextAssemblerOutput = context_assembler_chain.invoke({
            "jd_text": job.jd_text,
            "skills": ", ".join(ctx.skills),
            "experience_companies": ", ".join(e.company for e in ctx.experience),
            "project_names": ", ".join(p.name for p in ctx.projects),
            "prev_score": f"{prev_score:.0%}" if prev_score else "N/A",
            "prev_missing": ", ".join(prev_missing) if prev_missing else "none",
            "iteration": iteration + 1,
            "retry_directives": retry_directives,
        })
    except Exception as e:
        raise RuntimeError(f"Context assembler failed: {e}")

    # Return as a transient field read exclusively by the resume_drafter node
    return {"_assembler_output": result}