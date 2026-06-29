"""
Node: resume_drafter
Generates the structured Pydantic resume draft ensuring zero hallucination.
"""
from __future__ import annotations

import json
from uuid import uuid4
from src.schemas.state import AgentState
from src.schemas.structured_outputs import resume_drafter_chain
from src.schemas.models import ResumeDraft

def resume_drafter(state: AgentState) -> dict:
    """
    Generates the resume draft payload. Increments iteration counter.
    """
    ctx = state["verified_context"]
    job = next(j for j in state["jobs"] if j.job_id == state["current_job_id"])
    asm_out = state.get("_assembler_output")
    iteration = state.get("iteration", 0) + 1

    if not asm_out:
        raise ValueError("resume_drafter triggered without _assembler_output transient state.")

    try:
        result = resume_drafter_chain.invoke({
            "skills": ", ".join(ctx.skills),
            "experience_json": json.dumps([e.model_dump() for e in ctx.experience]),
            "projects_json": json.dumps([p.model_dump() for p in ctx.projects]),
            "education_json": json.dumps([e.model_dump() for e in ctx.education]),
            "achievements": "\n".join(f"- {a}" for a in ctx.achievements) or "none",
            "job_title": job.title,
            "job_company": job.company,
            "jd_keywords": ", ".join(asm_out.jd_keywords),
            "keyword_gap": ", ".join(asm_out.keyword_gap),
            "iteration": iteration,
            "iteration_focus": asm_out.iteration_focus,
            "recommended_projects": ", ".join(asm_out.recommended_projects),
            "priority_skills": ", ".join(asm_out.priority_skills),
        })
    except Exception as e:
        raise RuntimeError(f"Resume drafting failed: {e}")

    draft = ResumeDraft(
        draft_id=str(uuid4()),
        job_id=job.job_id,
        iteration=iteration,
        content=result,
        # ats_score and breakdown will be hydrated by the ats_scorer node next
    )

    return {
        "draft_pool": [draft],
        "iteration": iteration,
        "_assembler_output": None # Clear transient state
    }