"""
Node: render_resume
Takes the user-approved draft and compiles the final physical documents.
"""
from __future__ import annotations

from src.schemas.state import AgentState
from src.tools.doc_renderer import compile_tailored_resume

def render_resume(state: AgentState) -> dict:
    """
    Renders the .docx file and resets loop state variables.
    """
    if not state.get("approved_draft"):
        raise ValueError("render_resume triggered without an approved_draft.")

    config = state["user_config"]
    job_id = state["current_job_id"]
    approved_draft = state["approved_draft"]

    output_filename = f"outputs/{job_id}_tailored_resume.docx"
    
    try:
        final_path = compile_tailored_resume(
            template_path=config.base_resume_path, 
            output_path=output_filename, 
            modifications=approved_draft.content
        )
    except Exception as e:
        return {"last_error": f"Failed to render document: {e}"}

    # Job is fully processed, mark status transition
    completed_job = next(j for j in state["jobs"] if j.job_id == job_id)
    completed_job.status = "resume_ready"

    return {
        # Update job status in state (jobs_append_reducer upserts)
        "jobs": [completed_job],
        "rendered_outputs": {
            "docx": final_path,
            "job_id": job_id,
            "ats_score": approved_draft.ats_score
        },
        # Nullify pointers to prepare state machine for the next pipeline
        "current_job_id": None,
        "approved_draft": None,
        "top2_drafts": None,
        "user_decision": None
    }