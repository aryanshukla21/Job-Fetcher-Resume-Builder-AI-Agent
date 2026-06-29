"""
Node: store_context
Validates raw context and locks the immutable verified_context state.
"""
from __future__ import annotations

from src.schemas.state import AgentState
from src.schemas.models import VerifiedContext

def store_context(state: AgentState) -> dict:
    """
    Reads unvalidated data from context_builder, projects it into the 
    immutable domain model, and triggers the freeze_reducer.
    """
    # Guard clause: Do not re-execute if the context is already frozen
    if state.get("verified_context") is not None:
        return {}

    config = state["user_config"]
    raw = state.get("_raw_context")
    
    if not raw:
        raise ValueError("store_context triggered without _raw_context payload from context_builder.")
        
    parsed_data = raw["parsed_resume"]
    projects = raw["github_projects"]
    
    # Deduplicate and merge manual skills with extracted skills
    merged_skills = list(set(config.manual_skills + parsed_data.skills))
    
    # Construct the immutable verified context
    context = VerifiedContext(
        skills=merged_skills,
        experience=parsed_data.experience,
        achievements=parsed_data.achievements,
        internships=parsed_data.internships,
        education=parsed_data.education,
        projects=projects,
        user_config=config
    )
    
    return {
        "verified_context": context,
        "_raw_context": None # Purge transient state to prevent checkpointer bloat
    }