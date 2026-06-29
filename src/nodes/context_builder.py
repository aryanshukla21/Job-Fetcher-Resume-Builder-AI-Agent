"""
Node: context_builder
Fetches raw data from the base resume and external integrations.
Passes unvalidated data to store_context via transient state.
"""
from __future__ import annotations

from src.schemas.state import AgentState
from src.tools.resume_parser import parse_resume
from src.tools.github_tools import fetch_relevant_github_projects

def context_builder(state: AgentState) -> dict:
    """
    Executes all data-gathering tool calls. Does not validate the final schema.
    """
    config = state["user_config"]
    
    # 1. I/O: Parse base resume
    parsed_data = parse_resume(config.base_resume_path)
    
    # 2. I/O: Fetch GitHub repositories
    projects = fetch_relevant_github_projects(
        username=config.github_username,
        manual_skills=config.manual_skills,
        max_projects=5
    )
    
    # 3. Write to transient state payload
    return {
        "_raw_context": {
            "parsed_resume": parsed_data,
            "github_projects": projects
        }
    }