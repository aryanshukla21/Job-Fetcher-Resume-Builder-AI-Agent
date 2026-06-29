"""
Node: relevance_scorer
Evaluates newly fetched jobs against the user's profile to filter out noise.
"""
from __future__ import annotations

from src.schemas.state import AgentState
from src.schemas.structured_outputs import relevance_scorer_chain
from src.schemas.models import RelevanceScorerOutput

def relevance_scorer(state: AgentState) -> dict:
    """
    Scores pending jobs. Updates relevance_score and keyword_matches.
    """
    ctx = state["verified_context"]
    updated_jobs = []
    
    # We only want to score jobs that haven't been evaluated yet
    jobs_to_score = [j for j in state["jobs"] if j.relevance_score == 0.0 and j.status == "pending"]
    
    if not jobs_to_score:
        return {} # No state mutation needed

    target_roles_str = ", ".join(ctx.user_config.target_roles)
    skills_str = ", ".join(ctx.skills)

    for job in jobs_to_score:
        try:
            result: RelevanceScorerOutput = relevance_scorer_chain.invoke({
                "target_roles": target_roles_str,
                "skills": skills_str,
                "salary_min": ctx.user_config.salary_min,
                "work_type": ctx.user_config.work_type.value,
                "jd_text": job.jd_text,
            })
            
            # Mutate job record safely
            job.relevance_score = result.score
            job.keyword_matches = result.matched_keywords
            if result.hard_fail:
                job.status = "rejected"
                
            updated_jobs.append(job)
        except Exception as e:
            print(f"[ERROR] Relevance scorer failed for job {job.job_id}: {e}")
            job.status = "rejected" # Fail closed on scoring errors
            updated_jobs.append(job)

    # Trigger jobs_append_reducer to upsert the scored jobs
    return {"jobs": updated_jobs}