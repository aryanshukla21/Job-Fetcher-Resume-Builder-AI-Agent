"""
Shows how each LLM node uses .with_structured_output() to bind a Pydantic
schema to a LangChain model call. This eliminates free-text LLM responses
entirely — every LLM node in the graph produces a validated Pydantic object,
never a raw string.

How it works:
    model.with_structured_output(SchemaClass)
    → LangChain injects the schema as a tool/function call definition
    → The LLM is forced to return JSON matching the schema
    → LangChain parses + validates the JSON into a Pydantic instance
    → If validation fails, LangChain retries automatically (up to N times)

Anti-hallucination role of descriptions:
    Every Field(description=...) you wrote in models.py is sent to the LLM
    as part of the schema definition. The LLM reads "Do not invent percentages
    not in the source" and follows it. The Pydantic validator then catches
    anything that slips through.
"""

from __future__ import annotations

from src.llms.factory import get_llm
from langchain_core.prompts import ChatPromptTemplate

from src.schemas.models import (
    ATSScoreBreakdown,
    ContextAssemblerOutput,
    ResumeDraftContent,
    RelevanceScorerOutput,
    RetryPlannerOutput,
)
from src.schemas.state import AgentState


# ──────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# One model instance per tier — reuse across nodes of the same tier.
# ──────────────────────────────────────────────────────────────

tier1_llm = get_llm(tier=1, temperature=0.0)    # Fast, cheap (Scoring)
tier2_llm = get_llm(tier=2, temperature=0.1)    # Balanced (Context assembly)
tier3_llm = get_llm(tier=3, temperature=0.2)    # Capable (Drafting, Planning)


# ──────────────────────────────────────────────────────────────
# NODE: relevance_scorer
# Model: Haiku (fast + cheap — runs on every job fetched)
# ──────────────────────────────────────────────────────────────

relevance_scorer_chain = (
    ChatPromptTemplate.from_messages([
        ("system", """You are a job relevance classifier. Your ONLY job is to score
how well a job listing matches a candidate's profile.

STRICT RULES:
- Score based ONLY on the information given to you below.
- Do NOT use knowledge of the company or industry beyond what is in the JD text.
- Do NOT infer salary if not stated — leave hard_fail=False for salary if unknown.
- matched_keywords must appear VERBATIM or as clear synonyms in the JD text.
- summary must cite specific JD content — no generic statements."""),

        ("human", """CANDIDATE PROFILE:
Target roles: {target_roles}
Skills: {skills}
Salary minimum: ${salary_min}
Work type: {work_type}

JOB DESCRIPTION:
{jd_text}

Score this job against the candidate profile."""),
    ])
    | tier1_llm.with_structured_output(RelevanceScorerOutput)
)


def relevance_scorer(state: AgentState) -> AgentState:
    """
    Node: relevance_scorer
    Reads:  state.jobs (pending jobs), state.verified_context.user_config
    Writes: state.jobs (updated relevance_score + keyword_matches per job)
    Model:  Claude Haiku
    Output: RelevanceScorerOutput (structured)
    """
    ctx = state["verified_context"]
    updated_jobs = []

    for job in state["jobs"]:
        if job.relevance_score > 0:
            # Already scored — skip
            updated_jobs.append(job)
            continue

        result: RelevanceScorerOutput = relevance_scorer_chain.invoke({
            "target_roles": ", ".join(ctx.user_config.target_roles),
            "skills":       ", ".join(ctx.skills),
            "salary_min":   ctx.user_config.salary_min,
            "work_type":    ctx.user_config.work_type.value,
            "jd_text":      job.jd_text,
        })

        updated_job = job.model_copy(update={
            "relevance_score":  result.score,
            "keyword_matches":  result.matched_keywords,
            "status":           "rejected" if result.hard_fail else job.status,
        })
        updated_jobs.append(updated_job)

    return {"jobs": updated_jobs}


# ──────────────────────────────────────────────────────────────
# NODE: context_assembler
# Model: Sonnet (reasoning about gaps + priorities)
# ──────────────────────────────────────────────────────────────

context_assembler_chain = (
    ChatPromptTemplate.from_messages([
        ("system", """You are a resume strategy assistant. Your job is to analyse
a job description against a candidate's verified profile and produce a structured
context object that will guide resume generation.

CRITICAL RULES — READ BEFORE RESPONDING:
1. recommended_projects must ONLY contain project names that exist in the
   provided projects list. Do not invent project names.
2. recommended_experience must ONLY contain company names from the experience list.
3. jd_keywords must come from the JD text — do not add buzzwords not in the JD.
4. keyword_gap must be keywords IN the JD but NOT in the candidate's skills list.
5. iteration_focus must be specific and actionable — reference exact keywords or sections."""),

        ("human", """JOB DESCRIPTION:
{jd_text}

CANDIDATE VERIFIED PROFILE:
Skills: {skills}
Experience companies: {experience_companies}
Projects: {project_names}

PREVIOUS DRAFT ATS SCORE: {prev_score}
MISSING KEYWORDS FROM LAST DRAFT: {prev_missing}
ITERATION NUMBER: {iteration}
RETRY DIRECTIVES (if any): {retry_directives}

Produce the context assembly for this resume generation iteration."""),
    ])
    | tier2_llm.with_structured_output(ContextAssemblerOutput)
)


def context_assembler(state: AgentState) -> dict:
    """
    Node: context_assembler
    Reads:  state.verified_context, state.current_job_id, state.iteration,
            state.draft_pool (for prev score/gaps), state.retry_context
    Writes: does not write state directly — returns data consumed by resume_drafter
            via the prompt. In practice, store output in a transient state field
            or pass directly to resume_drafter in the same node chain.
    Model:  Claude Sonnet
    Output: ContextAssemblerOutput (structured)
    """
    ctx = state["verified_context"]
    job = next(j for j in state["jobs"] if j.job_id == state["current_job_id"])
    iteration = state.get("iteration", 0)

    # Get previous draft's gap data (if any)
    prev_score = 0.0
    prev_missing: list[str] = []
    if state.get("draft_pool"):
        last_draft = state["draft_pool"][-1]
        prev_score = last_draft.ats_score
        prev_missing = last_draft.keyword_gap

    retry_directives = ""
    if state.get("retry_context"):
        directives = state["retry_context"].improvement_directives
        retry_directives = "\n".join(f"- {d}" for d in directives)

    result: ContextAssemblerOutput = context_assembler_chain.invoke({
        "jd_text":              job.jd_text,
        "skills":               ", ".join(ctx.skills),
        "experience_companies": ", ".join(e.company for e in ctx.experience),
        "project_names":        ", ".join(p.name for p in ctx.projects),
        "prev_score":           f"{prev_score:.0%}" if prev_score else "N/A (first iteration)",
        "prev_missing":         ", ".join(prev_missing) or "none",
        "iteration":            iteration + 1,
        "retry_directives":     retry_directives or "none",
    })

    # Return as transient field for resume_drafter to read
    return {"_assembler_output": result}


# ──────────────────────────────────────────────────────────────
# NODE: resume_drafter
# Model: Opus (highest stakes generation in the pipeline)
# ──────────────────────────────────────────────────────────────

RESUME_DRAFTER_SYSTEM = """You are a professional resume writer. You produce ATS-optimised
resumes from verified candidate data.

═══ ABSOLUTE RULES — VIOLATION CAUSES REJECTION ═══

1. EVERY fact must come from the VERIFIED CONTEXT provided below.
   - Do not invent metrics (e.g. "improved performance by 40%") unless the
     exact metric appears in the experience bullets or project description.
   - Do not add skills not listed in the skills section.
   - Do not mention companies, tools, or projects not in the verified context.

2. source_citations is MANDATORY for every section.
   - Map each section to its verified_context source path.
   - Example: {{"experience[0]": "experience[0]", "projects[0]": "projects[2]"}}

3. Skills section: only use skills from the provided skills list.
   Prioritise skills that appear in the JD keywords list.

4. Summary: must not claim experience not supported by the experience entries.
   Do not use years of experience unless computable from the dates provided.

5. Bullets: preserve quantified metrics verbatim. Do not round, estimate, or embellish.

═══ ATS OPTIMISATION RULES ═══
- Incorporate keywords from keyword_gap naturally into bullets.
- Use exact keyword spelling from the JD (e.g. "machine learning", not "ML").
- Ensure all required sections are present: summary, experience, skills, education.
- Use standard section headings — no creative alternatives."""

resume_drafter_chain = (
    ChatPromptTemplate.from_messages([
        ("system", RESUME_DRAFTER_SYSTEM),
        ("human", """═══ VERIFIED CONTEXT (source of truth — do not deviate) ═══

SKILLS AVAILABLE: {skills}

EXPERIENCE:
{experience_json}

PROJECTS:
{projects_json}

EDUCATION:
{education_json}

ACHIEVEMENTS:
{achievements}

═══ JOB TARGET ═══
Job Title: {job_title}
Company: {job_company}
JD Keywords to cover: {jd_keywords}
Keyword gaps from last draft: {keyword_gap}

═══ ITERATION INSTRUCTIONS ═══
Iteration: {iteration}
Focus for this iteration: {iteration_focus}
Recommended projects to feature: {recommended_projects}
Priority skills to lead with: {priority_skills}

Generate the complete resume as structured output."""),
    ])
    | tier3_llm.with_structured_output(ResumeDraftContent)
)


def resume_drafter(state: AgentState) -> dict:
    """
    Node: resume_drafter
    Reads:  state.verified_context, state.current_job_id, state.iteration,
            state._assembler_output (from context_assembler)
    Writes: state.draft_pool (new ResumeDraft appended), state.iteration (+1)
    Model:  Claude Opus
    Output: ResumeDraftContent (structured) → wrapped in ResumeDraft
    """
    import json
    from uuid import uuid4
    from src.schemas.models import ResumeDraft

    ctx     = state["verified_context"]
    job     = next(j for j in state["jobs"] if j.job_id == state["current_job_id"])
    asm_out = state.get("_assembler_output")  # ContextAssemblerOutput
    iteration = (state.get("iteration") or 0) + 1

    result: ResumeDraftContent = resume_drafter_chain.invoke({
        "skills":              ", ".join(ctx.skills),
        "experience_json":     json.dumps([e.model_dump() for e in ctx.experience], indent=2),
        "projects_json":       json.dumps([p.model_dump() for p in ctx.projects],    indent=2),
        "education_json":      json.dumps([e.model_dump() for e in ctx.education],   indent=2),
        "achievements":        "\n".join(f"- {a}" for a in ctx.achievements) or "none",
        "job_title":           job.title,
        "job_company":         job.company,
        "jd_keywords":         ", ".join(asm_out.jd_keywords)          if asm_out else "",
        "keyword_gap":         ", ".join(asm_out.keyword_gap)           if asm_out else "",
        "iteration":           iteration,
        "iteration_focus":     asm_out.iteration_focus                  if asm_out else "Focus on completeness",
        "recommended_projects":  ", ".join(asm_out.recommended_projects)  if asm_out else "",
        "priority_skills":       ", ".join(asm_out.priority_skills)       if asm_out else "",
    })

    draft = ResumeDraft(
        draft_id=str(uuid4()),
        job_id=job.job_id,
        iteration=iteration,
        content=result,
        # ats_score and ats_breakdown set by ats_scorer in next node
    )

    return {
        "draft_pool": [draft],
        "iteration":  iteration,
    }


# ──────────────────────────────────────────────────────────────
# NODE: retry_planner
# Model: Opus (nuanced diff analysis)
# ──────────────────────────────────────────────────────────────

retry_planner_chain = (
    ChatPromptTemplate.from_messages([
        ("system", """You are a resume improvement strategist. You analyse two rejected
resume drafts and produce specific, actionable improvement directives for the next
generation loop.

RULES:
- improvement_directives must reference specific verified_context fields.
  Example: "Use exact bullet from experience[0].bullets[2] which contains the 40% metric"
- Do NOT direct the LLM to add information not present in the verified context.
- weakness_analysis must cite specific ATS breakdown findings — no generic advice.
- user_feedback_addressed must quote the user's exact words before describing the fix."""),

        ("human", """REJECTED DRAFT A (score: {score_a}):
{draft_a_json}

ATS BREAKDOWN A:
{breakdown_a}

REJECTED DRAFT B (score: {score_b}):
{draft_b_json}

ATS BREAKDOWN B:
{breakdown_b}

USER REJECTION FEEDBACK: {user_feedback}

JOB DESCRIPTION:
{jd_text}

Analyse both drafts and produce improvement directives for the next loop."""),
    ])
    | tier3_llm.with_structured_output(RetryPlannerOutput)
)


def retry_planner(state: AgentState) -> dict:
    """
    Node: retry_planner
    Reads:  state.top2_drafts, state.rejection_feedback, state.current_job_id
    Writes: state.retry_context, state.iteration (reset to 0), state.draft_pool (cleared)
    Model:  Claude Opus
    Output: RetryPlannerOutput (structured)
    """
    import json

    drafts = state["top2_drafts"] or []
    if len(drafts) < 2:
        drafts = drafts + [drafts[0]] if drafts else []

    job = next(j for j in state["jobs"] if j.job_id == state["current_job_id"])
    a, b = drafts[0], drafts[1] if len(drafts) > 1 else drafts[0]

    result: RetryPlannerOutput = retry_planner_chain.invoke({
        "score_a":      f"{a.ats_score:.0%}",
        "draft_a_json": json.dumps(a.content.model_dump(), indent=2),
        "breakdown_a":  json.dumps(a.ats_breakdown.model_dump(), indent=2) if a.ats_breakdown else "{}",
        "score_b":      f"{b.ats_score:.0%}",
        "draft_b_json": json.dumps(b.content.model_dump(), indent=2),
        "breakdown_b":  json.dumps(b.ats_breakdown.model_dump(), indent=2) if b.ats_breakdown else "{}",
        "user_feedback": state.get("rejection_feedback") or "No feedback provided.",
        "jd_text":       job.jd_text,
    })

    return {
        "retry_context": result,
        "iteration":     0,         # reset counter for new loop
        "draft_pool":    [],        # clear old drafts
        "user_decision": None,      # clear stale decision
    }