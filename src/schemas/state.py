"""
graph/state.py
==============
LangGraph AgentState definition.

Design principles:
- TypedDict (not Pydantic BaseModel) because LangGraph requires TypedDict for State.
- Every field has a description comment explaining:
    - what node WRITES it
    - what nodes READ it
    - what reducer applies (and why)
- Reducers are defined here alongside the state so they're never separated
  from the fields they govern.
- Immutability is enforced via freeze_reducer for verified_context — the most
  critical hallucination guard in the system.
"""

from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict

from src.schemas.models import (
    JobRecord,
    ResumeDraft,
    ResumeDraftContent,
    RetryPlannerOutput,
    UserConfig,
    VerifiedContext,
)


# ──────────────────────────────────────────────────────────────
# REDUCERS
# Each reducer is a pure function: (current_value, update) → new_value
# LangGraph calls the reducer when a node returns a partial state update.
# ──────────────────────────────────────────────────────────────

def freeze_reducer(current: VerifiedContext | None, update: VerifiedContext | None) -> VerifiedContext | None:
    """
    Immutability guard for verified_context.

    WRITTEN BY: store_context (exactly once, at session start)
    READ BY:    all LLM nodes downstream

    Once set, verified_context must never be overwritten. Any node attempting
    to write a non-None value over an existing non-None value raises an error.
    This is the core anti-hallucination contract — the LLM cannot silently
    change the ground-truth facts it was given.
    """
    if current is not None and update is not None:
        raise ValueError(
            "verified_context is immutable after store_context writes it. "
            "No downstream node may overwrite this field."
        )
    return update if update is not None else current


def jobs_append_reducer(current: list[JobRecord] | None, update: list[JobRecord]) -> list[JobRecord]:
    """
    Append-only reducer for the job list.

    WRITTEN BY: job_fetcher (appends new batches), archive_job / spawn_resume_worker
                (update status of individual jobs)
    READ BY:    relevance_scorer, append_jobs_to_db, job_hitl, select_top2

    - New jobs are appended only if their job_id is not already present (dedup).
    - Status updates for existing jobs are applied in-place by job_id.
    - The list is sorted by relevance_score descending after every update so
      the HITL queue always shows the most relevant job first.
    """
    existing = current or []
    existing_map: dict[str, JobRecord] = {j.job_id: j for j in existing}

    for job in update:
        if job.job_id in existing_map:
            # Status update for an existing job — replace in-place
            existing_map[job.job_id] = job
        else:
            # New job — append
            existing_map[job.job_id] = job

    return sorted(existing_map.values(), key=lambda j: j.relevance_score, reverse=True)


def drafts_reducer(current: list[ResumeDraft] | None, update: list[ResumeDraft]) -> list[ResumeDraft]:
    """
    Sorted-merge reducer for the resume draft pool.

    WRITTEN BY: resume_drafter (adds new draft each iteration),
                ats_scorer (updates ats_score + ats_breakdown on existing drafts)
    READ BY:    route_ats, select_top2, retry_planner

    - New drafts are appended by draft_id.
    - Score updates replace the existing draft record in-place.
    - Pool is always sorted by ats_score descending so select_top2 can simply
      take pool[:2] without re-sorting.
    - Draft pool is scoped to current_job_id — it is CLEARED when a new job
      starts (handled in spawn_resume_worker via clear_draft_pool node).
    """
    existing = current or []
    existing_map: dict[str, ResumeDraft] = {d.draft_id: d for d in existing}

    for draft in update:
        existing_map[draft.draft_id] = draft  # upsert

    return sorted(existing_map.values(), key=lambda d: d.ats_score, reverse=True)


def iteration_reducer(current: int | None, update: int) -> int:
    """
    Iteration counter reducer.

    WRITTEN BY: resume_drafter (increments by 1 each call),
                spawn_resume_worker (resets to 0 for each new job),
                retry_planner (resets to 0 when retry loop starts)
    READ BY:    route_ats, context_assembler, resume_hitl
    """
    return update  # last-write-wins — node is responsible for correct value


def user_decision_reducer(current: str | None, update: str | None) -> str | None:
    """
    Last-write-wins reducer for HITL user decisions.

    WRITTEN BY: job_hitl (sets "approved" | "rejected"),
                resume_hitl (sets "approved" | "retry" | "cancel"),
                spawn_resume_worker (resets to None for next job)
    READ BY:    route_job_decision, route_resume_decision
    """
    return update  # always replace — stale decisions must not persist


# ──────────────────────────────────────────────────────────────
# AGENT STATE
# ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The complete shared state of the AI Job Agent LangGraph.

    Conventions:
    - Fields annotated with a reducer are managed by LangGraph's channel system.
    - Fields WITHOUT a reducer are last-write-wins by default.
    - Every field documents: who writes it, who reads it, and why it exists.
    - Optional fields default to None — nodes must handle None gracefully.

    HALLUCINATION GUARD:
    verified_context uses freeze_reducer — once written by store_context,
    no node can change it. All LLM nodes must derive facts from this field only.
    """

    # ── 1. User config ──────────────────────────────────────────
    user_config: UserConfig
    """
    WRITTEN BY: graph entry point (user-provided at session start)
    READ BY:    context_builder, job_fetcher, relevance_scorer, context_assembler,
                resume_drafter, ats_scorer, route_ats, retry_planner
    REDUCER:    none (last-write-wins; never changes after session start)

    The user's session parameters. All nodes that need to know target roles,
    salary thresholds, ATS threshold, or max iterations read from here.
    Do NOT hardcode any of these values inside node logic.
    """

    # ── 2. Verified context (immutable after store_context) ──────
    verified_context: Annotated[Optional[VerifiedContext], freeze_reducer]
    """
    WRITTEN BY: store_context (once, at session start)
    READ BY:    context_assembler, resume_drafter, retry_planner, ats_scorer
    REDUCER:    freeze_reducer — raises if any node tries to overwrite

    The ground-truth store. All resume content must trace back to fields in
    this object. freeze_reducer enforces immutability at the framework level.
    """

    # ── 3. Job list ──────────────────────────────────────────────
    jobs: Annotated[list[JobRecord], jobs_append_reducer]
    """
    WRITTEN BY: job_fetcher (appends new batches every cycle),
                relevance_scorer (updates relevance_score + keyword_matches),
                archive_job (sets status=rejected),
                spawn_resume_worker (sets status=approved),
                render_resume (sets status=resume_ready)
    READ BY:    append_jobs_to_db, job_hitl, route_job_decision
    REDUCER:    jobs_append_reducer — append-only with status-update upsert

    The live job database. Sorted by relevance_score descending so the HITL
    queue always presents the best match first. Never cleared — grows
    throughout the session.
    """

    # ── 4. Current job being processed ───────────────────────────
    current_job_id: Optional[str]
    """
    WRITTEN BY: spawn_resume_worker (sets to approved job's job_id),
                render_resume (sets to None after delivery)
    READ BY:    context_assembler, resume_drafter, ats_scorer, select_top2,
                resume_hitl, retry_planner, render_resume
    REDUCER:    none (last-write-wins)

    Pointer to the job currently being processed in the resume loop.
    Allows all resume loop nodes to look up the full JobRecord from state.jobs.
    """

    # ── 5. Resume draft pool ─────────────────────────────────────
    draft_pool: Annotated[list[ResumeDraft], drafts_reducer]
    """
    WRITTEN BY: resume_drafter (appends new draft each iteration),
                ats_scorer (updates ats_score + ats_breakdown on draft),
                spawn_resume_worker (clears pool for new job)
    READ BY:    route_ats, select_top2, resume_hitl, retry_planner
    REDUCER:    drafts_reducer — upsert by draft_id, sorted by ats_score desc

    All resume drafts generated for current_job_id in the current loop.
    Cleared (reset to []) when a new job starts so previous job's drafts
    don't pollute the next job's selection.
    """

    # ── 6. Loop iteration counter ─────────────────────────────────
    iteration: Annotated[int, iteration_reducer]
    """
    WRITTEN BY: resume_drafter (i += 1 each call),
                spawn_resume_worker (reset to 0),
                retry_planner (reset to 0)
    READ BY:    route_ats, context_assembler (for iteration_focus prompt)
    REDUCER:    iteration_reducer (last-write-wins)

    Tracks how many resume generation attempts have been made for the
    current job. route_ats uses this alongside user_config.max_iterations
    to decide whether to loop or exit.
    """

    # ── 7. HITL user decision ─────────────────────────────────────
    user_decision: Annotated[Optional[str], user_decision_reducer]
    """
    WRITTEN BY: job_hitl (sets "approved" | "rejected"),
                resume_hitl (sets "approved" | "retry" | "cancel"),
                spawn_resume_worker (resets to None)
    READ BY:    route_job_decision, route_resume_decision
    REDUCER:    user_decision_reducer (last-write-wins)

    The human's decision at each HITL checkpoint. The conditional edge
    functions read this field to determine which branch to follow.
    Reset to None after each routing decision to prevent stale values
    from influencing the next HITL checkpoint.
    """

    # ── 8. Rejection feedback (retry loop only) ───────────────────
    rejection_feedback: Optional[str]
    """
    WRITTEN BY: resume_hitl (captured when user_decision == "retry")
    READ BY:    retry_planner
    REDUCER:    none (last-write-wins)

    Free-text feedback from the user explaining why they rejected both
    top-2 resumes. Passed into retry_planner so it can produce targeted
    improvement directives. None if the user provided no feedback.
    """

    # ── 9. Retry context (set by retry_planner) ───────────────────
    retry_context: Optional[RetryPlannerOutput]
    """
    WRITTEN BY: retry_planner
    READ BY:    context_assembler (injected into iteration_focus on first
                iteration of the retry loop)
    REDUCER:    none (last-write-wins)

    The structured output from retry_planner containing diff_summary,
    weakness_analysis, and improvement_directives. context_assembler reads
    this on iteration 1 of a retry loop and injects directives into the
    LLM prompt. Cleared to None by spawn_resume_worker for new jobs.
    """

    # ── 10. Top-2 selected drafts ─────────────────────────────────
    top2_drafts: Optional[list[ResumeDraft]]
    """
    WRITTEN BY: select_top2 (picks draft_pool[:2])
    READ BY:    resume_hitl (displays to user), retry_planner (receives rejected pair)
    REDUCER:    none (last-write-wins)

    The two best-scoring drafts selected for user review. Always exactly
    2 items (or 1 if only 1 draft was generated). Displayed side-by-side
    in the HITL UI with ATS score breakdowns.
    """

    # ── 11. Final approved draft ──────────────────────────────────
    approved_draft: Optional[ResumeDraft]
    """
    WRITTEN BY: resume_hitl (set when user_decision == "approved")
    READ BY:    render_resume
    REDUCER:    none (last-write-wins)

    The specific draft the user approved. render_resume reads this to
    produce the final .docx and .pdf outputs. None until approval.
    """

    # ── 12. Rendered output paths ─────────────────────────────────
    rendered_outputs: Optional[dict[str, str]]
    """
    WRITTEN BY: render_resume
    READ BY:    final UI / dashboard
    REDUCER:    none (last-write-wins)

    Paths to the rendered resume files:
    {
        "docx": "/outputs/{job_id}_resume.docx",
        "pdf":  "/outputs/{job_id}_resume.pdf",
        "job_id": "...",
        "ats_score": "0.87"
    }
    """

    # ── 13. Error tracking ────────────────────────────────────────
    last_error: Optional[str]
    """
    WRITTEN BY: any node on exception (caught at graph level)
    READ BY:    error recovery logic, LangSmith tracing
    REDUCER:    none (last-write-wins)

    Last error message if a node failed. Used for graceful recovery and
    displayed in the LangSmith trace for debugging. None in normal operation.
    """