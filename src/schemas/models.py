"""
All Pydantic domain models for the AI Job Agent.

Design principles:
- Every field carries a `description` → fed directly into LLM structured-output
  prompts so the model knows exactly what to populate and what NOT to invent.
- `Field(...)` with no default = required; the LLM must produce it.
- `Field(None)` = optional; the LLM leaves it None rather than hallucinating.
- Validators enforce data contracts so bad LLM output is caught before it
  reaches downstream nodes.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────
# ENUMS  (constrain LLM choices to valid values)
# ─────────────────────────────────────────────

class WorkType(str, Enum):
    REMOTE     = "REMOTE"
    ONSITE     = "ONSITE"
    HYBRID     = "HYBRID"
    CONTRACT   = "CONTRACT"
    FULLTIME   = "FULLTIME"
    PARTTIME   = "PARTTIME"
    INTERNSHIP = "INTERNSHIP"


class JobStatus(str, Enum):
    PENDING      = "pending"       # fetched, not yet shown to user
    APPROVED     = "approved"      # user approved for resume generation
    REJECTED     = "rejected"      # user rejected — archive
    RESUME_READY = "resume_ready"  # final resume generated and approved


class ResumeDecision(str, Enum):
    APPROVED = "approved"  # user approved one of the top-2 resumes
    RETRY    = "retry"     # user rejected both — trigger retry loop
    CANCEL   = "cancel"    # user cancels this job entirely


class ATSSection(str, Enum):
    """Standard resume sections that ATS systems scan for."""
    CONTACT    = "contact"
    SUMMARY    = "summary"
    EXPERIENCE = "experience"
    EDUCATION  = "education"
    SKILLS     = "skills"
    PROJECTS   = "projects"
    CERTIFICATIONS = "certifications"
    ACHIEVEMENTS   = "achievements"


# ─────────────────────────────────────────────
# USER CONFIG
# ─────────────────────────────────────────────

class UserConfig(BaseModel):
    """
    Configuration provided by the user at session start.
    This is the single source of truth for all search and generation parameters.
    """

    github_username: str = Field(
        ...,
        description="GitHub username for fetching projects dynamically. Never hardcode this in nodes."
    )

    base_resume_path: str = Field(
        ...,
        description="Absolute path to the base .docx resume for structural parsing."
    )

    target_roles: list[str] = Field(
        ...,
        description=(
            "List of exact job titles the user is targeting. "
            "Examples: ['Machine Learning Engineer', 'Data Scientist']. "
            "Use these verbatim when constructing Apify search queries."
        ),
        min_length=1,
    )

    manual_skills: list[str] = Field(
        ...,
        description=(
            "Skills explicitly provided by the user. "
            "These are ground truth — do NOT add, remove, or rephrase any skill. "
            "Examples: ['Python', 'PyTorch', 'LangChain', 'SQL']."
        ),
    )

    salary_min: int = Field(
        ...,
        description=(
            "Minimum acceptable annual salary in USD (integer). "
            "Jobs below this figure must be hard-filtered out before HITL review."
        ),
        ge=0,
    )

    salary_max: Optional[int] = Field(
        None,
        description=(
            "Maximum acceptable annual salary in USD. "
            "None means no upper limit. Do not infer or estimate this value."
        ),
        ge=0,
    )

    work_type: WorkType = Field(
        ...,
        description=(
            "Preferred work arrangement. Must be one of the WorkType enum values. "
            "Use this to filter Apify actor search parameters."
        ),
    )

    location: str = Field(
        ...,
        description=(
            "Target location for job search. "
            "Use country name for remote roles (e.g. 'usa', 'india'). "
            "Use city + country for on-site roles (e.g. 'bangalore, india')."
        ),
    )

    max_iterations: int = Field(
        ...,
        description=(
            "Maximum number of resume generation attempts per job before "
            "surfacing the top-2 drafts to the user, regardless of ATS score. "
            "Recommended range: 3–10."
        ),
        ge=1,
        le=20,
    )

    ats_threshold: float = Field(
        ...,
        description=(
            "Minimum ATS score (0.0–1.0) required to exit the generation loop early. "
            "Example: 0.80 means exit as soon as a draft scores ≥ 80. "
            "The loop also exits when max_iterations is reached, whichever comes first."
        ),
        ge=0.0,
        le=1.0,
    )

    platforms: list[str] = Field(
        default=["linkedin", "indeed", "google_jobs"],
        description=(
            "Job platforms to search via Apify actors. "
            "Valid values: 'linkedin', 'indeed', 'google_jobs', 'greenhouse', 'lever'. "
            "Determines which Apify actor IDs to invoke in job_fetcher."
        ),
    )

    @model_validator(mode="after")
    def salary_range_valid(self) -> "UserConfig":
        if self.salary_max is not None and self.salary_max < self.salary_min:
            raise ValueError("salary_max must be greater than or equal to salary_min.")
        return self


# ─────────────────────────────────────────────
# RESUME SOURCE MODELS
# (parsed from real documents — never LLM-generated)
# ─────────────────────────────────────────────

class ExperienceEntry(BaseModel):
    """
    A single work experience entry extracted from the base resume.
    ALL fields must be copied verbatim from the source document.
    Do NOT rephrase, summarise, or expand any field.
    """

    company: str = Field(
        ...,
        description=(
            "Exact company name as written in the resume. "
            "Do not abbreviate or expand. Example: 'Google LLC', not 'Google'."
        ),
    )

    role: str = Field(
        ...,
        description=(
            "Exact job title as written in the resume. "
            "Example: 'Senior Software Engineer', not 'SWE' or 'Software Engineer'."
        ),
    )

    start_date: str = Field(
        ...,
        description=(
            "Start date exactly as written in the resume. "
            "Preserve format: 'Jan 2022', '2022-01', 'January 2022' — do not reformat."
        ),
    )

    end_date: Optional[str] = Field(
        None,
        description=(
            "End date exactly as written in the resume. "
            "None if the position is current. Do not infer or estimate."
        ),
    )

    is_current: bool = Field(
        ...,
        description=(
            "True if this is the user's current position (end_date is None or 'Present'). "
            "Set based on resume content only."
        ),
    )

    location: Optional[str] = Field(
        None,
        description=(
            "Work location as written in the resume. "
            "None if not mentioned. Do not infer from company name."
        ),
    )

    bullets: list[str] = Field(
        ...,
        description=(
            "List of achievement/responsibility bullets copied verbatim from the resume. "
            "Preserve exact wording including metrics (e.g. '40% reduction in latency'). "
            "Do not paraphrase, combine, or add bullets not present in the source."
        ),
        min_length=1,
    )

    technologies: list[str] = Field(
        default_factory=list,
        description=(
            "Technologies explicitly mentioned in the bullets for this role. "
            "Extract only — do not infer technologies from the company or role name."
        ),
    )


class InternshipEntry(BaseModel):
    """
    An internship entry extracted from the base resume or verified from the
    internship URL. Fields sourced from URL must be marked with source='url'.
    """

    company: str = Field(
        ...,
        description="Exact company name as written in the resume. Do not alter.",
    )

    role: str = Field(
        ...,
        description="Exact internship title as written in the resume.",
    )

    start_date: str = Field(
        ...,
        description="Start date as written in the resume. Preserve format.",
    )

    end_date: Optional[str] = Field(
        None,
        description="End date as written in the resume. None if ongoing.",
    )

    bullets: list[str] = Field(
        default_factory=list,
        description=(
            "Responsibility/achievement bullets from the resume. "
            "If sourced from the internship URL, copy the exact text from the page. "
            "Never invent bullets."
        ),
    )

    source_url: Optional[str] = Field(
        None,
        description=(
            "The URL from which additional details were fetched, if any. "
            "None if all data came from the resume document."
        ),
    )

    verified_from_url: bool = Field(
        default=False,
        description=(
            "True if the company/role details were cross-checked against source_url. "
            "This flag tells downstream nodes this entry has extra verification."
        ),
    )


class EducationEntry(BaseModel):
    """Education entry extracted verbatim from the base resume."""

    institution: str = Field(
        ...,
        description="Exact institution name as written in the resume.",
    )

    degree: str = Field(
        ...,
        description=(
            "Exact degree name as written. "
            "Example: 'B.Tech in Computer Science', not 'Bachelor of Technology'."
        ),
    )

    graduation_year: Optional[str] = Field(
        None,
        description=(
            "Graduation year or expected year as written. "
            "Example: '2024', 'May 2025'. Do not compute or estimate."
        ),
    )

    gpa: Optional[str] = Field(
        None,
        description=(
            "GPA as written in the resume (e.g. '8.5/10', '3.8/4.0'). "
            "None if not mentioned. Do not calculate or estimate."
        ),
    )

    relevant_coursework: list[str] = Field(
        default_factory=list,
        description=(
            "Courses listed in the resume under this education entry only. "
            "Do not add courses not explicitly mentioned."
        ),
    )


class ProjectEntry(BaseModel):
    """
    A project entry sourced from GitHub or the base resume.
    All technical details must come from the actual repo/README — never invented.
    """

    name: str = Field(
        ...,
        description="Exact repository or project name. Do not rename or stylise.",
    )

    description: str = Field(
        ...,
        description=(
            "Project description taken verbatim from the GitHub README or resume. "
            "Do not expand, summarise, or rephrase. "
            "Maximum 2 sentences. If README is empty, use the repo description field."
        ),
    )

    tech_stack: list[str] = Field(
        ...,
        description=(
            "Technologies explicitly stated in the README or detected from repo language stats. "
            "Do not infer frameworks from the project name or description."
        ),
        min_length=1,
    )

    deployed_url: Optional[str] = Field(
        None,
        description=(
            "Live URL of the deployed project if present in the README or repo metadata. "
            "None if no deployment link exists. Do not construct or guess URLs."
        ),
    )

    github_url: str = Field(
        ...,
        description="Full GitHub repository URL. Must start with 'https://github.com/'.",
    )

    is_pinned: bool = Field(
        default=False,
        description=(
            "True if the user has pinned this repo on their GitHub profile. "
            "Pinned repos are prioritised in resume generation."
        ),
    )

    last_commit_date: Optional[str] = Field(
        None,
        description=(
            "Date of the most recent commit in ISO 8601 format (YYYY-MM-DD). "
            "Used to prioritise recently active projects."
        ),
    )

    stars: int = Field(
        default=0,
        description="GitHub star count at time of fetch. Used as a quality signal.",
        ge=0,
    )

    @field_validator("github_url")
    @classmethod
    def must_be_github(cls, v: str) -> str:
        if not v.startswith("https://github.com/"):
            raise ValueError("github_url must start with 'https://github.com/'")
        return v


# ─────────────────────────────────────────────
# VERIFIED CONTEXT STORE
# ─────────────────────────────────────────────

class VerifiedContext(BaseModel):
    """
    The immutable ground-truth store written by store_context and read by all
    downstream LLM nodes. No LLM node may write to this object after it is set.

    ANTI-HALLUCINATION CONTRACT:
    Every field in every resume draft must trace back to a field in this object.
    If a fact does not exist here, it must not appear in any generated resume.
    """

    skills: list[str] = Field(
        ...,
        description=(
            "Deduplicated union of user-provided manual_skills and skills extracted "
            "from the base resume. This is the authoritative skills list. "
            "Resume drafts may only use skills from this list."
        ),
        min_length=1,
    )

    experience: list[ExperienceEntry] = Field(
        ...,
        description=(
            "All work experience entries extracted from the base resume. "
            "Ordered chronologically, most recent first. "
            "Resume drafts must use only bullets and facts from these entries."
        ),
    )

    achievements: list[str] = Field(
        default_factory=list,
        description=(
            "Standalone achievements listed in the resume outside of experience bullets. "
            "Examples: awards, publications, open-source contributions. "
            "Copy verbatim — do not paraphrase."
        ),
    )

    internships: list[InternshipEntry] = Field(
        default_factory=list,
        description=(
            "All internship entries from the resume, optionally enriched from URLs. "
            "Resume drafts may reference these for early-career context."
        ),
    )

    education: list[EducationEntry] = Field(
        ...,
        description=(
            "All education entries from the resume. "
            "Must contain at least one entry."
        ),
        min_length=1,
    )

    projects: list[ProjectEntry] = Field(
        default_factory=list,
        description=(
            "Projects from GitHub, ordered by: pinned first, then by stars, "
            "then by last_commit_date. Resume drafts pick from this list only."
        ),
    )

    user_config: UserConfig = Field(
        ...,
        description=(
            "The user's session configuration. Read-only reference for all nodes. "
            "Do not modify after store_context writes it."
        ),
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when this context was built. Used for audit/trace.",
    )


# ─────────────────────────────────────────────
# JOB MODELS
# ─────────────────────────────────────────────

class JobRecord(BaseModel):
    """
    A single normalised job listing fetched from Apify and scored by
    relevance_scorer. This is the structure stored in the job database.
    """

    job_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier assigned at normalisation time. Never reassign.",
    )

    title: str = Field(
        ...,
        description=(
            "Job title as returned by the Apify actor. "
            "Do not normalise or rephrase — preserve the employer's exact wording."
        ),
    )

    company: str = Field(
        ...,
        description="Company name as returned by the Apify actor.",
    )

    location: Optional[str] = Field(
        None,
        description=(
            "Job location as returned by the Apify actor. "
            "None if fully remote with no location specified."
        ),
    )

    salary_raw: Optional[str] = Field(
        None,
        description=(
            "Salary string exactly as returned by the Apify actor. "
            "Example: '$120,000 - $160,000 a year'. "
            "None if not provided. Do not compute or estimate salary."
        ),
    )

    salary_min_usd: Optional[int] = Field(
        None,
        description=(
            "Lower bound of the salary range parsed from salary_raw, in USD. "
            "None if salary_raw is None or cannot be reliably parsed. "
            "Do not estimate — leave None if uncertain."
        ),
        ge=0,
    )

    work_type: Optional[WorkType] = Field(
        None,
        description=(
            "Normalised work arrangement parsed from the listing. "
            "None if not clearly stated in the listing."
        ),
    )

    jd_text: str = Field(
        ...,
        description=(
            "Full job description text as returned by the Apify actor. "
            "This is the primary input to relevance_scorer and context_assembler. "
            "Do not truncate or summarise."
        ),
    )

    url: str = Field(
        ...,
        description=(
            "Canonical job listing URL with tracking parameters stripped. "
            "Used as the deduplication key across platforms."
        ),
    )

    source_platform: str = Field(
        ...,
        description=(
            "Platform this listing was fetched from. "
            "Valid values: 'linkedin', 'indeed', 'google_jobs', 'greenhouse', 'lever'."
        ),
    )

    relevance_score: float = Field(
        default=0.0,
        description=(
            "Score assigned by relevance_scorer (0.0–10.0) indicating how well "
            "this job matches the user's profile. Jobs below 3.0 are soft-filtered "
            "from the HITL queue. Do not set this manually."
        ),
        ge=0.0,
        le=10.0,
    )

    keyword_matches: list[str] = Field(
        default_factory=list,
        description=(
            "Skills from verified_context.skills that appear in jd_text. "
            "Populated by relevance_scorer. Used for HITL display and ATS scoring."
        ),
    )

    status: JobStatus = Field(
        default=JobStatus.PENDING,
        description=(
            "Current status of this job in the pipeline. "
            "Transitions: pending → approved/rejected → resume_ready."
        ),
    )

    fetched_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when this job was fetched from Apify.",
    )

    @field_validator("url")
    @classmethod
    def strip_tracking(cls, v: str) -> str:
        return v.split("?")[0].lower().rstrip("/")


# ─────────────────────────────────────────────
# ATS SCORING MODELS
# ─────────────────────────────────────────────

class ATSScoreBreakdown(BaseModel):
    """
    Deterministic ATS score breakdown produced by the rule-based ats_scorer node.
    This is NEVER produced by an LLM — it is computed by spaCy/NLTK rule logic.
    """

    keyword_score: float = Field(
        ...,
        description=(
            "Fraction of JD keywords found in the resume draft (0.0–1.0). "
            "Computed as: matched_keywords / total_jd_keywords. Weight: 40%."
        ),
        ge=0.0,
        le=1.0,
    )

    section_score: float = Field(
        ...,
        description=(
            "Fraction of required ATS sections present in the draft (0.0–1.0). "
            "Required sections: contact, summary, experience, education, skills. "
            "Weight: 25%."
        ),
        ge=0.0,
        le=1.0,
    )

    format_score: float = Field(
        ...,
        description=(
            "Format compliance score (0.0–1.0). Checks: single-column layout, "
            "no tables, no images, standard section headings, valid date formats. "
            "Each violation deducts 0.2. Weight: 20%."
        ),
        ge=0.0,
        le=1.0,
    )

    density_score: float = Field(
        ...,
        description=(
            "Word count and bullet density score (0.0–1.0). "
            "Full score for 450–600 words and 3–5 bullets per role. "
            "Linear deduction outside these ranges. Weight: 15%."
        ),
        ge=0.0,
        le=1.0,
    )

    total_score: float = Field(
        ...,
        description=(
            "Weighted total: (keyword×0.4) + (section×0.25) + (format×0.20) + (density×0.15). "
            "Range: 0.0–1.0. Loop exits when this meets ats_threshold."
        ),
        ge=0.0,
        le=1.0,
    )

    matched_keywords: list[str] = Field(
        ...,
        description="JD keywords found verbatim or as synonyms in the draft.",
    )

    missing_keywords: list[str] = Field(
        ...,
        description=(
            "JD keywords absent from the draft. "
            "Passed back to context_assembler as keyword_gap on the next iteration."
        ),
    )

    missing_sections: list[ATSSection] = Field(
        default_factory=list,
        description="Required sections absent from the draft.",
    )

    format_violations: list[str] = Field(
        default_factory=list,
        description=(
            "List of specific format issues found. "
            "Examples: 'table detected', 'non-standard heading: Core Competencies'."
        ),
    )

    word_count: int = Field(
        ...,
        description="Total word count of the resume draft.",
        ge=0,
    )

    @model_validator(mode="after")
    def compute_total(self) -> "ATSScoreBreakdown":
        computed = (
            self.keyword_score * 0.40
            + self.section_score * 0.25
            + self.format_score  * 0.20
            + self.density_score * 0.15
        )
        if abs(self.total_score - computed) > 0.01:
            raise ValueError(
                f"total_score {self.total_score} does not match computed "
                f"weighted sum {computed:.3f}. Do not set total_score manually."
            )
        return self


# ─────────────────────────────────────────────
# RESUME DRAFT MODELS
# (structured output from resume_drafter LLM node)
# ─────────────────────────────────────────────

class ResumeBullet(BaseModel):
    """
    A single bullet point in a resume section.
    Every bullet must cite its source to prevent hallucination.
    """

    text: str = Field(
        ...,
        description=(
            "The bullet text to appear in the resume. "
            "Must be grounded in a fact from verified_context. "
            "Quantify with real metrics only if they appear in the source. "
            "Do not invent percentages, team sizes, or impact numbers."
        ),
    )

    source_field: str = Field(
        ...,
        description=(
            "Dot-path to the verified_context field this bullet is derived from. "
            "Examples: 'experience[0].bullets[2]', 'projects[1].description'. "
            "This citation is validated post-generation against verified_context."
        ),
    )

    uses_keywords: list[str] = Field(
        default_factory=list,
        description=(
            "JD keywords intentionally incorporated into this bullet. "
            "Must appear verbatim in the bullet text."
        ),
    )


class ResumeExperienceSection(BaseModel):
    """Structured experience entry for the generated resume."""

    company: str = Field(
        ...,
        description=(
            "Company name from verified_context.experience[i].company. "
            "Copy exactly — do not alter."
        ),
    )

    role: str = Field(
        ...,
        description=(
            "Role title from verified_context.experience[i].role. "
            "May be lightly reworded to align with the JD title "
            "ONLY IF the core function is identical. Never fabricate a title."
        ),
    )

    start_date: str = Field(
        ...,
        description="From verified_context.experience[i].start_date. Copy exactly.",
    )

    end_date: Optional[str] = Field(
        None,
        description="From verified_context.experience[i].end_date. None if current.",
    )

    location: Optional[str] = Field(
        None,
        description="From verified_context.experience[i].location. None if remote.",
    )

    bullets: list[ResumeBullet] = Field(
        ...,
        description=(
            "3–5 bullets selected and/or lightly reworded from "
            "verified_context.experience[i].bullets to best match the JD keywords. "
            "Never add bullets not derived from the source. Min 3, max 6."
        ),
        min_length=3,
    )


class ResumeProjectSection(BaseModel):
    """Structured project entry for the generated resume."""

    name: str = Field(
        ...,
        description="From verified_context.projects[i].name. Copy exactly.",
    )

    tech_stack: list[str] = Field(
        ...,
        description=(
            "From verified_context.projects[i].tech_stack. "
            "Select technologies that overlap with JD requirements. "
            "Do not add technologies not in the source."
        ),
    )

    bullets: list[ResumeBullet] = Field(
        ...,
        description=(
            "1–3 bullets derived from verified_context.projects[i].description. "
            "Reword to highlight JD-relevant aspects. "
            "Never invent features or metrics not in the source."
        ),
        min_length=1,
    )

    deployed_url: Optional[str] = Field(
        None,
        description="From verified_context.projects[i].deployed_url. None if absent.",
    )

    github_url: str = Field(
        ...,
        description="From verified_context.projects[i].github_url. Copy exactly.",
    )


class ResumeDraftContent(BaseModel):
    """
    The full structured content of a generated resume.
    This is the schema the resume_drafter LLM must produce as structured output.

    ANTI-HALLUCINATION RULES embedded in field descriptions:
    - Every field must trace to verified_context.
    - source_citations map each section to its origin.
    - The validator checks citations are non-empty.
    """

    summary: str = Field(
        ...,
        description=(
            "2–3 sentence professional summary. "
            "Must only reference: skills from verified_context.skills, "
            "roles from verified_context.experience, and the target role from user_config. "
            "Do not claim years of experience not supported by the experience entries. "
            "Do not mention companies not in verified_context."
        ),
    )

    skills_used: list[str] = Field(
        ...,
        description=(
            "Skills selected from verified_context.skills that are relevant to this JD. "
            "Do not add skills not in verified_context.skills. "
            "Order by: JD keyword match first, then general relevance."
        ),
        min_length=5,
    )

    experience: list[ResumeExperienceSection] = Field(
        ...,
        description=(
            "Experience entries selected from verified_context.experience. "
            "Include all entries unless max resume length requires trimming. "
            "Most recent first."
        ),
        min_length=1,
    )

    projects: list[ResumeProjectSection] = Field(
        default_factory=list,
        description=(
            "Up to 3 projects selected from verified_context.projects. "
            "Prioritise: pinned repos, tech overlap with JD, recent commits."
        ),
    )

    education: list[dict] = Field(
        ...,
        description=(
            "Education entries copied from verified_context.education. "
            "Copy all fields verbatim — do not reformat dates or alter institution names."
        ),
        min_length=1,
    )

    achievements: list[str] = Field(
        default_factory=list,
        description=(
            "Selected achievements from verified_context.achievements "
            "that are relevant to this JD. Copy verbatim — do not rephrase."
        ),
    )

    source_citations: dict[str, str] = Field(
        ...,
        description=(
            "Map of resume section → verified_context source path. "
            "Example: {'summary': 'user_config.target_roles + experience[0].role', "
            "'experience[0]': 'experience[0]', 'projects[0]': 'projects[2]'}. "
            "Every section must have an entry. Validated post-generation."
        ),
    )

    @field_validator("source_citations")
    @classmethod
    def citations_non_empty(cls, v: dict) -> dict:
        if not v:
            raise ValueError(
                "source_citations must not be empty. "
                "Every generated section needs a citation to verified_context."
            )
        return v


class ResumeDraft(BaseModel):
    """
    A single resume generation attempt — wraps ResumeDraftContent with
    metadata for tracking iterations and ATS scoring across the loop.
    """

    draft_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique ID for this draft. Auto-assigned — do not set manually.",
    )

    job_id: str = Field(
        ...,
        description="job_id from the JobRecord this draft was generated for.",
    )

    iteration: int = Field(
        ...,
        description=(
            "1-indexed iteration number within the resume loop for this job. "
            "Starts at 1, increments each time resume_drafter runs."
        ),
        ge=1,
    )

    content: ResumeDraftContent = Field(
        ...,
        description="The structured resume content produced by resume_drafter.",
    )

    ats_score: float = Field(
        default=0.0,
        description=(
            "ATS total score assigned by ats_scorer after generation. "
            "Set by ats_scorer — do not set in resume_drafter output."
        ),
        ge=0.0,
        le=1.0,
    )

    ats_breakdown: Optional[ATSScoreBreakdown] = Field(
        default=None,
        description=(
            "Full ATS breakdown set by ats_scorer. "
            "None until ats_scorer runs on this draft."
        ),
    )

    keyword_gap: list[str] = Field(
        default_factory=list,
        description=(
            "JD keywords missing from this draft — from ats_breakdown.missing_keywords. "
            "Passed into the next iteration's context_assembler prompt."
        ),
    )

    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when resume_drafter produced this draft.",
    )


# ─────────────────────────────────────────────
# LLM STRUCTURED OUTPUT SCHEMAS
# (what each LLM node must return — used with .with_structured_output())
# ─────────────────────────────────────────────

class RelevanceScorerOutput(BaseModel):
    """
    Structured output schema for the relevance_scorer node (Claude Haiku).
    The LLM must return exactly this schema — nothing more, nothing less.
    """

    score: float = Field(
        ...,
        description=(
            "Relevance score from 0.0 to 10.0. "
            "10.0 = perfect match for all role, skill, salary, and work-type criteria. "
            "0.0 = completely irrelevant. "
            "Score based ONLY on the job description text and user_config. "
            "Do not factor in company reputation or location preference."
        ),
        ge=0.0,
        le=10.0,
    )

    matched_keywords: list[str] = Field(
        ...,
        description=(
            "Skills from the user's skills list that appear explicitly in the JD text. "
            "Only include skills that appear verbatim or as clear synonyms. "
            "Do not infer skill matches."
        ),
    )

    hard_fail: bool = Field(
        ...,
        description=(
            "True if this job fails a hard filter and must be excluded from HITL queue. "
            "Hard fail conditions: salary below salary_min (if salary is parseable), "
            "work_type mismatch (if clearly stated), role completely unrelated to target_roles."
        ),
    )

    hard_fail_reason: Optional[str] = Field(
        None,
        description=(
            "Brief reason for hard_fail=True. "
            "None if hard_fail=False. "
            "Example: 'Salary $40k below minimum $80k threshold'."
        ),
    )

    summary: str = Field(
        ...,
        description=(
            "1–2 sentence summary of why this job is or isn't a good match. "
            "Shown to user in the HITL review UI. Be specific — mention actual JD content."
        ),
    )


class ContextAssemblerOutput(BaseModel):
    """
    Structured output from context_assembler (Claude Sonnet).
    Produces the fully-formed prompt context for resume_drafter.
    """

    keyword_gap: list[str] = Field(
        ...,
        description=(
            "JD keywords that are absent from verified_context.skills AND "
            "the previous draft (if iteration > 1). "
            "These are the gaps the next draft must address. "
            "Only include keywords that genuinely appear in the JD."
        ),
    )

    priority_skills: list[str] = Field(
        ...,
        description=(
            "Skills from verified_context.skills ordered by JD relevance. "
            "Most critical to the JD first. Used to guide skill section ordering."
        ),
    )

    recommended_projects: list[str] = Field(
        ...,
        description=(
            "project names from verified_context.projects recommended for this JD. "
            "Select based on tech_stack overlap with JD. Max 3. "
            "Only names that exist in verified_context.projects."
        ),
    )

    recommended_experience: list[str] = Field(
        ...,
        description=(
            "Company names from verified_context.experience recommended for emphasis. "
            "Order by relevance to JD. Only companies in verified_context.experience."
        ),
    )

    jd_keywords: list[str] = Field(
        ...,
        description=(
            "All significant technical keywords extracted from the JD text. "
            "Exclude generic words (team player, communication, etc). "
            "These feed the ATS keyword scorer."
        ),
    )

    iteration_focus: str = Field(
        ...,
        description=(
            "Specific instruction for this iteration's resume_drafter. "
            "Iteration 1: 'Focus on keyword coverage and complete sections'. "
            "Iteration 2+: 'Address these gaps from last draft: [gap list]'. "
            "Be concrete — cite specific missing keywords or weak sections."
        ),
    )


class RetryPlannerOutput(BaseModel):
    """
    Structured output from retry_planner (Claude Opus).
    Produced when user rejects both top-2 resumes.
    """

    diff_summary: str = Field(
        ...,
        description=(
            "2–3 sentences comparing what differed between draft A and draft B. "
            "Reference specific sections, keywords, or structural choices. "
            "Do not evaluate which was 'better' — just describe the differences."
        ),
    )

    weakness_analysis: dict[str, str] = Field(
        ...,
        description=(
            "Map of section_name → specific weakness found in BOTH drafts. "
            "Example: {'summary': 'Too generic — no mention of ML frameworks', "
            "'experience[0]': 'Missing quantified impact metrics from source'}. "
            "Only cite weaknesses supported by the ATS breakdown data."
        ),
    )

    improvement_directives: list[str] = Field(
        ...,
        description=(
            "Concrete, actionable instructions for the next resume loop. "
            "Each directive must reference a specific verified_context field or "
            "a specific ATS breakdown finding. "
            "Do NOT direct the LLM to add information not in verified_context. "
            "Example: 'Lead experience[0] bullets with the 40% latency metric'."
        ),
        min_length=1,
    )

    sections_to_prioritise: list[ATSSection] = Field(
        ...,
        description=(
            "Sections that need the most improvement in the next loop. "
            "Derived from ats_breakdown.missing_sections and weakness_analysis."
        ),
    )

    user_feedback_addressed: Optional[str] = Field(
        None,
        description=(
            "How the user's rejection feedback will be addressed in the next loop. "
            "None if user provided no feedback. "
            "Quote the user's exact words then describe the specific change."
        ),
    )