"""
Hybrid ATS Scoring Engine.
Uses exact token matching for keyword metrics and LLMs for semantic profiling.
"""
from __future__ import annotations

import re
from src.llms.factory import get_llm
from src.schemas.models import ATSScoreBreakdown
from langchain_core.prompts import ChatPromptTemplate

# Resolve light tier model for fast, deterministic evaluation structures
ats_llm = get_llm(tier=1, temperature=0.0)

ats_evaluation_chain = (
    ChatPromptTemplate.from_messages([
        ("system", """You are an advanced corporate Applicant Tracking System (ATS) parser. 
Analyze the provided clean resume against the targeted job description.
Evaluate structural alignment, missing technical qualifications, and soft skills context.
Provide an objective alignment score out of 100 based entirely on the texts provided."""),
        ("human", "JOB DESCRIPTION:\n{jd}\n\nRESUME CONTENT:\n{resume}")
    ])
    | ats_llm.with_structured_output(ATSScoreBreakdown)
)

def _tokenize_and_clean(text: str) -> set[str]:
    """Extracts clean alphanumeric lowercase tokens for strict structural set calculations."""
    words = re.findall(r'\b[a-zA-Z0-9_\-\.\+#]+\b', text.lower())
    return set(words)

def evaluate_ats_compliance(resume_text: str, jd_text: str, target_keywords: list[str]) -> tuple[float, list[str]]:
    """
    Computes a hybrid evaluation score.
    Returns a unified numerical score along with an array of verified keyword matches.
    """
    # 1. Deterministic Token Search (Prevents LLM math/tokenization errors)
    resume_tokens = _tokenize_and_clean(resume_text)
    matched_keywords = []
    
    for kw in target_keywords:
        kw_clean = kw.lower().strip()
        # Handle exact word boundaries or direct sub-token intersections
        if kw_clean in resume_tokens or any(kw_clean in tok for tok in resume_tokens):
            matched_keywords.append(kw)

    # 2. Semantic Analysis Phase executed via LLM
    try:
        llm_breakdown: ATSScoreBreakdown = ats_evaluation_chain.invoke({
            "jd": jd_text,
            "resume": resume_text
        })
        
        # Merge physical metrics with structural analysis values safely
        semantic_score = llm_breakdown.overall_score
    except Exception as e:
        print(f"LLM ATS evaluation sub-node failed: {e}. Falling back to keyword ratio.")
        semantic_score = (len(matched_keywords) / len(target_keywords) * 100) if target_keywords else 50.0

    # 3. Formulate structural final weighted score
    keyword_coverage_ratio = len(matched_keywords) / len(target_keywords) if target_keywords else 1.0
    weighted_score = (semantic_score * 0.6) + ((keyword_coverage_ratio * 100) * 0.4)
    
    return round(weighted_score, 2), matched_keywords