"""
Document parsing tool for base resumes.
Combines structural extraction via python-docx with Haiku structured output 
to guarantee compliance with Pydantic domain models.
"""

from __future__ import annotations

import docx
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src.schemas.models import ExperienceEntry, EducationEntry, InternshipEntry
from src.schemas.structured_outputs import haiku

class ParsedResumeData(BaseModel):
    """Internal model used to extract arrays of validated objects via LLM."""
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    internships: list[InternshipEntry] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)

# Tier 1 (Haiku) is perfect for fast, cheap, and reliable schema projection
extraction_chain = (
    ChatPromptTemplate.from_messages([
        ("system", """You are an expert data extractor. Convert the raw resume text 
into the exact structured schema provided. 
- Do NOT invent missing data. 
- If an end date is not specified or says 'Present', set is_current=True and end_date=null. 
- Extract explicit technologies mentioned in the bullets into the technologies list.
- Keep bullets verbatim."""),
        ("human", "RAW RESUME TEXT:\n\n{resume_text}")
    ])
    | haiku.with_structured_output(ParsedResumeData)
)

def extract_raw_text_from_docx(file_path: str) -> str:
    """Extracts text while preserving basic semantic section boundaries."""
    try:
        doc = docx.Document(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read docx file at {file_path}: {e}")

    full_text = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
            
        # Highlight structural boundaries (headings/bolding) for the LLM
        if para.style.name.startswith('Heading') or (para.runs and para.runs[0].bold):
            full_text.append(f"\n[{text.upper()}]")
        else:
            full_text.append(text)
            
    return "\n".join(full_text)

def parse_resume(file_path: str) -> ParsedResumeData:
    """
    Reads a .docx resume and safely parses it into validated Pydantic models.
    """
    raw_text = extract_raw_text_from_docx(file_path)
    
    # Offload the schema projection to Haiku to avoid fragile Regex maintenance
    parsed_data: ParsedResumeData = extraction_chain.invoke({
        "resume_text": raw_text
    })
    
    return parsed_data