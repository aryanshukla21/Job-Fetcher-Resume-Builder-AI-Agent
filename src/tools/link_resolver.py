"""
URL resolution tool for enriching internship/experience data.
Uses trafilatura for deterministic HTML-to-text extraction, and Tier 1 LLM 
for structured mapping if strict schema adherence is required.
"""
from __future__ import annotations

import httpx
import trafilatura
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from src.llms.factory import get_llm

# Initialize Tier 1 model for fast structured extraction
extraction_llm = get_llm(tier=1, temperature=0.0)

class ExtractedInternshipDetails(BaseModel):
    """Schema to map the raw webpage text into structured bullets."""
    bullets: list[str] = Field(
        description="Key responsibilities and achievements extracted from the page text."
    )
    technologies_used: list[str] = Field(
        description="Explicit tools, languages, or frameworks mentioned."
    )

extraction_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Extract the core responsibilities, achievements, and technologies "
                   "for an internship or job posting from the provided text. Keep bullets concise."),
        ("human", "PAGE TEXT:\n\n{page_text}")
    ])
    | extraction_llm.with_structured_output(ExtractedInternshipDetails)
)

def fetch_and_clean_html(url: str, timeout_seconds: int = 10) -> str | None:
    """
    Fetches URL and uses Trafilatura to deterministically strip DOM boilerplate
    (navbars, footers, scripts) down to core human-readable text.
    """
    if not url.startswith(("http://", "https://")):
        return None

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            
            # Trafilatura handles the heavy lifting of DOM manipulation natively
            extracted_text = trafilatura.extract(
                response.text, 
                include_links=False, 
                include_images=False,
                include_formatting=False
            )
            
            return extracted_text[:8000] if extracted_text else None
            
    except Exception as e:
        print(f"Failed to resolve URL {url}: {e}")
        return None

def extract_structured_internship_data(url: str) -> ExtractedInternshipDetails | None:
    """
    End-to-end pipeline: Fetches HTML -> Cleans to Text -> Extracts via LLM.
    """
    clean_text = fetch_and_clean_html(url)
    
    if not clean_text:
        return None
        
    # Now that the text is clean and small, we can safely use the LLM
    result: ExtractedInternshipDetails = extraction_chain.invoke({
        "page_text": clean_text
    })
    
    return result