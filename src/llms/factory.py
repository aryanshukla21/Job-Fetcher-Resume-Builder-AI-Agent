"""
Factory for instantiating LLMs dynamically based on configuration.
Ensures we are not locked into a single vendor.
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from src.config import settings

def get_llm(tier: int, temperature: float = 0.0) -> BaseChatModel:
    """
    Returns the appropriate ChatModel instance based on the configured provider and tier.
    """
    provider = settings.llm_provider.lower()
    
    # Determine the model string based on the requested tier
    if tier == 1:
        model_name = settings.tier1_model
    elif tier == 2:
        model_name = settings.tier2_model
    elif tier == 3:
        model_name = settings.tier3_model
    else:
        raise ValueError(f"Invalid LLM tier: {tier}. Must be 1, 2, or 3.")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name, 
            temperature=temperature, 
            api_key=settings.anthropic_api_key
        )
        
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name, 
            temperature=temperature, 
            api_key=settings.openai_api_key
        )
        
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=temperature, 
            api_key=settings.google_api_key
        )
        
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")