from typing import Optional

from knowledge_rag.app.providers.llm_interface import LLMClient

from .anthropic_client import AnthropicAIClient
from .openai_client import OpenAIClient


def get_llm_client(provider: str, model: Optional[str] = None) -> LLMClient:
    if provider == "OpenAI":
        return OpenAIClient(model)
    elif provider == "Anthropic":
        return AnthropicAIClient(model)
    else:
        raise ValueError(f"Invalid provider: {provider}")
