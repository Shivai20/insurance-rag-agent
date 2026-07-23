"""LLM / embeddings factory.
"""
from functools import lru_cache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app import config


@lru_cache(maxsize=4)
def get_llm(tier: str = "fast", temperature: float = 0.0) -> ChatOpenAI:
    """tier='fast' for cheap high-frequency calls (router/grade/guardrail),
    tier='smart' for the final customer-facing answer."""
    model = config.SMART_MODEL if tier == "smart" else config.FAST_MODEL
    return ChatOpenAI(model=model, temperature=temperature)


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=config.EMBED_MODEL)
