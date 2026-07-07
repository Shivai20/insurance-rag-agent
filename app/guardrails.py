"""Guardrail helpers.

Maps to TCS's 'responsible agentic AI guardrails: PII detection, hallucination
prevention, behavior control'. The actual graph wiring lives in nodes.py; this
module holds the reusable checks.
"""
import re
from langchain_core.messages import HumanMessage

from app.llm import get_llm
from app import prompts, config

# --- PII patterns (illustrative; a real system would use a dedicated PII service) ---
_PII_PATTERNS = [
    (re.compile(r"\b[\w.\-]+@[\w.\-]+\.\w+\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[CARD]"),       # card-like number
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),          # US SSN format
    (re.compile(r"\b\+?\d[\d \-]{8,}\d\b"), "[PHONE]"),       # phone-like
]


def redact_pii(text: str) -> str:
    """Scrub obvious PII before it is logged or returned. Output guardrail."""
    out = text
    for pattern, token in _PII_PATTERNS:
        out = pattern.sub(token, out)
    return out


def classify(prompt_template: str, **kwargs) -> str:
    """Run a constrained classification on the FAST model and return the raw text."""
    msg = prompt_template.format(**kwargs)
    resp = get_llm(tier="fast").invoke([HumanMessage(content=msg)])
    return resp.content.strip().lower()


def is_grounded(answer: str, context: str) -> bool:
    """Faithfulness check (hallucination prevention). LLM-as-judge.

    Honest caveat to state in interview: an LLM judging an LLM reduces hallucination,
    it does not eliminate it -- and it adds a round-trip of latency/cost.
    """
    if not config.FAITHFULNESS_CHECK:
        return True
    verdict = classify(prompts.FAITHFULNESS, context=context, answer=answer)
    return verdict.startswith("yes")
