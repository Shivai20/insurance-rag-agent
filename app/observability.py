"""Observability & logging.

  1. Structured (JSON) logs per step, not print().
  2. Per-node latency so we can point at the bottleneck.
  
If LANGCHAIN_TRACING_V2=true and a LANGCHAIN_API_KEY is set, LangSmith also
captures full traces automatically with zero extra code.
"""
import json
import logging
import sys
import time
from functools import wraps

logger = logging.getLogger("acme")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def log_event(event: str, **fields):
    """Emit one structured JSON log line."""
    logger.info(json.dumps({"event": event, **fields}, default=str))


def timed_node(name: str):
    """Decorator for LangGraph nodes: logs entry, exit, and latency in ms.

    The latency log line per node is what lets you say in an interview:
    'retrieval is 80ms, the faithfulness check is 1.2s -- here is my bottleneck.'
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(state, *args, **kwargs):
            start = time.perf_counter()
            log_event("node_start", node=name)
            result = fn(state, *args, **kwargs)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
            log_event("node_end", node=name, latency_ms=elapsed_ms)
            return result
        return wrapper
    return decorator
