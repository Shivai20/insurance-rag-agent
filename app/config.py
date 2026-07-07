"""Central configuration. One place to change models, paths, and limits.

Interview note: keeping config in one module (not scattered os.getenv calls)
is what lets you swap providers / vector stores / limits without touching logic.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Models (tiering: cheap model for the many small calls, smart for final answer) ---
FAST_MODEL = os.getenv("FAST_MODEL", "gpt-4o-mini")    # router, grading, guardrails
SMART_MODEL = os.getenv("SMART_MODEL", "gpt-4o-mini")  # final answer generation
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# --- Paths ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(ROOT, "data", "kb")
CHROMA_DIR = os.path.join(ROOT, "data", "chroma")
COLLECTION = "acme_kb"

# --- RAG params ---
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 4

# --- Agentic loop limits (these caps are your latency / cost guardrails) ---
MAX_RETRIEVE_RETRIES = 1   # how many times to rewrite the query + re-retrieve
MAX_GENERATE_RETRIES = 1   # how many times to regenerate if answer is not grounded

# --- Behaviour flags ---
FAITHFULNESS_CHECK = True   # output guardrail: verify answer is grounded in context

# Mandatory compliance line appended to every substantive answer.
COMPLIANCE_DISCLAIMER = (
    "This is general information from AcmeInsure and not financial or legal advice; "
    "for decisions specific to your policy, please confirm with a human agent."
)

# --- Memory backend: Postgres checkpointer if set, else in-process MemorySaver ---
DATABASE_URL = os.getenv("DATABASE_URL", "")
