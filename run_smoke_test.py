"""Fast smoke test -- run this FIRST to catch env/key/version breakage in seconds,
not three hours in.

  python run_smoke_test.py
"""
import os
import sys


def main():
    # 1. Key present?
    if not os.getenv("OPENAI_API_KEY"):
        # dotenv is loaded by app.config import below, so re-check after import
        from dotenv import load_dotenv
        load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("FAIL: OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)
    print("ok: API key present")

    # 2. Imports resolve?
    try:
        from app import config
        from app.ingest import build_index
        from app.graph import ask
    except Exception as e:
        print(f"FAIL: import error -> {e}")
        sys.exit(1)
    print("ok: imports resolve")

    # 3. Index exists? Build if not.
    if not os.path.exists(config.CHROMA_DIR):
        print("... no index found, building (one-time, hits the embeddings API)")
        build_index()
    print("ok: vector index ready")

    # 4. One end-to-end question.
    print("... asking a test question")
    answer = ask("What does Premium auto cover?", thread_id="smoke")
    assert isinstance(answer, str) and len(answer) > 0, "empty answer"
    print("ok: end-to-end answer received\n")
    print("---- sample answer ----")
    print(answer)
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
