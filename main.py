"""Interactive CLI. Demonstrates multi-turn MEMORY via a stable thread_id.

Usage:
  python -m app.ingest      # once, to build the index
  python main.py            # chat
  python main.py --trace    # also print the node-by-node trace each turn
"""
import sys
import uuid

from app.graph import graph_app


def main():
    show_trace = "--trace" in sys.argv
    # --thread=<id> gives a stable conversation id, so with the Postgres
    # checkpointer you can quit, restart, and resume the same conversation.
    thread_id = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--thread=")),
                     str(uuid.uuid4()))
    print(f"(thread: {thread_id})")
    cfg = {"configurable": {"thread_id": thread_id}}

    print("AcmeInsure assistant. Type 'quit' to exit.\n")
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit"}:
            break
        if not q:
            continue
        result = graph_app.invoke({"question": q}, config=cfg)
        print(f"\nbot > {result['generation']}\n")
        if show_trace:
            print("  trace: " + "  ->  ".join(result.get("trace", [])) + "\n")


if __name__ == "__main__":
    main()
