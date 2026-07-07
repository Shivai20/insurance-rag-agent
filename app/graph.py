"""Assemble the StateGraph: nodes + conditional edges + memory checkpointer.

The checkpointer is what turns per-run STATE into cross-turn MEMORY: pass the same
thread_id and the graph resumes the prior conversation. MemorySaver is in-process;
in production you swap it for SqliteSaver / PostgresSaver / Redis -- one line.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app import config
from app.state import GraphState
from app import nodes
from app.observability import log_event


def _get_checkpointer():
    """Durable memory if DATABASE_URL is set; in-process fallback otherwise.

    Interview note: with Postgres, conversations survive restarts and any
    stateless worker replica can resume any thread_id -- that's the horizontal
    scaling story. We deliberately do NOT use PostgresSaver.from_conn_string():
    it's a context manager, so the connection would close when the `with` block
    exits, killing memory mid-chat. setup() creates the tables; it's idempotent.
    """
    if config.DATABASE_URL:
        from psycopg import Connection
        from psycopg.rows import dict_row
        from langgraph.checkpoint.postgres import PostgresSaver

        conn = Connection.connect(
            config.DATABASE_URL,
            autocommit=True,          # checkpoint writes commit immediately
            prepare_threshold=0,
            row_factory=dict_row,
        )
        saver = PostgresSaver(conn)
        saver.setup()
        log_event("memory_backend", backend="postgres")
        return saver
    log_event("memory_backend", backend="in-process MemorySaver")
    return MemorySaver()


# ---- conditional edge functions (the agent's decision points) ----
def after_input(state):
    return "reject" if state["input_flag"] != "ok" else "router"


def after_router(state):
    return {"tool_call": "call_tool", "escalate": "escalate"}.get(state["intent"], "retrieve")


def after_grade(state):
    if state.get("docs_relevant"):
        return "generate"
    if state.get("retrieve_retries", 0) < config.MAX_RETRIEVE_RETRIES:
        return "rewrite_query"      # CRAG: try a better query
    return "generate"               # give up retrying; generate will admit uncertainty


def after_output(state):
    if state.get("grounded"):
        return "finalize"
    if state.get("generate_retries", 0) < config.MAX_GENERATE_RETRIES:
        return "regenerate"         # ungrounded -> one more attempt
    return "finalize"               # accept after cap (latency guardrail)


def _bump_generate_retry(state):
    return {"generate_retries": state.get("generate_retries", 0) + 1}


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("input_guardrail", nodes.input_guardrail)
    g.add_node("router", nodes.router)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade_documents", nodes.grade_documents)
    g.add_node("rewrite_query", nodes.rewrite_query)
    g.add_node("call_tool", nodes.call_tool)
    g.add_node("generate", nodes.generate)
    g.add_node("regenerate", _bump_generate_retry)  # tiny node: count the retry, then generate
    g.add_node("output_guardrail", nodes.output_guardrail)
    g.add_node("finalize", nodes.finalize)
    g.add_node("escalate", nodes.escalate)
    g.add_node("reject", nodes.reject)

    g.add_edge(START, "input_guardrail")
    g.add_conditional_edges("input_guardrail", after_input,
                            {"reject": "reject", "router": "router"})
    g.add_conditional_edges("router", after_router,
                            {"call_tool": "call_tool", "escalate": "escalate", "retrieve": "retrieve"})
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", after_grade,
                            {"generate": "generate", "rewrite_query": "rewrite_query"})
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("call_tool", "generate")
    g.add_edge("generate", "output_guardrail")
    g.add_conditional_edges("output_guardrail", after_output,
                            {"finalize": "finalize", "regenerate": "regenerate"})
    g.add_edge("regenerate", "generate")
    g.add_edge("finalize", END)
    g.add_edge("escalate", END)
    g.add_edge("reject", END)

    # MEMORY: compile with a checkpointer so state persists across turns by thread_id.
    return g.compile(checkpointer=_get_checkpointer())


# Single shared compiled app.
graph_app = build_graph()


def ask(question: str, thread_id: str = "default") -> str:
    """Run one turn. Same thread_id => the agent remembers the conversation."""
    config_ = {"configurable": {"thread_id": thread_id}}
    result = graph_app.invoke({"question": question}, config=config_)
    return result["generation"]
