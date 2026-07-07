"""LangGraph nodes. Each node is a pure function: state in -> partial state out.

Flow (the 'agentic mesh'):
  input_guardrail -> router -> {retrieve|tool|escalate}
  retrieve -> grade -> {generate | rewrite->retrieve}
  tool -> generate
  generate -> output_guardrail -> {END | regenerate}
This is the Corrective-RAG (CRAG) self-correction pattern plus routing + tool-use.
"""
from langchain_core.messages import AIMessage, HumanMessage

from app import config, prompts
from app.observability import timed_node, log_event
from app.retriever import retrieve as vector_retrieve
from app.tools import TOOLS
from app.guardrails import classify, is_grounded, redact_pii
from app.llm import get_llm


def _format_docs(docs) -> str:
    return "\n\n".join(f"[{d.metadata.get('source','?')}] {d.page_content}" for d in docs)


def _note(state, msg: str) -> list:
    return state.get("trace", []) + [msg]


# ---------------------------------------------------------------- guardrail (input)
@timed_node("input_guardrail")
def input_guardrail(state):
    q = state["question"]
    label = classify(prompts.INPUT_GUARDRAIL, question=q)
    flag = "ok"
    if "injection" in label:
        flag = "reject_injection"
    elif "offtopic" in label:
        flag = "reject_offtopic"
    log_event("guardrail_input", flag=flag)
    return {"input_flag": flag,
            "retrieve_retries": 0, "generate_retries": 0,
            "trace": _note(state, f"input_guardrail -> {flag}")}


# ---------------------------------------------------------------- router
@timed_node("router")
def router(state):
    import json
    raw = classify(prompts.ROUTER, question=state["question"])
    intent, tool_name, tool_arg = "kb_qa", "", ""
    try:
        data = json.loads(raw)
        intent = data.get("intent", "kb_qa")
        tool_name = data.get("tool_name", "") or ""
        tool_arg = data.get("tool_arg", "") or ""
    except Exception:
        # Robustness: if the small model returns malformed JSON, default to KB Q&A.
        log_event("router_parse_fallback", raw=raw)
    log_event("router", intent=intent, tool=tool_name)
    return {"intent": intent, "tool_name": tool_name, "tool_arg": tool_arg,
            "trace": _note(state, f"router -> {intent} {tool_name}".strip())}


# ---------------------------------------------------------------- retrieve
@timed_node("retrieve")
def retrieve(state):
    docs = vector_retrieve(state["question"])
    log_event("retrieve", k=len(docs),
              sources=[d.metadata.get("source") for d in docs])
    return {"documents": docs, "trace": _note(state, f"retrieve -> {len(docs)} chunks")}


# ---------------------------------------------------------------- grade documents
@timed_node("grade_documents")
def grade_documents(state):
    verdict = classify(prompts.GRADE_DOCS,
                       question=state["question"],
                       documents=_format_docs(state["documents"]))
    relevant = verdict.startswith("yes")
    log_event("grade_documents", relevant=relevant)
    return {"docs_relevant": relevant,
            "trace": _note(state, f"grade -> {'relevant' if relevant else 'weak'}")}


# ---------------------------------------------------------------- rewrite query (CRAG)
@timed_node("rewrite_query")
def rewrite_query(state):
    new_q = get_llm(tier="fast").invoke(
        [HumanMessage(content=prompts.REWRITE.format(question=state["question"]))]
    ).content.strip()
    retries = state.get("retrieve_retries", 0) + 1
    log_event("rewrite_query", retries=retries, new_question=new_q)
    return {"question": new_q, "retrieve_retries": retries,
            "trace": _note(state, f"rewrite_query -> '{new_q}'")}


# ---------------------------------------------------------------- tool call
@timed_node("call_tool")
def call_tool(state):
    name, arg = state.get("tool_name"), state.get("tool_arg", "")
    fn = TOOLS.get(name)
    result = fn(arg) if fn else f"Unknown tool '{name}'."
    return {"tool_result": result, "trace": _note(state, f"call_tool {name}({arg})")}


# ---------------------------------------------------------------- generate
@timed_node("generate")
def generate(state):
    # Context is either retrieved KB chunks or a tool result.
    if state.get("tool_result"):
        context = state["tool_result"]
    else:
        context = _format_docs(state.get("documents", []))
    answer = get_llm(tier="smart").invoke(
        [HumanMessage(content=prompts.GENERATE.format(
            context=context, question=state["question"]))]
    ).content.strip()
    log_event("generate", chars=len(answer))
    return {"generation": answer, "trace": _note(state, "generate -> draft")}


# ---------------------------------------------------------------- guardrail (output)
@timed_node("output_guardrail")
def output_guardrail(state):
    context = state.get("tool_result") or _format_docs(state.get("documents", []))
    grounded = is_grounded(state["generation"], context)
    log_event("guardrail_output", grounded=grounded)
    return {"grounded": grounded,
            "trace": _note(state, f"output_guardrail -> {'grounded' if grounded else 'ungrounded'}")}


# ---------------------------------------------------------------- finalize / terminal nodes
def _finalize(state, text: str):
    """Scrub PII, append disclaimer, and write the turn into conversation memory."""
    safe = redact_pii(text)
    if state.get("intent") == "kb_qa" or state.get("tool_result"):
        safe = f"{safe}\n\n_{config.COMPLIANCE_DISCLAIMER}_"
    return {"generation": safe,
            "messages": [HumanMessage(content=state["question"]), AIMessage(content=safe)]}


@timed_node("finalize")
def finalize(state):
    return _finalize(state, state["generation"])


@timed_node("escalate")
def escalate(state):
    msg = ("I'm creating a ticket for a human agent, who will follow up within one "
           "business day. Is there anything else I can help with in the meantime?")
    return _finalize(state, msg)


@timed_node("reject")
def reject(state):
    if state.get("input_flag") == "reject_injection":
        msg = "I can only help with AcmeInsure policies, claims, billing, and account questions."
    else:
        msg = "I'm AcmeInsure's assistant, so I can only help with insurance-related questions."
    return {"generation": msg,
            "messages": [HumanMessage(content=state["question"]), AIMessage(content=msg)]}
