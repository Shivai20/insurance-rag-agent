"""The shared graph state.

Interview distinction to draw clearly:
  - STATE (this object) = scratchpad for ONE run, passed node-to-node.
  - MEMORY = the checkpointer persisting state across turns by thread_id (see graph.py).

`messages` uses the add_messages reducer so each turn appends rather than overwrites
-- that is what gives the agent conversational memory.
"""
from typing import Annotated, Literal, TypedDict
from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    # Conversation memory (persisted by the checkpointer across turns)
    messages: Annotated[list, add_messages]

    # Per-run scratch fields
    question: str                       # current user question
    input_flag: Literal["ok", "reject_offtopic", "reject_injection"]
    intent: Literal["kb_qa", "tool_call", "escalate"]
    tool_name: str                      # which tool the router picked
    tool_arg: str                       # argument extracted for the tool
    documents: list                     # retrieved chunks
    docs_relevant: bool                 # output of the grader
    tool_result: str                    # output of a tool call
    generation: str                     # the drafted answer
    grounded: bool                      # output guardrail verdict
    retrieve_retries: int
    generate_retries: int
    trace: list                         # human-readable step log for the UI/walkthrough
