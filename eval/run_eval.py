"""Evaluation harness -- the 'evaluator bots'.

Scores each item on three axes that map to the interview talking points:
  - routing_ok   : did the agent take the right path (kb_qa / tool_call / escalate / reject)?
  - faithful     : is the answer grounded (anti-hallucination)?  [LLM-as-judge]
  - correct      : does the answer match the reference?          [LLM-as-judge]

Run:  python -m eval.run_eval
"""
import json
import os
import time

from langchain_core.messages import HumanMessage
from tabulate import tabulate

from app.graph import graph_app
from app.llm import get_llm

HERE = os.path.dirname(os.path.abspath(__file__))

JUDGE_CORRECT = """You are grading a support answer against a reference answer.
Question: {q}
Reference: {ref}
Answer: {ans}
Does the Answer convey the same key facts as the Reference (ignoring wording/extra politeness)?
Respond ONLY "yes" or "no"."""


def judge(template, **kw) -> bool:
    r = get_llm(tier="fast").invoke([HumanMessage(content=template.format(**kw))])
    return r.content.strip().lower().startswith("yes")


def infer_route(trace: list, generation: str) -> str:
    """Recover which path the run took, from the trace."""
    joined = " ".join(trace)
    if "reject" in joined or "I can only help" in generation or "insurance-related" in generation:
        return "reject"
    if "escalate" in joined or "human agent" in generation:
        return "escalate"
    if "call_tool" in joined:
        return "tool_call"
    return "kb_qa"


def main():
    with open(os.path.join(HERE, "dataset.json")) as f:
        data = json.load(f)

    rows, faithful_n, correct_n, route_n = [], 0, 0, 0
    total_latency = 0.0

    for i, item in enumerate(data, 1):
        cfg = {"configurable": {"thread_id": f"eval-{i}"}}
        t0 = time.perf_counter()
        res = graph_app.invoke({"question": item["question"]}, config=cfg)
        latency = time.perf_counter() - t0
        total_latency += latency

        gen = res.get("generation", "")
        route = infer_route(res.get("trace", []), gen)
        route_ok = route == item["expect"]

        # Only judge faithfulness/correctness for substantive answers.
        if item["expect"] in {"kb_qa", "tool_call"}:
            grounded = bool(res.get("grounded", True))
            correct = judge(JUDGE_CORRECT, q=item["question"], ref=item["reference"], ans=gen)
        else:
            grounded = correct = route_ok  # for reject/escalate, correctness == correct routing

        faithful_n += int(grounded)
        correct_n += int(correct)
        route_n += int(route_ok)

        rows.append([i, item["question"][:38] + "…",
                     f"{route}{'' if route_ok else ' (exp '+item['expect']+')'}",
                     "Y" if grounded else "N",
                     "Y" if correct else "N",
                     f"{latency:.1f}s"])

    n = len(data)
    print(tabulate(rows,
          headers=["#", "question", "route", "faithful", "correct", "latency"],
          tablefmt="github"))
    print()
    print(tabulate([
        ["Routing accuracy", f"{route_n}/{n}", f"{100*route_n/n:.0f}%"],
        ["Faithfulness",     f"{faithful_n}/{n}", f"{100*faithful_n/n:.0f}%"],
        ["Answer correctness", f"{correct_n}/{n}", f"{100*correct_n/n:.0f}%"],
        ["Avg latency / turn", f"{total_latency/n:.2f}s", ""],
    ], headers=["metric", "score", "pct"], tablefmt="github"))


if __name__ == "__main__":
    main()
