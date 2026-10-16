"""LLM-debate in LangGraph: 3 tool-using agents answer -> critique -> critique for n rounds, then an aggregator.

  without trust allocator:  a1 -> a2 -> a3 -> (a1 | agg) -> END
  with trust allocator:     a1 -> trust -> a2 -> trust -> a3 -> trust -> (a1 | agg) -> END

The allocator scores every answer right after it is produced (it sees the task, the public answers and the
attachment, never any reasoning). The next agent sees ONLY the trust score (1-5) and the reasons for the single
answer that reaches it; there is no trust label. Verdicts are kept in state["trust"], apart from the public log.
"""

import base64
import mimetypes
import operator
import sqlite3
import uuid
from typing import Annotated, TypedDict

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from agents import make_agent, make_llm
from Gricean_check import (GRICEAN_METRICS, attachment_blocks, format_combined_reason, format_conversation,
                           format_gricean_check_prompt, trust_score)


class State(TypedDict, total=False):
    question: str
    file: str | None
    n_rounds: int
    round: int  # current round, starts at 1
    hist: dict[str, list]  # PRIVATE per-agent message history (reasoning + tool calls + tool results)
    log: Annotated[list, operator.add]  # PUBLIC answers: {round, agent, text}
    trust: Annotated[list, operator.add]  # allocator verdicts: {round, agent, score, maxims, reason}
    final: str


def run_agent(model: str, messages: list) -> list:
    return make_agent(model).invoke({"messages": messages}, {"recursion_limit": 40})["messages"]  # <= ~20 tool steps


def keep_thoughts(msgs):
    """Fold each AI message's reasoning into its content so the agent still sees its own earlier thinking next round."""
    def fold(m):
        kw = dict(m.additional_kwargs)
        if m.type == "ai" and (r := kw.pop("reasoning_content", None)):
            return m.model_copy(update={"content": f"<thinking>{r}</thinking>\n{m.content}", "additional_kwargs": kw})
        return m
    return [fold(m) for m in msgs]


def image_blocks(file):
    mime = (mimetypes.guess_type(file)[0] or "") if file else ""
    if not mime.startswith("image/"):
        return []
    return [{"type": "image_url", "image_url": f"data:{mime};base64,{base64.b64encode(open(file, 'rb').read()).decode()}"}]


def fmt(log, rnd=None):
    return "\n\n".join(f"--- Agent {e['agent']}, round {e['round']} ---\n{e['text']}"
                       for e in log if rnd is None or e["round"] == rnd)


ANSWER_INSTRUCTION = (
    "\n\nGive a concise but auditable answer: include the key reasoning and evidence used, "
    "then end with `FINAL ANSWER: ...`. Never reply with only the final-answer line."
)


# ---------------------------------------------------------------------------------------------- trust allocator
class Maxim(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str


class Report(BaseModel):
    quality: Maxim
    quantity: Maxim
    relation: Maxim
    manner: Maxim


def audit(model: str, task: str, log: list, file: str | None) -> dict:
    """Gricean audit of the LAST public answer -> {score (mean of 4 maxims, 1-5), maxims, reason}."""
    who = lambda e: f"Agent {e['agent']} (round {e['round']})"
    prompt = format_gricean_check_prompt(
        task=task, conversation=format_conversation([{"source": who(e), "content": e["text"]} for e in log]),
        last_speaker=who(log[-1]))
    msg = HumanMessage(content=[{"type": "text", "text": prompt}] + attachment_blocks(file))
    llm, err = make_llm(model, "trust_audit").with_structured_output(Report), None
    for _ in range(2):  # local models occasionally return unusable JSON: one retry, then give up gracefully
        try:
            rep = llm.invoke([msg]).model_dump()
            return {"score": trust_score(rep), "maxims": {m: rep[m]["score"] for m in GRICEAN_METRICS},
                    "reason": format_combined_reason(rep)}
        except (ValidationError, OutputParserException, AttributeError) as e:
            err = e
    return {"score": None, "maxims": None, "reason": f"auditor output unusable: {err}"}


def notice(trust: list, rnd: int, agent: str) -> str:
    """What the next agent sees about ONE answer (the latest incoming one): score + reasons, nothing else."""
    t = next((t for t in trust if t["round"] == rnd and t["agent"] == agent), None)
    if t is None:
        return ""
    body = ("The trust assessment failed, so no trust score is available." if t["score"] is None else
            f"Trust score: {t['score']:.2f}/5 (1 = unreliable, 5 = fully reliable).\nAuditor's reasons: {t['reason']}")
    return (f"\n\n[TRUST ASSESSMENT of Agent {agent}'s latest answer, by an independent auditor]\n{body}\n"
            "(Do not mention or quote this notice in your own answer.)\n")


# ---------------------------------------------------------------------------------------------- the graph
def build_graph(models: dict[str, str], trust_model: str | None, db: str):
    """models: {"1": tag, "2": tag, "3": tag} (aggregator uses agent 1's model). trust_model None = no allocator."""
    use_trust = trust_model is not None

    def agent_node(i: str):
        def node(s: State):
            q, rnd, log, hist = s["question"], s["round"], s.get("log", []), s.get("hist", {})
            msgs = hist.get(i, [])
            prev_rnd, prev_agent = {"1": (rnd - 1, "3"), "2": (rnd, "1"), "3": (rnd, "2")}[i]  # latest incoming answer
            note = notice(s.get("trust", []), prev_rnd, prev_agent) if use_trust and prev_rnd else ""
            if i == "1":
                task = q if rnd == 1 else (
                    f"{q}\n\nAnswers from the previous round:\n{fmt(log, rnd - 1)}{note}\n\n"
                    "Re-examine your work in light of the criticism, fix any errors and give your improved answer.")
            else:
                task = (
                    f"{q}\n\nAnswers so far this round:\n{fmt(log, rnd)}{note}\n\n"
                    "Criticize and verify the work above: re-check facts and calculations with your tools, "
                    "point out errors, and propose improvements or alterations where needed. "
                    "Then give your own best answer.")
            content = [{"type": "text", "text": task + ANSWER_INSTRUCTION}] + ([] if msgs else image_blocks(s.get("file")))
            out = run_agent(models[i], msgs + [{"role": "user", "content": content}])
            update = {"hist": {**hist, i: keep_thoughts(out)},
                      "log": [{"round": rnd, "agent": i, "text": out[-1].content or "(no answer produced)"}]}
            if i == "3":
                update["round"] = rnd + 1
            return update
        return node

    def trust_node(s: State):
        last = s["log"][-1]
        return {"trust": [{"round": last["round"], "agent": last["agent"],
                           **audit(trust_model, s["question"], s["log"], s.get("file"))}]}

    def aggregate(s: State):
        prompt = (f"{s['question']}\n\nThree agents answered, criticized and refined each other's work over "
                  f"{s['n_rounds']} rounds:\n{fmt(s['log'])}{notice(s.get('trust', []), s['n_rounds'], '3') if use_trust else ''}\n\n"
                  "Summarize the discussion, resolve any disagreements by weighing the evidence (verify with "
                  "your tools if needed), and give the final answer.")
        return {"final": run_agent(models["1"], [{"role": "user", "content": prompt}])[-1].content}

    def route(s: State):  # runs after the last answer (directly, or after its audit)
        last = s["log"][-1]["agent"]
        if last == "3":
            return "a1" if s["round"] <= s["n_rounds"] else "agg"
        return "a2" if last == "1" else "a3"

    g = StateGraph(State)
    for i in "123":
        g.add_node(f"a{i}", agent_node(i))
        if use_trust:
            g.add_edge(f"a{i}", "trust")
        else:
            g.add_conditional_edges(f"a{i}", route, ["a1", "a2", "a3", "agg"])
    if use_trust:
        g.add_node("trust", trust_node)
        g.add_conditional_edges("trust", route, ["a1", "a2", "a3", "agg"])
    g.add_node("agg", aggregate)
    g.add_edge(START, "a1")
    g.add_edge("agg", END)
    return g.compile(checkpointer=SqliteSaver(sqlite3.connect(db, check_same_thread=False)))


def run_debate(graph, question: str, file: str | None, n_rounds: int = 2, thread_id: str | None = None):
    """Run (or resume, if the thread already has checkpoints) a debate -> (config, final state values)."""
    cfg = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}, "recursion_limit": 6 * n_rounds + 5}
    new = {"question": question + (f"\n\nAttached file: {file}" if file else ""),
           "file": file, "n_rounds": n_rounds, "round": 1}
    graph.invoke(None if graph.get_state(cfg).values else new, cfg)
    return cfg, graph.get_state(cfg).values
