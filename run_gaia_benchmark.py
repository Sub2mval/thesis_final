#!/usr/bin/env python3
"""Run GAIA validation tasks through LLM-debate (local Ollama models) with / without the trust allocator,
and inject an FM-2.x error into message 1 (Agent 1's first answer), identically for every condition.

    python run_gaia_benchmark.py --n-questions 5                       # Ollama Cloud, gemma4:31b everywhere (default)
    python run_gaia_benchmark.py --backend local --task-ids <id> --agent-models gemma4:31b gemma4:12b qwen3.5:9b --trust-model muse-glimmer:30b

Per task this gives the 4 conditions of the experiment (slide 6):
    <output-dir>/<task_id>/baseline_off.{json,md,full.md}   1  clean run, no trust allocator
    <output-dir>/<task_id>/baseline_on.{json,md,full.md}    2  clean run, with trust allocator
    <output-dir>/<task_id>/fork_off.{json,md,full.md}       3  same start, but message 1 replaced by its corrupted copy, no allocator
    <output-dir>/<task_id>/fork_on.{json,md,full.md}        4  same as 3, with allocator
Message 1 and its corruption are generated once per task and shared by all four conditions.
.md = concise public trace; .full.md = + every model call (inputs, reasoning, outputs). Then run: python analyze.py <results>

Setup: .env next to this script with TAVILY_API_KEY[_1, _2, ...] and OLLAMA_API_KEY[_1, _2, ...] (several keys rotate) and, for the ONE-TIME GAIA download only, HF_TOKEN.
A local Ollama server must have every model pulled (`ollama pull <tag>`). pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dotenv import load_dotenv

import agents
from agents import AGENT_MODELS, TRUST_MODELS
from gaia_data import ensure_gaia, extract_answer, gaia_score, load_tasks


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------------------------------
# Trace data collection: every LLM call (tokens, timing, input, reasoning, output), split into phases:
# "shared" (message 1 + the injector, once per task), then "clean_<on|off>" and "fork_<on|off>" per allocator setting.
# --------------------------------------------------------------------------------------------------
def msg_dict(m: Any) -> dict:
    """LangChain message / dict -> plain dict (images dropped, reasoning and tool calls kept)."""
    if isinstance(m, dict):
        d = dict(m)
    else:
        d = {"role": getattr(m, "type", type(m).__name__), "content": getattr(m, "content", str(m))}
        if getattr(m, "tool_calls", None):
            d["tool_calls"] = m.tool_calls
        if getattr(m, "name", None):
            d["name"] = m.name
        r = (getattr(m, "additional_kwargs", None) or {}).get("reasoning_content")
        if r:
            d["reasoning"] = r
    if isinstance(d.get("content"), list):
        d["content"] = [{**b, "image_url": "<image omitted>"} if isinstance(b, dict) and b.get("type") == "image_url" else b
                        for b in d["content"]]
    return d


class Recorder:
    def __init__(self) -> None:
        self.lock, self.verbose = threading.Lock(), True
        self.reset()

    def reset(self) -> None:
        self.calls: list[dict] = []
        self.phase, self.started = "shared", {}
        self.begin("shared")

    def begin(self, phase: str) -> None:
        self.phase, self.started[phase] = phase, now_iso()

    def add(self, **kw: Any) -> None:
        with self.lock:
            self.calls.append({"call_index": len(self.calls) + 1, "phase": self.phase, **kw})
            c = self.calls[-1]
        if self.verbose:
            log(f"      llm call #{c['call_index']} {c['call_type']:<11} {c['model']:<17} {c['elapsed_seconds']:6.1f}s  "
                f"tokens in/out {c['prompt_tokens']}/{c['completion_tokens']}" + ("" if c["status"] == "ok" else f"  FAILED: {c['error']}"))

    def stats(self, phase: str, prior: list = ()) -> dict:
        """Token / time stats of one phase; `prior` = calls of an earlier, interrupted attempt of the same phase (resumed runs)."""
        calls = list(prior) + [{k: v for k, v in c.items() if k != "phase"} for c in self.calls if c["phase"] == phase]
        ok = [c for c in calls if c["status"] == "ok"]
        pt, ct = sum(c["prompt_tokens"] or 0 for c in ok), sum(c["completion_tokens"] or 0 for c in ok)
        return {"n_llm_calls": len(calls), "n_successful_calls": len(ok), "n_failed_attempts": len(calls) - len(ok),
                "prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct,
                "total_elapsed_seconds": sum(c["elapsed_seconds"] for c in calls),
                "calls": calls}


def install_recorder(rec: Recorder, retries: int = 2) -> None:
    """Wrap ChatOllama._generate once: every call (agents, trust allocator, injector) is recorded, and network-level
    failures (dropped / timed-out connection) are retried. The call type comes from the `role` metadata (agents.make_llm)."""
    import httpx
    from langchain_ollama import ChatOllama

    orig = ChatOllama._orig_generate = getattr(ChatOllama, "_orig_generate", ChatOllama._generate)  # idempotent: re-installing re-wraps the original

    def _generate(self, messages, *a, **kw):
        kind, model = (self.metadata or {}).get("role", "llm_call"), self.model
        inp = [msg_dict(m) for m in messages]
        for attempt in range(retries + 1):
            t0 = time.time()
            try:
                res = orig(self, messages, *a, **kw)
                break
            except Exception as e:
                retryable = isinstance(e, (httpx.TransportError, ConnectionError, TimeoutError))
                rec.add(call_type=kind, provider="ollama", model=model, elapsed_seconds=time.time() - t0, prompt_tokens=None,
                        completion_tokens=None, status="error", error=f"{type(e).__name__}: {e}", input=inp, output=None, reasoning=None)
                if not retryable or attempt == retries:
                    raise
                time.sleep(min(5 * 2 ** attempt, 30))
        m = res.generations[0].message
        out = msg_dict(m)
        um = getattr(m, "usage_metadata", None) or {}
        done = agents.done_reason(res)  # "length" = cut off (output cap or context window full)
        if done == "length":
            log(f"      !! {model} was cut off at the context limit (done_reason=length); consider a larger --num-ctx")
        rec.add(call_type=kind, provider="ollama", model=model, elapsed_seconds=time.time() - t0,
                prompt_tokens=um.get("input_tokens"), completion_tokens=um.get("output_tokens"), status="ok", error=None,
                input=inp, output=out, reasoning=out.get("reasoning"), done_reason=done)
        return res

    ChatOllama._generate = _generate


# --------------------------------------------------------------------------------------------------
# Records. Every public message is judged like the final answer: its FINAL ANSWER is scored against the GAIA
# ground truth (step_correct). Trust verdicts are kept separately (trust_history), joined by message_index.
# --------------------------------------------------------------------------------------------------
def public_messages(state: dict, truth: str) -> list[dict]:
    out = []
    for i, e in enumerate(state.get("log", [])):
        ans = extract_answer(e["text"])
        out.append({"log_index": i, "source": f"Agent{e['agent']}", "round": e["round"], "content": e["text"],
                    "answer": ans, "step_correct": gaia_score(ans, truth)})
    return out


def trust_history(state: dict) -> list[dict]:
    log, out = state.get("log", []), []
    for step, t in enumerate(state.get("trust", [])):
        idx = next((i for i, e in enumerate(log) if e["round"] == t["round"] and e["agent"] == t["agent"]), None)
        out.append({"step": step, "message_index": idx, "evaluated_source": f"Agent{t['agent']}", "agent_id": int(t["agent"]) - 1,
                    "trust_score": t["score"], "maxim_scores": t["maxims"], "reason": t["reason"]})
    return out


def build_record(*, task: dict, trust: bool, state: dict, condition: str, thread_id: str, stats: dict, started: str,
                 args: argparse.Namespace, injection: Optional[dict] = None, error: Optional[str] = None) -> dict:
    final = state.get("final")
    extracted = extract_answer(final)
    return {
        "task_id": task["task_id"], "question": task["Question"], "ground_truth": task["Final answer"], "level": task["Level"],
        "attached_file": task["file"], "thread_id": thread_id, "condition": condition, "trust_allocator": trust,
        "agent_models": dict(zip("123", args.agent_models)), "trust_model": args.trust_model,
        "injector_model": args.injector_model, "backend": args.backend, "temperature": args.temperature, "seed": args.seed, "n_rounds": args.n_rounds,
        "final_answer_raw": final, "final_answer_extracted": extracted,
        "correct": gaia_score(extracted, task["Final answer"]) if final is not None else False,
        "messages": public_messages(state, task["Final answer"]), "trust_history": trust_history(state),
        "injection": injection, "error": error, "token_stats": stats, "started_at": started, "finished_at": now_iso(),
        "contexts": {f"Agent{a}": [msg_dict(m) for m in ms] for a, ms in (state.get("hist") or {}).items()},
    }


def injection_record(inj: dict) -> dict:
    """The shared injection without the (non-serialisable) corrupted agent context."""
    return {k: v for k, v in inj.items() if k != "first"}


# --------------------------------------------------------------------------------------------------
# Markdown + JSON output
# --------------------------------------------------------------------------------------------------
def _fence(text: Any) -> str:
    return f"````\n{text if text is not None else '(none)'}\n````"


def render_md(title: str, r: dict, full: bool = False, prior: tuple = ()) -> str:
    """Public report; full=True adds every model call (`prior` = [(heading, calls)] of earlier phases, e.g. the shared ones)."""
    inj = r.get("injection") or {}
    o = [f"# {title}" + (" - full trace" if full else ""), ""]
    if r.get("error"):
        o += [f"> **RUN FAILED:** `{r['error']}`", ">",
              "> Everything below is the partial trace up to the failure. Rerun the same command: only the failed part is repeated.", ""]
    o += [
         f"**Task ID:** `{r['task_id']}`", f"**Condition:** `{r['condition']}`",
         f"**Trust allocator:** {'ON' if r['trust_allocator'] else 'OFF'}",
         f"**Agent models:** {r['agent_models']} | **Trust model:** {r['trust_model'] if r['trust_allocator'] else '-'}",
         f"**Ground truth:** {r['ground_truth']}", f"**Final answer:** {r['final_answer_extracted']}",
         f"**Correct:** {r['correct']}", "", "## Initial question", "", r["question"]]
    if r.get("attached_file"):
        o += ["", f"**Attachment:** `{r['attached_file']}`"]
    if inj:
        o += ["", "### Injected error (message 1, identical in all injected conditions of this task)", "",
              f"- Eligibility: **{'ELIGIBLE' if inj['eligible'] else 'INELIGIBLE'}**"]
        o += [f"- Failure mode chosen by the injector: `{inj['fm_id']}` - {inj['fm_name']}", f"- Injector's reason: {inj['reason'] or '(none)'}"] \
            if inj["eligible"] else ["- No corrupted message was inserted; this record repeats the clean run."]
    trust = {t["message_index"]: t for t in r["trust_history"]}
    o += ["", "---", "", "## Public flow", ""]
    for m in r["messages"]:
        mark = " - **CORRUPTED (shared injection)**" if inj.get("log_index") == m["log_index"] else ""
        o += [f"### [{m['log_index'] + 1}] {m['source']} - round {m['round']}{mark}", "", _fence(m["content"]), "",
              f"**Step accuracy:** {'correct' if m['step_correct'] else 'incorrect'} (answer: `{m['answer']}`)", ""]
        if m["log_index"] in trust:
            t = trust[m["log_index"]]
            o += [f"**Trust allocator output -> score {t['trust_score']}/5**", "", t["reason"], ""]
    if full:
        o += ["---", "", "## Model-call trace", ""]
        phases = [*prior, ("THIS RUN", r["token_stats"]["calls"])]
        titles = {"agent_call": "Agent / model call", "trust_audit": "Trust allocator", "corruption": "Error injector"}
        for name, calls in phases:
            o += [f"## {name}", ""] if calls else []
            for i, c in enumerate(calls, 1):
                o += [f"### Call {i} - {titles.get(c['call_type'], c['call_type'])}", "", f"**Model:** {c['model']}",
                      f"**Status:** {c['status']}", f"**Elapsed:** {c['elapsed_seconds']:.1f}s"]
                o += [f"**Error:** {c['error']}"] if c.get("error") else []
                o += ["", "#### Input", "", _fence(json.dumps(c["input"], indent=2, ensure_ascii=False, default=str))]
                o += ["", "#### Reasoning", "", _fence(c["reasoning"])] if c.get("reasoning") else []
                o += ["", "#### Output", "", _fence(json.dumps(c["output"], indent=2, ensure_ascii=False, default=str))]
    s = r["token_stats"]
    o += ["---", "", "## Final answer", "", _fence(r.get("final_answer_raw") or "(none)"), "", "## Run summary", "",
          f"- LLM calls: {s['n_llm_calls']} ({s['n_failed_attempts']} failed attempts)",
          f"- Tokens: {s['prompt_tokens']} prompt + {s['completion_tokens']} completion = {s['total_tokens']}",
          f"- LLM time: {s['total_elapsed_seconds']:.1f}s"]
    return "\n".join(o) + "\n"


def write_outputs(task_dir: Path, name: str, rec: dict, prior: tuple = ()) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / f"{name}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (task_dir / f"{name}.md").write_text(render_md(name, rec), encoding="utf-8")
    (task_dir / f"{name}.full.md").write_text(render_md(name, rec, full=True, prior=prior), encoding="utf-8")


def summary_row(r: dict) -> dict:
    return {"task_id": r["task_id"], "level": r["level"], "condition": r["condition"],
            "agent_models": "+".join(r["agent_models"].values()), "trust_model": r["trust_model"],
            "fm_id": (r["injection"] or {}).get("fm_id"), "eligible": (r["injection"] or {}).get("eligible", ""),
            "ground_truth": r["ground_truth"], "answer": r["final_answer_extracted"], "correct": r["correct"], "error": r["error"] or ""}


def write_summary(out_dir: Path) -> list[dict]:
    """summary.csv rebuilt from every record on disk (so resumed / repeated runs never duplicate rows)."""
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("*/*_o[nf]*.json"))]
    rows = [summary_row(r) for r in recs if "condition" in r]
    if rows:
        with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


def load_record(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def finished(rec: Optional[dict]) -> bool:
    """A record that needs no more work: no error and a final answer."""
    return bool(rec) and not rec.get("error") and rec.get("final_answer_raw") is not None


# --------------------------------------------------------------------------------------------------
# One task. Everything is resumable, so a failure costs only the failed part on the next run:
#   shared.json  message 1 and its corruption (written as soon as each exists, never regenerated)
#   <cond>.json  finished conditions are skipped; a failed one is resumed from its LangGraph checkpoint (deterministic thread
#                id in work_dir/gaia.db): the failed agent turn and what follows are redone, completed turns are not
# --------------------------------------------------------------------------------------------------
def save_shared(path: Path, first: dict, inj: Optional[dict], calls: list, started: str) -> None:
    from langchain_core.messages import messages_to_dict

    ser = lambda f: f and {"text": f["text"], "hist": messages_to_dict(f["hist"])}
    data = {"first": ser(first), "injection": inj and {k: v for k, v in inj.items() if k != "first"},
            "bad_first": inj and ser(inj["first"]), "calls": calls, "started_at": started}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")


def load_shared(path: Path) -> Optional[dict]:
    from langchain_core.messages import messages_from_dict

    d = load_record(path)
    if not d or not d.get("first"):
        return None
    de = lambda f: f and {"text": f["text"], "hist": messages_from_dict(f["hist"])}
    inj = d.get("injection") and {**d["injection"], "first": de(d.get("bad_first"))}
    return {"first": de(d["first"]), "inj": inj, "calls": d.get("calls", []), "started_at": d.get("started_at")}


def run_task(task: dict, args: argparse.Namespace, rec: Recorder, out_dir: Path, work_dir: Path) -> bool:
    """-> False if any condition of this task is (still) failed. Raises agents.KeysExhausted after saving what exists."""
    from debate import build_graph, first_answer, run_debate
    from error_injection import make_injection

    settings = {"off": [False], "on": [True], "both": [False, True]}[args.trust]
    task_dir, models, common = out_dir / task["task_id"], dict(zip("123", args.agent_models)), dict(task=task, args=args)
    name = lambda kind, trust: f"{kind}_{'on' if trust else 'off'}"
    old = {(k, t): None if args.fresh else load_record(task_dir / f"{name(k, t)}.json") for k in ("baseline", "fork") for t in settings}
    todo = [t for t in settings if not all(finished(old[(k, t)]) for k in ("baseline", "fork"))]
    if not todo:
        log("  skip: all conditions already finished")
        return True
    salt = f"::fresh{int(time.time())}" if args.fresh else ""  # --fresh: new checkpoint threads, nothing resumed
    rec.reset()

    # ---- shared part: message 1 and its corruption, saved as soon as each exists
    sh = None if args.fresh else load_shared(task_dir / "shared.json")
    prior_shared = (sh or {}).get("calls", [])
    try:
        first = sh["first"] if sh else first_answer(models["1"], task["Question"], task["file"])
        if not sh:
            save_shared(task_dir / "shared.json", first, None, rec.stats("shared")["calls"], rec.started["shared"])
        inj = sh["inj"] if sh and sh["inj"] else make_injection(task["Question"], task["file"], first, args.injector_model)
        if not (sh and sh["inj"]):
            save_shared(task_dir / "shared.json", first, inj, rec.stats("shared", prior_shared)["calls"], (sh or {}).get("started_at") or rec.started["shared"])
        log(f"  shared: message 1 {'reused' if sh and sh['inj'] else 'made'}; injection {inj['fm_id'] if inj['eligible'] else 'INELIGIBLE (no injected runs)'}")
    except agents.KeysExhausted:
        raise
    except Exception as e:
        log(f"  ERROR (shared part): {type(e).__name__}: {e}")
        traceback.print_exc()
        r = build_record(state={}, condition=name("baseline", todo[0]), trust=todo[0], thread_id="-", stats=rec.stats("shared", prior_shared),
                         started=rec.started["shared"], error=f"{type(e).__name__}: {e}", **common)
        write_outputs(task_dir, r["condition"], r)
        return False
    shared = [("SHARED: message 1 + error injection (identical in all 4 conditions)", rec.stats("shared", prior_shared)["calls"])]

    ok, exhausted = True, None
    for trust in todo:
        graph = build_graph(models, args.trust_model if trust else None, str(work_dir / "gaia.db"))
        recs = {}
        for kind, start in (("baseline", first), ("fork", inj["first"])):
            cond = name(kind, trust)
            prev = old[(kind, trust)]
            if finished(prev):  # done in an earlier run: reuse, only needed as context for the fork's md
                recs[kind] = prev
                continue
            if exhausted:  # the keys ran out earlier in this task: leave this condition untouched, it resumes next run
                ok = False
                continue
            tid, phase, err, state = f"{task['task_id']}::{cond}{salt}", cond, None, {}
            prior_calls = ((prev or {}).get("token_stats") or {}).get("calls", [])  # work already spent by a failed attempt
            rec.begin(phase)
            if start is None:  # ineligible injection: the fork record repeats the clean run (flagged by injection.eligible)
                r = {**recs["baseline"], "condition": cond, "injection": injection_record(inj),
                     "error": recs["baseline"].get("error"), "token_stats": rec.stats(phase)}
                r.pop("_state", None)
                write_outputs(task_dir, cond, r, prior=shared)
                recs[kind] = r
                continue
            else:
                try:
                    _, state = run_debate(graph, task["Question"], task["file"], args.n_rounds, tid, first=start)
                except Exception as e:  # keep whatever was reached (checkpointed state) and report the failure
                    err = f"{type(e).__name__}: {e}"
                    exhausted = str(e) if isinstance(e, agents.KeysExhausted) else None
                    log(f"  ERROR {cond}: {err}")
                    traceback.print_exc()
                    state = graph.get_state({"configurable": {"thread_id": tid}}).values or {}
            r = build_record(state=state, condition=cond, trust=trust, thread_id=tid, stats=rec.stats(phase, prior_calls),
                             started=(prev or {}).get("started_at") or rec.started[phase],
                             injection=injection_record(inj) if kind == "fork" else None, error=err, **common)
            clean_calls = (recs.get("baseline") or {}).get("token_stats", {}).get("calls", [])
            write_outputs(task_dir, cond, r, prior=shared + ([("CLEAN RUN", clean_calls)] if kind == "fork" else []))
            recs[kind] = {**r, "_state": state}
            ok &= not err
            log(f"  {cond:12s}: answer={r['final_answer_extracted']!r} correct={r['correct']}" + (f" ERROR={err}" if err else ""))
    if exhausted:
        raise agents.KeysExhausted(exhausted)
    return ok


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task-ids", nargs="+", metavar="ID", help="GAIA validation task id(s) (space or comma separated)")
    p.add_argument("--n-questions", type=int, metavar="N", help="random subset of N validation questions; ignored if --task-ids is given")
    p.add_argument("--backend", choices=("cloud", "local"), default="cloud",
                   help="cloud (default): Ollama Cloud with rotating OLLAMA_API_KEY[_n] keys, gemma4:31b only; local: your own Ollama server")
    p.add_argument("--agent-models", nargs="+", choices=AGENT_MODELS, default=["gemma4:31b"], metavar="TAG",
                   help=f"ONE tag for all 3 agents, or THREE (agent 1 2 3). Choices: {', '.join(AGENT_MODELS)}")
    p.add_argument("--trust-model", choices=TRUST_MODELS, default="gemma4:31b", help="trust allocator model (independent of the agents)")
    p.add_argument("--injector-model", choices=AGENT_MODELS, default="gemma4:31b", help="error injector model")
    p.add_argument("--trust", choices=("off", "on", "both"), default="both", help="trust allocator setting(s) to run")
    p.add_argument("--n-rounds", type=int, default=2, help="debate rounds (default 2)")
    p.add_argument("--num-ctx", type=int, default=32768, help="Ollama context window in tokens (default 32768)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-think", action="store_true", help="disable thinking mode (it is skipped automatically for models without it)")
    p.add_argument("--llm-timeout", type=float, help="seconds an Ollama request may stay silent before a retry (default 60 cloud, 900 local)")
    p.add_argument("--seed", type=int, default=42, help="seed for --n-questions sampling and Ollama sampling")
    p.add_argument("--output-dir", help="default: results/agents-<models>__trust-<model>")
    p.add_argument("--work-dir", default=str(HERE / ".gaia_work"), help="scratch dir for the sqlite checkpoints")
    p.add_argument("--fresh", action="store_true", help="ignore earlier results and checkpoints and redo everything (default: resume - finished "
                   "conditions are skipped and a failed one repeats only its failed part)")
    p.add_argument("--max-output-tokens", type=int, default=12288, help="cap on tokens generated per model call (stops runaway thinking loops)")
    p.add_argument("--max-key-wait", type=float, default=7200.0, help="longest a call waits for a rate-limited Ollama key to come back before the run stops "
                   "cleanly (resume later with the same command)")
    p.add_argument("--quiet", action="store_true", help="do not print one line per LLM call")
    p.add_argument("--dotenv", default=str(HERE / ".env"))
    a = p.parse_args()
    if len(a.agent_models) not in (1, 3):
        p.error("--agent-models takes 1 or 3 tags")
    if not a.task_ids and not a.n_questions:
        p.error("give --task-ids and/or --n-questions")
    a.agent_models = a.agent_models * 3 if len(a.agent_models) == 1 else a.agent_models
    a.task_ids = [t for x in (a.task_ids or []) for t in x.split(",") if t]
    return a


def check_models(args: argparse.Namespace) -> None:
    """Fail fast if Ollama is unreachable, a model is not pulled, or an agent model cannot call tools."""
    import ollama

    if args.backend == "cloud":  # only gemma4:31b is served there; the keys are checked by the first call
        bad = (set(args.agent_models) | {args.trust_model, args.injector_model}) - set(agents.CLOUD_MODELS)
        if bad:
            raise SystemExit(f"Ollama Cloud backend serves only {', '.join(agents.CLOUD_MODELS)} (got {', '.join(sorted(bad))}); use --backend local")
        if not agents.numbered_keys("OLLAMA_API_KEY"):
            raise SystemExit("No OLLAMA_API_KEY / OLLAMA_API_KEY_1, _2, ... in .env")
        return

    try:
        have = {m.model for m in ollama.Client().list().models}
    except Exception as e:
        raise SystemExit(f"Cannot reach the local Ollama server ({type(e).__name__}: {e}). Start it with `ollama serve`.")
    need = set(args.agent_models) | {args.injector_model} | ({args.trust_model} if args.trust != "off" else set())
    missing = [m for m in sorted(need) if m not in have and f"{m}:latest" not in have]
    if missing:
        raise SystemExit("Models not pulled: " + ", ".join(missing) + "\n" + "\n".join(f"  ollama pull {m}" for m in missing))
    for m in set(args.agent_models):
        caps = agents.capabilities(m)
        if caps and "tools" not in caps:
            raise SystemExit(f"{m} does not report tool-calling support (capabilities: {caps}); agents need tools for GAIA")


def main() -> int:
    args = parse_args()
    load_dotenv(args.dotenv, override=False)
    if not agents.tavily_keys():
        raise SystemExit("TAVILY_API_KEY / TAVILY_API_KEY_1, _2, ... missing in .env (web search tool)")
    log(f"Tavily keys: {len(agents.tavily_keys())} (1000 searches each) | Ollama Cloud keys: {len(agents.numbered_keys('OLLAMA_API_KEY'))}")
    agents.CFG.update(cloud=args.backend == "cloud", num_ctx=args.num_ctx, think=not args.no_think, temperature=args.temperature, seed=args.seed, max_tokens=args.max_output_tokens, max_key_wait=args.max_key_wait, timeout=args.llm_timeout or (60.0 if args.backend == "cloud" else 900.0))
    check_models(args)
    ensure_gaia(next((os.getenv(v) for v in ("HF_TOKEN", "HUGGINGFACE_TOKEN") if os.getenv(v)), None))  # no-op after the 1st run
    tasks = load_tasks()
    safe = lambda s: re.sub(r"[^A-Za-z0-9.+_-]", "-", s)
    out_dir = Path(args.output_dir or HERE / "results" / f"agents-{safe('+'.join(dict.fromkeys(args.agent_models)))}__trust-{safe(args.trust_model)}")
    work_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.task_ids:
        missing = [t for t in args.task_ids if t not in tasks]
        if missing:
            raise SystemExit(f"Not in the GAIA validation set: {', '.join(missing)}")
        ids = args.task_ids
    else:
        ids = random.Random(args.seed).sample(sorted(tasks), min(args.n_questions, len(tasks)))
    log(f"{len(tasks)} validation tasks | running {len(ids)} | agents {args.agent_models} | trust {args.trust_model} | results: {out_dir}")

    rec = Recorder()
    rec.verbose = not args.quiet
    agents.TOOL_HOOK = lambda ev, name, info: rec.verbose and log(
        f"      tool {name}: {info}" if ev == "start" else f"        -> {name} done in {info}" if ev == "end" else f"      !! tool {name} ABANDONED after {info}")
    agents.NOTE_HOOK = lambda msg: log(f"      !! {msg}")
    install_recorder(rec, retries=0 if args.backend == "cloud" else 2)  # cloud: a failed call moves to the next key instead
    failed, stopped = 0, False
    for ti, tid in enumerate(ids, 1):
        log(f"=== Task {ti}/{len(ids)}: {tid} (level {tasks[tid]['Level']}) ===")
        try:
            failed += not run_task(tasks[tid], args, rec, out_dir, work_dir)
        except agents.KeysExhausted as e:
            log(f"STOPPING: {e}")
            stopped = True
            break
    rows = write_summary(out_dir)
    log(f"{'Stopped early' if stopped else 'Done'}. {len(rows)} records, {sum(bool(x['error']) for x in rows)} failed. Summary: {out_dir / 'summary.csv'}. "
        f"{'Rerun the same command to resume (only unfinished parts are repeated). ' if failed or stopped else ''}Analyse with: python analyze.py {out_dir.parent}")
    return 2 if stopped else 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
