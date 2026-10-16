#!/usr/bin/env python3
"""Run GAIA validation tasks through LLM-debate (local Ollama models) with / without the trust allocator,
and inject an FM-2.x error into one public message.

    python run_gaia_benchmark.py --n-questions 5 --agent-models gemma4:31b --trust-model qwen3.8:27b
    python run_gaia_benchmark.py --task-ids <id> --agent-models gemma4:31b gemma4:12b qwen3.5:9b --trust-model muse-glimmer:30b

Per task this gives the 4 conditions of the experiment (slide 6):
    <output-dir>/<task_id>/baseline_off.{json,md,full.md}   1  clean run, no trust allocator
    <output-dir>/<task_id>/baseline_on.{json,md,full.md}    2  clean run, with trust allocator
    <output-dir>/<task_id>/fork_off.{json,md,full.md}       3  same clean run forked at one message: injected error, no allocator
    <output-dir>/<task_id>/fork_on.{json,md,full.md}        4  same as 3, with allocator
.md = concise public trace; .full.md = + every model call (inputs, reasoning, outputs). Then run: python analyze.py <results>

Setup: .env next to this script with TAVILY_API_KEY (web search) and, for the ONE-TIME GAIA download only, HF_TOKEN.
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
# Trace data collection: every LLM call (tokens, timing, input, reasoning, output), split into the
# clean phase ("baseline") and the phase after the fork ("fork", includes the injector call).
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
        self.phase = "baseline"
        self.started_baseline, self.started_fork = now_iso(), None

    def begin_fork(self) -> None:
        self.phase, self.started_fork = "fork", now_iso()

    def add(self, **kw: Any) -> None:
        with self.lock:
            self.calls.append({"call_index": len(self.calls) + 1, "phase": self.phase, **kw})
            c = self.calls[-1]
        if self.verbose:
            log(f"      llm call #{c['call_index']} {c['call_type']:<11} {c['model']:<17} {c['elapsed_seconds']:6.1f}s  "
                f"tokens in/out {c['prompt_tokens']}/{c['completion_tokens']}" + ("" if c["status"] == "ok" else f"  FAILED: {c['error']}"))

    def stats(self, phase: str) -> dict:
        calls = [c for c in self.calls if c["phase"] == phase]
        ok = [c for c in calls if c["status"] == "ok"]
        pt, ct = sum(c["prompt_tokens"] or 0 for c in ok), sum(c["completion_tokens"] or 0 for c in ok)
        return {"n_llm_calls": len(calls), "n_successful_calls": len(ok), "n_failed_attempts": len(calls) - len(ok),
                "prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct,
                "total_elapsed_seconds": sum(c["elapsed_seconds"] for c in calls),
                "calls": [{k: v for k, v in c.items() if k != "phase"} for c in calls]}


def install_recorder(rec: Recorder, retries: int = 2) -> None:
    """Wrap ChatOllama._generate once: every call (agents, trust allocator, injector) is recorded, and network-level
    failures (dropped / timed-out connection) are retried. The call type comes from the `role` metadata (agents.make_llm)."""
    import httpx
    from langchain_ollama import ChatOllama

    orig = ChatOllama._generate

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
        rec.add(call_type=kind, provider="ollama", model=model, elapsed_seconds=time.time() - t0,
                prompt_tokens=um.get("input_tokens"), completion_tokens=um.get("output_tokens"), status="ok", error=None,
                input=inp, output=out, reasoning=out.get("reasoning"))
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
                 finished: str, args: argparse.Namespace, injection: Optional[dict] = None, error: Optional[str] = None) -> dict:
    final = state.get("final")
    extracted = extract_answer(final)
    return {
        "task_id": task["task_id"], "question": task["Question"], "ground_truth": task["Final answer"], "level": task["Level"],
        "attached_file": task["file"], "thread_id": thread_id, "condition": condition, "trust_allocator": trust,
        "agent_models": dict(zip("123", args.agent_models)), "trust_model": args.trust_model,
        "injector_model": args.injector_model, "temperature": args.temperature, "seed": args.seed, "n_rounds": args.n_rounds,
        "final_answer_raw": final, "final_answer_extracted": extracted,
        "correct": gaia_score(extracted, task["Final answer"]) if final is not None else False,
        "messages": public_messages(state, task["Final answer"]), "trust_history": trust_history(state),
        "injection": injection, "error": error, "token_stats": stats, "started_at": started, "finished_at": finished,
        "contexts": {f"Agent{a}": [msg_dict(m) for m in ms] for a, ms in (state.get("hist") or {}).items()},
    }


def injection_record(res: dict) -> dict:
    keys = ("eligible", "requested_n", "n", "attempts", "fm_id", "fm_name", "reason", "original", "corrupted", "prompt", "injector_response")
    inj = {k: res.get(k) for k in keys}
    inj["log_index"] = res["n"] - 1 if res["eligible"] else None  # one public message per agent turn
    inj["target"] = f"Agent{res['agent']} (round {res['round']})" if res["eligible"] else None
    return inj


# --------------------------------------------------------------------------------------------------
# Markdown + JSON output
# --------------------------------------------------------------------------------------------------
def _fence(text: Any) -> str:
    return f"````\n{text if text is not None else '(none)'}\n````"


def render_md(title: str, r: dict, full: bool = False, base: Optional[dict] = None) -> str:
    """Public report; full=True adds every model call (for a fork: the clean calls first, then the resumed ones)."""
    inj = r.get("injection") or {}
    o = [f"# {title}" + (" - full trace" if full else ""), "",
         f"**Task ID:** `{r['task_id']}`", f"**Condition:** `{r['condition']}`",
         f"**Trust allocator:** {'ON' if r['trust_allocator'] else 'OFF'}",
         f"**Agent models:** {r['agent_models']} | **Trust model:** {r['trust_model'] if r['trust_allocator'] else '-'}",
         f"**Ground truth:** {r['ground_truth']}", f"**Final answer:** {r['final_answer_extracted']}",
         f"**Correct:** {r['correct']}", "", "## Initial question", "", r["question"]]
    if r.get("attached_file"):
        o += ["", f"**Attachment:** `{r['attached_file']}`"]
    if inj:
        o += ["", "### Fork / divergence", "", f"- Eligibility: **{'ELIGIBLE' if inj['eligible'] else 'INELIGIBLE'}**",
              f"- Requested public message: `{inj['requested_n']}`, injected message: `{inj['n']}`"]
        if inj["eligible"]:
            o += [f"- Target: {inj['target']}", f"- Failure mode chosen by the injector: `{inj['fm_id']}` - {inj['fm_name']}",
                  f"- Injector's reason: {inj['reason'] or '(none)'}"]
            if inj["n"] != inj["requested_n"]:
                o += ["- The requested message was ineligible, so injection advanced to the next one."]
        else:
            o += ["- No corrupted message was inserted; the clean run is the only execution trace."]
    trust = {t["message_index"]: t for t in r["trust_history"]}
    o += ["", "---", "", "## Public flow", ""]
    for m in r["messages"]:
        mark = " - **CORRUPTED / fork point**" if inj.get("log_index") == m["log_index"] else ""
        if mark:
            o += ["", "## Fork begins here", "", "Everything above is the clean pre-fork execution; this is the first changed output.", ""]
        o += [f"### [{m['log_index'] + 1}] {m['source']} - round {m['round']}{mark}", "", _fence(m["content"]), "",
              f"**Step accuracy:** {'correct' if m['step_correct'] else 'incorrect'} (answer: `{m['answer']}`)", ""]
        if m["log_index"] in trust:
            t = trust[m["log_index"]]
            o += [f"**Trust allocator output -> score {t['trust_score']}/5**", "", t["reason"], ""]
    if full:
        o += ["---", "", "## Model-call trace", ""]
        phases = [("PRE-FORK / CLEAN", base["token_stats"]["calls"]), ("FORK / RESUMED", r["token_stats"]["calls"])] if base \
            else [("RUN", r["token_stats"]["calls"])]
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


def write_outputs(task_dir: Path, name: str, rec: dict, base: Optional[dict] = None) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / f"{name}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (task_dir / f"{name}.md").write_text(render_md(name, rec), encoding="utf-8")
    (task_dir / f"{name}.full.md").write_text(render_md(name, rec, full=True, base=base), encoding="utf-8")


def summary_row(r: dict) -> dict:
    return {"task_id": r["task_id"], "level": r["level"], "condition": r["condition"],
            "agent_models": "+".join(r["agent_models"].values()), "trust_model": r["trust_model"],
            "fm_id": (r["injection"] or {}).get("fm_id"), "eligible": (r["injection"] or {}).get("eligible", ""),
            "ground_truth": r["ground_truth"], "answer": r["final_answer_extracted"], "correct": r["correct"], "error": r["error"] or ""}


# --------------------------------------------------------------------------------------------------
# One (task, trust setting): clean baseline run + injected fork
# --------------------------------------------------------------------------------------------------
def run_condition(task: dict, trust: bool, args: argparse.Namespace, rec: Recorder, out_dir: Path, work_dir: Path) -> list[dict]:
    from debate import build_graph, run_debate
    from error_injection import inject_error

    on = "on" if trust else "off"
    task_dir = out_dir / task["task_id"]
    if args.skip_existing and all((task_dir / f"{c}_{on}.json").exists() for c in ("baseline", "fork")):
        log(f"  skip trust={on}: outputs already exist")
        return []
    rec.reset()
    thread_id = f"{task['task_id']}::{'trust' if trust else 'plain'}::{uuid.uuid4().hex[:8]}"
    graph = build_graph(dict(zip("123", args.agent_models)), args.trust_model if trust else None, str(work_dir / "gaia.db"))
    base_state = res = err = None
    try:
        cfg, base_state = run_debate(graph, task["Question"], task["file"], args.n_rounds, thread_id)
        rec.begin_fork()
        res = inject_error(graph, cfg, args.injector_model, args.message_index)
    except Exception as e:  # keep whatever finished (e.g. the clean run) and report the failure
        err = f"{type(e).__name__}: {e}"
        log(f"  ERROR trust={on}: {err}")
        traceback.print_exc()
    common = dict(task=task, trust=trust, thread_id=thread_id, args=args)
    if base_state is None:  # the clean run itself failed: nothing to save but the error
        r = build_record(state={}, condition=f"baseline_{on}", stats=rec.stats("baseline"), started=rec.started_baseline,
                         finished=now_iso(), error=err, **common)
        write_outputs(task_dir, f"baseline_{on}", r)
        return [summary_row(r)]
    fork_start = rec.started_fork or now_iso()
    base = build_record(state=base_state, condition=f"baseline_{on}", stats=rec.stats("baseline"),
                        started=rec.started_baseline, finished=fork_start, **common)
    write_outputs(task_dir, f"baseline_{on}", base)
    log(f"  trust={on:3s} clean: answer={base['final_answer_extracted']!r} correct={base['correct']}")
    # ineligible (or failed) injection -> the fork record falls back to the clean state, flagged via injection.eligible
    fork = build_record(state=(res or {}).get("final_state") or base_state, condition=f"fork_{on}", stats=rec.stats("fork"),
                        started=fork_start, finished=now_iso(), injection=injection_record(res) if res else None, error=err, **common)
    write_outputs(task_dir, f"fork_{on}", fork, base=base)
    log(f"  trust={on:3s} fork : answer={fork['final_answer_extracted']!r} correct={fork['correct']} "
        f"fm={(fork['injection'] or {}).get('fm_id')}" + (f" ERROR={err}" if err else ""))
    return [summary_row(base), summary_row(fork)]


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task-ids", nargs="+", metavar="ID", help="GAIA validation task id(s) (space or comma separated)")
    p.add_argument("--n-questions", type=int, metavar="N", help="random subset of N validation questions; ignored if --task-ids is given")
    p.add_argument("--agent-models", nargs="+", choices=AGENT_MODELS, default=["gemma4:31b"], metavar="TAG",
                   help=f"ONE tag for all 3 agents, or THREE (agent 1 2 3). Choices: {', '.join(AGENT_MODELS)}")
    p.add_argument("--trust-model", choices=TRUST_MODELS, default="gemma4:31b", help="trust allocator model (independent of the agents)")
    p.add_argument("--injector-model", choices=AGENT_MODELS, default="gemma4:31b", help="error injector model")
    p.add_argument("--trust", choices=("off", "on", "both"), default="both", help="trust allocator setting(s) to run")
    p.add_argument("--n-rounds", type=int, default=2, help="debate rounds (default 2)")
    p.add_argument("--message-index", type=int, default=1, metavar="N",
                   help="public message to corrupt, 1-based (default 1 = Agent 1's first answer, the first one the allocator rates)")
    p.add_argument("--num-ctx", type=int, default=32768, help="Ollama context window in tokens (default 32768)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-think", action="store_true", help="disable thinking mode (it is skipped automatically for models without it)")
    p.add_argument("--llm-timeout", type=float, default=900.0, help="seconds an Ollama request may stay silent before a retry (default 900)")
    p.add_argument("--seed", type=int, default=42, help="seed for --n-questions sampling and Ollama sampling")
    p.add_argument("--output-dir", help="default: results/agents-<models>__trust-<model>")
    p.add_argument("--work-dir", default=str(HERE / ".gaia_work"), help="scratch dir for the sqlite checkpoints")
    p.add_argument("--skip-existing", action="store_true", help="skip task/trust combos whose outputs already exist")
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
    if not os.getenv("TAVILY_API_KEY", "").strip():
        raise SystemExit("TAVILY_API_KEY missing in .env (web search tool)")
    agents.CFG.update(num_ctx=args.num_ctx, think=not args.no_think, temperature=args.temperature, seed=args.seed, timeout=args.llm_timeout)
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
    install_recorder(rec)
    rows: list[dict] = []
    for ti, tid in enumerate(ids, 1):
        log(f"=== Task {ti}/{len(ids)}: {tid} (level {tasks[tid]['Level']}) ===")
        for trust in {"off": [False], "on": [True], "both": [False, True]}[args.trust]:
            rows += run_condition(tasks[tid], trust, args, rec, out_dir, work_dir)
    if rows:
        csv_path, new = out_dir / "summary.csv", not (out_dir / "summary.csv").exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)
        log(f"Done. Summary: {csv_path}. Analyse with: python analyze.py {out_dir.parent}")
    return 1 if any(r["error"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
