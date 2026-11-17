"""Loading for analyze.py: result folders (one JSON record per task x condition) -> three tidy tables.

    runs   one row per record  (task x condition x setup): outcome, injection, cost, timing
    steps  one row per public message: step accuracy, trust score + its 4 maxims, the judge's prompt length
    tasks  one row per task x setup: the validity flags that decide which tasks enter which paired analysis

A "setup" is one experimental configuration (agent models + trust model). The two setups of the thesis are two cases of the
same experiment: case 1 = a strong judge supervising equally strong agents, case 2 = a judge ~3x the agents' size.
Every exclusion rule lives here, so the report can count and explain every dropped task."""

from __future__ import annotations

import difflib
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:  # the official scorer, so "original message correct" is judged exactly like every step
    from gaia_data import extract_answer, gaia_score
except ImportError:  # analysis run away from the project: fall back to the stored step_correct of a matching clean run
    extract_answer = gaia_score = None

DEFAULT_CTX = {"cloud": 262144, "local": 32768}  # judge context window when the record does not store num_ctx (--ctx-limit)
MAXIMS = ("quality", "quantity", "relation", "manner")


# ------------------------------------------------------------------------------------------ setup naming
def params_b(tag: str) -> float | None:
    """Nominal parameter count in billions from an Ollama tag ('qwen3.5:9b' -> 9, 'gemma4:31b-cloud' -> 31)."""
    m = re.search(r"(\d+(?:\.\d+)?)b", tag.split(":")[-1].lower()) or re.search(r"(\d+(?:\.\d+)?)b\b", tag.lower())
    return float(m.group(1)) if m else None


def setup_names(rec: dict) -> dict:
    agents = list(dict.fromkeys(rec["agent_models"].values()))
    judge = rec["trust_model"]
    pa, pj = params_b(agents[0]), params_b(judge)
    ratio = pj / pa if pa and pj else None
    fam = re.match(r"[A-Za-z]+", agents[0]).group(0).capitalize()
    return {"setup_id": f"agents={'+'.join(agents)} | trust={judge}", "family": fam, "agent_tag": "+".join(agents), "judge_tag": judge,
            "judge_ratio": ratio}


def label_setups(runs: pd.DataFrame) -> dict:
    """setup_id -> short plot label ('Case 1: gemma4:31b = judge' style), ordered by judge/agent size ratio (case 1 = weakest ratio)."""
    info = runs.drop_duplicates("setup_id").sort_values(["judge_ratio", "setup_id"], na_position="first")
    out = {}
    for i, r in enumerate(info.itertuples(), 1):
        same = r.judge_ratio is not None and abs(r.judge_ratio - 1) < 0.25
        out[r.setup_id] = (f"Case {i}: {r.agent_tag}" + (" (judge = agents)" if same else f" agents, {r.judge_tag} judge"
                           + (f" ({r.judge_ratio:.0f}x)" if r.judge_ratio and r.judge_ratio == r.judge_ratio else "")))
    return out


# ------------------------------------------------------------------------------------------ one record
def _valid(x) -> bool:
    return isinstance(x, (int, float)) and x == x and 1 <= x <= 5


def _words_changed(a: str, b: str) -> float:
    """Share of the words that differ between the original and the corrupted message (0 = identical, 1 = nothing in common)."""
    return 1 - difflib.SequenceMatcher(None, a.split(), b.split(), autojunk=False).ratio()


def _seconds(a: str | None, b: str | None) -> float:
    try:
        return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()
    except (TypeError, ValueError):
        return np.nan


def parse_record(r: dict, path: Path, root: Path) -> tuple[dict, list[dict]]:
    names = setup_names(r)
    kind, trust = r["condition"].split("_")
    inj = r.get("injection") or {}
    msgs, th = r.get("messages", []), r.get("trust_history", [])
    k = inj.get("log_index") if inj.get("eligible") else None
    ok = [bool(m["step_correct"]) for m in msgs]
    n = len(ok)
    calls = [c for c in r.get("token_stats", {}).get("calls", [])]
    good = [c for c in calls if c.get("status") == "ok"]
    by = lambda t, f: sum((c.get(f) or 0) for c in good if c["call_type"] == t)
    audit_calls = [c for c in good if c["call_type"] == "trust_audit"]
    trust_by_msg = {t["message_index"]: t for t in th if t.get("message_index") is not None}
    # i-th saved trust entry <-> i-th successful audit call (a resumed run repeats nothing); align from the end if counts differ
    tok = {}
    for t, c in zip(th[::-1], audit_calls[::-1]):
        tok[t["message_index"]] = c
    prop = None
    if k is not None and n:
        prop = 0
        while k + prop < n and not ok[k + prop]:
            prop += 1
    orig_text = inj.get("original")
    orig_correct = None
    if orig_text is not None and gaia_score is not None:
        orig_correct = bool(gaia_score(extract_answer(orig_text), r["ground_truth"]))
    first = msgs[k]["content"] if k is not None and k < n else (msgs[0]["content"] if msgs else None)
    row = {**names, "task_id": r["task_id"], "level": int(r["level"]), "kind": kind, "trust": trust, "condition": r["condition"],
           "backend": r.get("backend"), "num_ctx": r.get("num_ctx"), "error": r.get("error"), "ok": not r.get("error") and n > 0,
           "final_correct": bool(r["correct"]), "n_steps": n, "step_acc": float(np.mean(ok)) if ok else np.nan,
           "n_no_answer": sum("FINAL ANSWER:" not in m["content"] for m in msgs),
           "question_chars": len(r["question"]), "has_file": bool(r.get("attached_file")),
           "file_ext": Path(r["attached_file"]).suffix.lower() if r.get("attached_file") else "",
           "eligible": k is not None, "fm_id": inj.get("fm_id") if k is not None else None, "inj_step": k,
           "inj_step_correct": ok[k] if k is not None and k < n else None, "orig_correct": orig_correct,
           "trust_at_inj": trust_by_msg.get(k, {}).get("trust_score") if k is not None else None,
           "propagation": prop, "censored": bool(prop and k + prop == n),
           "orig_text": orig_text, "inj_text": inj.get("corrupted"), "first_text": first,
           "n_calls": len(calls), "n_failed_calls": len(calls) - len(good), "lean_turns": r.get("token_stats", {}).get("budget_lean_turns", 0),
           "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in good), "completion_tokens": sum(c.get("completion_tokens") or 0 for c in good),
           "llm_seconds": sum(c.get("elapsed_seconds") or 0 for c in calls), "wall_seconds": _seconds(r.get("started_at"), r.get("finished_at")),
           "tok_agent": by("agent_call", "prompt_tokens") + by("agent_call", "completion_tokens"),
           "tok_audit": by("trust_audit", "prompt_tokens") + by("trust_audit", "completion_tokens"),
           "sec_agent": sum(c.get("elapsed_seconds") or 0 for c in calls if c["call_type"] == "agent_call"),
           "sec_audit": sum(c.get("elapsed_seconds") or 0 for c in calls if c["call_type"] == "trust_audit"),
           "max_agent_prompt": max([c.get("prompt_tokens") or 0 for c in good if c["call_type"] == "agent_call"] or [0]),
           "max_audit_prompt": max([c.get("prompt_tokens") or 0 for c in audit_calls] or [0]),
           "n_audits": len(audit_calls), "n_trust_entries": len(th), "n_audit_unusable": sum(not _valid(t.get("trust_score")) for t in th),
           "audit_aligned": len(audit_calls) == len(th), "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path)}
    if k is not None and inj.get("original") and inj.get("corrupted"):
        a, b = extract_answer(inj["original"]) if extract_answer else "", extract_answer(inj["corrupted"]) if extract_answer else ""
        row.update(answer_changed=a != b, words_changed=_words_changed(inj["original"], inj["corrupted"]),
                   corrupt_len_ratio=len(inj["corrupted"]) / max(1, len(inj["original"])))
    steps = []
    for m in msgs:
        i, t = m["log_index"], trust_by_msg.get(m["log_index"], {})
        c = tok.get(i, {})
        mx = t.get("maxim_scores") or {}
        steps.append({"setup_id": names["setup_id"], "task_id": r["task_id"], "level": int(r["level"]), "kind": kind, "trust": trust, "step": i,
                      "agent": m["source"], "round": m["round"], "step_correct": bool(m["step_correct"]),
                      "has_answer": "FINAL ANSWER:" in m["content"], "trust_score": t.get("trust_score") if _valid(t.get("trust_score")) else np.nan,
                      "has_trust_entry": bool(t), **{f"maxim_{x}": mx.get(x, np.nan) for x in MAXIMS},
                      "saw_tool_outputs": t.get("saw_tool_outputs"), "injected": i == k, "rel_step": i - k if k is not None else np.nan,
                      "audit_prompt_tokens": c.get("prompt_tokens", np.nan), "audit_completion_tokens": c.get("completion_tokens", np.nan),
                      "audit_seconds": c.get("elapsed_seconds", np.nan), "content_chars": len(m["content"])})
    return row, steps


# ------------------------------------------------------------------------------------------ all records + validity flags
def load(roots: list[Path], ctx_limit: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs, steps = [], []
    for root in roots:
        for f in sorted(root.rglob("*.json")):
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(r, dict) or "condition" not in r or "messages" not in r:  # shared.json etc.
                continue
            a, b = parse_record(r, f, root)
            runs.append(a)
            steps += b
    if not runs:
        raise SystemExit(f"No run records found under {', '.join(map(str, roots))}")
    runs, steps = pd.DataFrame(runs), pd.DataFrame(steps)
    runs = runs.drop_duplicates(["setup_id", "task_id", "condition"], keep="last").reset_index(drop=True)  # same record found via two roots
    steps = steps.drop_duplicates(["setup_id", "task_id", "kind", "trust", "step"], keep="last").reset_index(drop=True)
    lim = {**DEFAULT_CTX, **(ctx_limit or {})}
    runs["ctx_limit"] = [float(n) if n == n and n else float(lim.get(b, lim["local"])) for n, b in zip(runs.num_ctx.astype(float), runs.backend)]
    steps = steps.merge(runs[["setup_id", "task_id", "kind", "trust", "ctx_limit"]], on=["setup_id", "task_id", "kind", "trust"], how="left")
    steps["audit_fill"] = steps.audit_prompt_tokens / steps.ctx_limit
    return runs, steps, build_tasks(runs, steps)


def build_tasks(runs: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    """One row per setup x task with the checks that gate every paired analysis (reason = why a task is left out, or '')."""
    rows = []
    ix = runs.set_index(["setup_id", "task_id", "condition"])
    trust_ix = steps.set_index(["setup_id", "task_id", "kind", "trust", "step"]).trust_score
    for (s, t), g in runs.groupby(["setup_id", "task_id"]):
        get = lambda c: ix.loc[(s, t, c)] if (s, t, c) in ix.index else None
        b_off, b_on, f_off, f_on = (get(c) for c in ("baseline_off", "baseline_on", "fork_off", "fork_on"))
        ok = lambda r: r is not None and bool(r.ok)
        row = {"setup_id": s, "task_id": t, "level": int(g.level.iloc[0]), "has_file": bool(g.has_file.iloc[0]),
               "question_chars": int(g.question_chars.iloc[0]), "n_records": len(g), "n_failed": int((~g.ok).sum()),
               "all_four": all(ok(r) for r in (b_off, b_on, f_off, f_on))}
        # --- the shared injection must be the very same message in both injected runs
        both = ok(f_off) and ok(f_on) and bool(f_off.eligible) and bool(f_on.eligible)
        row["fm_off"], row["fm_on"] = (f_off.fm_id if f_off is not None else None), (f_on.fm_id if f_on is not None else None)
        row["inj_identical"] = bool(both and f_off.inj_text == f_on.inj_text and f_off.orig_text == f_on.orig_text and f_off.fm_id == f_on.fm_id)
        row["eligible"] = bool(both)
        row["orig_correct"] = f_on.orig_correct if ok(f_on) else (f_off.orig_correct if ok(f_off) else None)
        # --- does the clean run's first message equal the "original" stored with the injection? (exact text match)
        for tr, b, f in (("off", b_off, f_off), ("on", b_on, f_on)):
            row[f"orig_match_{tr}"] = bool(ok(b) and f is not None and f.orig_text is not None and b.first_text == f.orig_text)
        if row["orig_correct"] is None or row["orig_correct"] != row["orig_correct"]:  # no scorer: take it from a matching clean run
            hit = [b for tr, b in (("off", b_off), ("on", b_on)) if row[f"orig_match_{tr}"]]
            row["orig_correct"] = bool(steps[(steps.setup_id == s) & (steps.task_id == t) & (steps.kind == "baseline") & (steps.trust == hit[0].trust)
                                             & (steps.step == 0)].step_correct.iloc[0]) if hit else None
        k = int(f_on.inj_step) if ok(f_on) and f_on.eligible else None
        tv = lambda kind, tr: trust_ix.get((s, t, kind, tr, k), np.nan) if k is not None else np.nan
        row["trust_orig"], row["trust_inj"] = tv("baseline", "on"), tv("fork", "on")
        row["trust_orig_valid"], row["trust_inj_valid"] = _valid(row["trust_orig"]), _valid(row["trust_inj"])
        row["trust_orig_recorded"] = bool(ok(b_on) and (steps[(steps.setup_id == s) & (steps.task_id == t) & (steps.kind == "baseline")
                                                              & (steps.trust == "on") & (steps.step == k)].has_trust_entry.any())) if k is not None else False
        # --- reasons
        row["rq_reason"] = ("" if row["inj_identical"] else "a fork run failed" if not (ok(f_off) and ok(f_on)) else
                            "injector ineligible" if not both else "injections differ between allocator off/on")
        row["vq2_reason"] = ("" if (ok(f_on) and f_on.eligible and ok(b_on) and row["orig_match_on"] and row["trust_orig_valid"] and row["trust_inj_valid"])
                             else "fork_on failed" if not ok(f_on) else "injector ineligible" if not f_on.eligible else "baseline_on failed" if not ok(b_on)
                             else "first message differs from stored original" if not row["orig_match_on"]
                             else "no saved trust-history record" if not (row["trust_orig_recorded"]) else "unusable trust score")
        rows.append(row)
    return pd.DataFrame(rows)
