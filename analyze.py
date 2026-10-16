#!/usr/bin/env python3
"""Analysis of all collected runs, following the thesis slides (7-9).

    python analyze.py results/            # scans every <task_id>/{baseline,fork}_{off,on}.json below it

Variables (slide 7), per run / per step:
  Step accuracy      is the step's FINAL ANSWER correct (GAIA scorer, same as the final answer)        -> messages[].step_correct
  Final accuracy     is the final answer correct                                                        -> correct
  Error propagation  number of inaccurate steps IN A ROW, counted from the injected step onwards
                     (0 if the injected step is itself still accurate; "censored" = never recovered before the last step)
  Trust score        mean of the 4 Gricean maxim scores (1-5) of the step                               -> trust_history[].trust_score
Research question (slide 8): does the trust allocator increase resiliency?  fork_off vs fork_on, paired by task.
Validation questions (slide 9): VQ1 trust score vs step inaccuracy (internal errors), VQ2 trust of the injected
message vs the original message (external / AiTM), plus the exploratory "allocator mistakes" analysis.
Everything is reported per experiment configuration (agent models + trust model). Writes <root>/analysis/{report.md,runs.csv,steps.csv}.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------------------------------ load -> tables
def load(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (runs, steps): one row per run (task x condition x config) and one per public message."""
    runs, steps = [], []
    for f in sorted(root.rglob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if "condition" not in r or "messages" not in r:  # not a run record
            continue
        am = list(dict.fromkeys(r["agent_models"].values()))
        cfg = f"agents={'+'.join(am) if len(am) == 1 else '/'.join(r['agent_models'].values())} | trust={r['trust_model']}"
        kind, trust = r["condition"].split("_")
        inj = r.get("injection") or {}
        k = inj.get("log_index") if inj.get("eligible") else None
        ok = [m["step_correct"] for m in r["messages"]]
        trust_at = {t["message_index"]: t["trust_score"] for t in r["trust_history"]}
        prop = None
        if k is not None:  # inaccurate steps in a row, from the injected step on
            prop = 0
            while k + prop < len(ok) and not ok[k + prop]:
                prop += 1
        runs.append({"config": cfg, "task_id": r["task_id"], "kind": kind, "trust": trust, "final_correct": bool(r["correct"]),
                     "step_acc": float(np.mean(ok)) if ok else np.nan, "n_steps": len(ok), "eligible": k is not None,
                     "fm_id": inj.get("fm_id"), "inj_step": k, "inj_step_correct": ok[k] if k is not None else None,
                     "trust_at_inj": trust_at.get(k) if k is not None else None,
                     "propagation": prop, "censored": bool(prop and k + prop == len(ok)), "error": r.get("error")})
        steps += [{"config": cfg, "task_id": r["task_id"], "kind": kind, "trust": trust, "step": m["log_index"], "agent": m["source"],
                   "round": m["round"], "step_correct": bool(m["step_correct"]), "trust_score": trust_at.get(m["log_index"]),
                   "injected": m["log_index"] == k} for m in r["messages"]]
    runs, steps = pd.DataFrame(runs), pd.DataFrame(steps)
    if not runs.empty:  # was the ORIGINAL message (clean run, same task and allocator setting) at the injected step correct?
        orig = steps[steps.kind == "baseline"].set_index(["config", "task_id", "trust", "step"]).step_correct
        runs["orig_step_correct"] = [orig.get((c, t, tr, int(k))) if k == k and k is not None else None
                                     for c, t, tr, k in zip(runs.config, runs.task_id, runs.trust, runs.inj_step)]
    return runs, steps


# ------------------------------------------------------------------------------------------ small stats helpers
def wilson(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    p, z = k / n, 1.96
    c, h = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{p:.2f} [{max(0, c - h):.2f}, {min(1, c + h):.2f}]"


def pv(p) -> str:
    return "n/a" if p is None or p != p else ("<0.001" if p < 0.001 else f"{p:.3f}")


def mcnemar(a: pd.Series, b: pd.Series) -> str:
    """Exact McNemar test on paired booleans (a = reference, b = comparison)."""
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = stats.binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return f"only-ref-correct={only_a}, only-comparison-correct={only_b}, exact McNemar p={pv(p)}"


def wilcoxon(x, y, alt="two-sided") -> float | None:
    try:
        return stats.wilcoxon(x, y, alternative=alt).pvalue
    except ValueError:  # all differences zero / no pairs
        return None


def auc(correct: np.ndarray, wrong: np.ndarray) -> float:
    """P(trust of a correct step > trust of an incorrect step), ties 1/2: how well trust ranks step accuracy."""
    return stats.mannwhitneyu(correct, wrong).statistic / (len(correct) * len(wrong))


def table(rows: list[dict]) -> list[str]:
    if not rows:
        return ["(no data)", ""]
    cols = list(rows[0])
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + \
           ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows] + [""]


# ------------------------------------------------------------------------------------------ one configuration
def analyse(cfg: str, runs: pd.DataFrame, steps: pd.DataFrame, high: float, low: float) -> list[str]:
    o = [f"# {cfg}", ""]
    ok_runs = runs[runs.error.isna()]
    forks = ok_runs[(ok_runs.kind == "fork") & ok_runs.eligible]
    o += [f"Tasks: {runs.task_id.nunique()} | runs with errors (excluded): {int(runs.error.notna().sum())} | "
          f"eligible injections: off={int((forks.trust == 'off').sum())}, on={int((forks.trust == 'on').sum())}",
          "Failure modes chosen by the injector: " + (", ".join(f"{k}: {v}" for k, v in forks.fm_id.value_counts().items()) or "none"), ""]

    # --- the four experimental conditions (slide 6); ineligible forks are not real injections, so they are left out
    o += ["## Conditions (slide 6) - variables of slide 7", ""]
    rows = []
    for name, kind, tr, d in (("1 baseline (no error, no allocator)", "baseline", "off", ok_runs),
                              ("2 allocator (no error)", "baseline", "on", ok_runs),
                              ("3 injected error, no allocator", "fork", "off", forks),
                              ("4 injected error + allocator", "fork", "on", forks)):
        d = d[(d.kind == kind) & (d.trust == tr)]
        st = steps[(steps.kind == kind) & (steps.trust == tr) & steps.task_id.isin(d.task_id)]
        rows.append({"condition": name, "n": len(d), "final accuracy [95% CI]": wilson(int(d.final_correct.sum()), len(d)),
                     "mean step accuracy": f"{d.step_acc.mean():.2f}" if len(d) else "n/a",
                     "mean trust score": f"{st.trust_score.mean():.2f}" if st.trust_score.notna().any() else "-",
                     "mean error propagation (steps)": f"{d.propagation.mean():.2f}" if d.propagation.notna().any() else "-",
                     "never recovered": int(d.censored.sum()) if d.propagation.notna().any() else "-"})
    o += table(rows)

    # --- RQ: does the trust allocator increase resiliency?  paired by task, fork_off vs fork_on
    p = forks.pivot(index="task_id", columns="trust", values=["final_correct", "propagation", "inj_step_correct", "orig_step_correct", "trust_at_inj"])
    p = p.dropna(subset=[("final_correct", "off"), ("final_correct", "on")]) if {"off", "on"} <= set(forks.trust) else p.iloc[0:0]
    o += ["## RQ (slide 8): does the trust allocator increase resiliency?", "",
          "Paired by task: the same task with an injected error, allocator OFF vs ON.", ""]
    for label, sub in (("all eligible injections", p),
                       ("only where the original message was correct (an error was truly introduced)",
                        p[p[("orig_step_correct", "off")].astype(bool) & p[("orig_step_correct", "on")].astype(bool)])):
        o += [f"**{label}** (n = {len(sub)} pairs)", ""]
        if len(sub) == 0:
            o += ["(no paired data)", ""]
            continue
        off, on = sub[("final_correct", "off")].astype(bool), sub[("final_correct", "on")].astype(bool)
        pr_off, pr_on = sub[("propagation", "off")].astype(float), sub[("propagation", "on")].astype(float)
        o += [f"- Overall - final accuracy: off {off.mean():.2f} vs on {on.mean():.2f}; {mcnemar(off, on)}",
              f"- Step level - error propagation (inaccurate steps in a row): off {pr_off.mean():.2f} vs on {pr_on.mean():.2f} "
              f"(lower = faster recovery); Wilcoxon signed-rank (on < off) p={pv(wilcoxon(pr_on, pr_off, 'less'))}", ""]
    base = ok_runs[ok_runs.kind == "baseline"].pivot(index="task_id", columns="trust", values="final_correct").dropna()
    inj = ok_runs[ok_runs.task_id.isin(p.index)].pivot_table(index="task_id", columns=["kind", "trust"], values="final_correct", aggfunc="first").dropna()
    o += ["Context:"]
    if len(base):
        o += [f"- Cost of the allocator without errors (baseline off vs on, n={len(base)}): {mcnemar(base['off'].astype(bool), base['on'].astype(bool))}"]
    if len(inj):
        o += [f"- Effect of the injected error, allocator off (baseline vs fork, n={len(inj)}): {mcnemar(inj[('baseline', 'off')].astype(bool), inj[('fork', 'off')].astype(bool))}",
              f"- Effect of the injected error, allocator on (baseline vs fork, n={len(inj)}): {mcnemar(inj[('baseline', 'on')].astype(bool), inj[('fork', 'on')].astype(bool))}"]
    o += [""]

    # --- VQ1: is the trust score a good predictor of step inaccuracy?  clean runs with allocator (no injected message)
    s = steps[(steps.kind == "baseline") & (steps.trust == "on") & steps.trust_score.notna()]
    c, w = s[s.step_correct].trust_score.to_numpy(float), s[~s.step_correct].trust_score.to_numpy(float)
    o += ["## VQ1 (internal / reasoning errors): is the trust score a good predictor of step inaccuracy?", "",
          f"Clean runs with the allocator: {len(s)} steps, {len(w)} inaccurate.", ""]
    if len(c) and len(w):
        a = auc(c, w)
        rng, tasks = np.random.default_rng(0), s.task_id.unique()
        boot = []
        for _ in range(1000):  # steps of one task are not independent -> bootstrap over tasks
            b = pd.concat([s[s.task_id == t] for t in rng.choice(tasks, len(tasks))])
            bc, bw = b[b.step_correct].trust_score.to_numpy(float), b[~b.step_correct].trust_score.to_numpy(float)
            if len(bc) and len(bw):
                boot.append(auc(bc, bw))
        rho = stats.spearmanr(s.trust_score, s.step_correct.astype(int))
        o += [f"- Mean trust: correct steps {c.mean():.2f} vs inaccurate steps {w.mean():.2f}; Mann-Whitney (correct > inaccurate) p={pv(stats.mannwhitneyu(c, w, alternative='greater').pvalue)}",
              f"- AUC of trust for detecting inaccurate steps: {a:.2f}" + (f" (task-bootstrap 95% CI {np.percentile(boot, 2.5):.2f}-{np.percentile(boot, 97.5):.2f})" if boot else "") + " (0.5 = chance)",
              f"- Spearman rho(trust, step correct) = {rho.statistic:.2f}, p={pv(rho.pvalue)}", ""]
    else:
        o += ["(needs both correct and inaccurate steps)", ""]

    # --- VQ2: is the trust score lower for the injected (AiTM) message than for the original one?
    o += ["## VQ2 (external / AiTM): is the trust score significantly lower for the injected message than for the original?", ""]
    if "on" in set(forks.trust):
        f_on = forks[forks.trust == "on"].set_index("task_id")
        orig = steps[(steps.kind == "baseline") & (steps.trust == "on")].set_index(["task_id", "step"]).trust_score
        pairs = pd.DataFrame({"fm": f_on.fm_id, "injected": f_on.trust_at_inj,
                              "original": [orig.get((t, int(k))) for t, k in zip(f_on.index, f_on.inj_step)]}).dropna(subset=["injected", "original"])
        pairs = pairs.astype({"injected": float, "original": float})
        if len(pairs):
            d = pairs.injected - pairs.original
            o += [f"n = {len(pairs)} pairs. Mean trust: original {pairs.original.mean():.2f} vs injected {pairs.injected.mean():.2f} "
                  f"(mean difference {d.mean():.2f}; lower for the injected message in {int((d < 0).sum())}/{len(d)}).",
                  f"Wilcoxon signed-rank (injected < original) p={pv(wilcoxon(pairs.injected, pairs.original, 'less'))}"
                  + (f"; Cohen's dz = {d.mean() / d.std(ddof=1):.2f}" if len(d) > 1 and d.std(ddof=1) > 0 else ""), ""]
            o += table([{"failure mode": k, "n": len(g), "mean trust original": f"{g.original.mean():.2f}",
                         "mean trust injected": f"{g.injected.mean():.2f}"} for k, g in pairs.groupby("fm")])
        else:
            o += ["(no scored injected messages)", ""]
    else:
        o += ["(no allocator-on forks)", ""]

    # --- exploratory: does an allocator MISTAKE make recovery worse than the no-allocator baseline?
    o += ["## Exploratory: allocator mistakes vs. recovery", "",
          f"A mistake = high trust (>= {high}) on an inaccurate step, or low trust (< {low}) on an accurate step "
          "(thresholds inherited from the former high / low labels; only used here).", ""]
    s_on = steps[(steps.kind == "baseline") & (steps.trust == "on") & steps.trust_score.notna()]
    cls = np.where(s_on.trust_score >= high, "high trust", np.where(s_on.trust_score < low, "low trust", "medium trust"))
    ct = pd.crosstab(pd.Series(cls, name="trust"), s_on.step_correct.map({True: "accurate step", False: "inaccurate step"}).rename("step"))
    o += ["Clean allocator-on steps:", ""] + table([{"": i, **{c: int(v) for c, v in r.items()}} for i, r in ct.iterrows()])
    if len(p) and p[("trust_at_inj", "on")].notna().any():
        q = p[p[("trust_at_inj", "on")].notna()].copy()
        inj_ok = q[("inj_step_correct", "on")].astype(bool)
        t = q[("trust_at_inj", "on")].astype(float)
        q["mistake"] = (~inj_ok & (t >= high)) | (inj_ok & (t < low))
        q["d_final"] = q[("final_correct", "on")].astype(int) - q[("final_correct", "off")].astype(int)
        q["d_prop"] = q[("propagation", "on")].astype(float) - q[("propagation", "off")].astype(float)
        q["worse"] = q.d_final < 0
        o += ["At the injected message (allocator on vs the paired allocator-off run; d = on - off, so d_final < 0 / d_prop > 0 = worse):", ""]
        o += table([{"allocator": "mistaken" if m else "right", "n": len(g), "mean d_final": f"{g.d_final.mean():.2f}",
                     "mean d_propagation": f"{g.d_prop.mean():.2f}", "worse on final": int(g.worse.sum())} for m, g in q.groupby("mistake")])
        if q.mistake.nunique() == 2:
            o += [f"Fisher exact (worse-on-final vs allocator mistaken) p={pv(stats.fisher_exact(pd.crosstab(q.mistake, q.worse))[1])}", ""]
    else:
        o += ["(no paired allocator-on forks)", ""]
    return o


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="results folder (scanned recursively)")
    ap.add_argument("--high", type=float, default=4.5, help="trust >= high counts as 'high trust' (exploratory part)")
    ap.add_argument("--low", type=float, default=3.0, help="trust < low counts as 'low trust' (exploratory part)")
    a = ap.parse_args()
    runs, steps = load(a.root)
    if runs.empty:
        raise SystemExit(f"No run records found under {a.root}")
    out = a.root / "analysis"
    out.mkdir(exist_ok=True)
    runs.to_csv(out / "runs.csv", index=False)
    steps.to_csv(out / "steps.csv", index=False)
    report = []
    for cfg in sorted(runs.config.unique()):
        report += analyse(cfg, runs[runs.config == cfg], steps[steps.config == cfg], a.high, a.low) + ["---", ""]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"Wrote {out}/report.md, runs.csv, steps.csv")


if __name__ == "__main__":
    main()
