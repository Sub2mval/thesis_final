#!/usr/bin/env python3
"""Analysis of the GAIA trust-allocator experiment: EDA, the research question, the validation questions, exploratory analyses,
both experimental setups side by side. Writes a LaTeX report (+ PDF if pdflatex exists), figures and CSV tables.

    python analyze.py results/                                   # every setup found below the folder
    python analyze.py results/agents-A results/agents-B --out analysis_both

Output folder (default <root>/analysis): report.tex / report.pdf / report.md, figures/*.png, tables/*.csv, runs.csv, steps.csv, tasks.csv.

Variables (thesis slide 7), per run / per step:
  step accuracy      is the step's FINAL ANSWER correct (GAIA scorer, same as the final answer)
  final accuracy     is the final answer correct
  error propagation  inaccurate steps IN A ROW from the injected step on (0 = the injected step was still accurate; censored = never recovered)
  trust score        mean of the 4 Gricean maxim scores (1-5) of a step
Questions: RQ (slide 8) allocator off vs on, paired by task; VQ1/VQ2 (slide 9); exploratory: allocator mistakes, judge prompt length.
Code layout: analysis_data (load + validity rules) / analysis_core (statistics, report writer) / analysis_eda / analysis_rq / analysis_validation."""

import argparse
from pathlib import Path

import pandas as pd

import analysis_core as core
from analysis_core import Ctx, Report, holm, pe, pv
from analysis_data import label_setups, load
from analysis_eda import eda
from analysis_rq import head_to_head, rq
from analysis_validation import ctxlen, mistakes, vq1, vq2


def sig(p: float) -> str:
    return "no significant difference" if p != p or p >= .05 else "a significant difference"


def summary(c: Ctx, rep: Report) -> None:
    """Executive summary assembled from the numbers the other chapters stored in rep.findings."""
    s = Report.section(rep, "Summary of results", 1, "What are the answers, in one page?")
    s.p("All numbers below are produced by the analyses in the later chapters; p-values are unadjusted unless stated. 'Error truly introduced' = tasks where the original Agent 1 message was correct.")
    rows, tests = [], []
    F = rep.findings
    for sid in c.ids:
        for key, nm, fmt in (("rq_acc_introduced", "RQ overall: accuracy, allocator on - off (pp)", lambda r: f"{100 * r['effect']:+.1f} [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}]"),
                             ("rq_prop_introduced", "RQ step level: propagation, on - off (steps)", lambda r: f"{r['effect']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]"),
                             ("vq1", "VQ1: AUC of trust for step accuracy", lambda r: f"{r['effect']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"),
                             ("vq2", "VQ2: mean trust drop, injected - original", lambda r: f"{r['effect']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]")):
            r = F.get((sid, key))
            if r:
                rows.append({"case": c.short[sid], "result": nm, "estimate [95% CI]": fmt(r), "n": r["n"], "p": pv(r["p"])})
                tests.append((c.short[sid], nm, r["p"]))
    s.table(pd.DataFrame(rows), "summary_headline", "Headline results per case")
    adj = holm([t[2] for t in tests])
    s.table(pd.DataFrame([{"case": a, "test": b, "p": pv(p), "Holm-adjusted p": pv(q)} for (a, b, p), q in zip(tests, adj)]), "summary_holm", "Headline tests with Holm correction for multiple testing",
            "Holm controls the chance of at least one false positive over these headline tests; one-sided and two-sided tests are mixed as specified per question.")
    for sid in c.ids:
        a, b, v1, v2 = (F.get((sid, k)) for k in ("rq_acc_introduced", "rq_prop_introduced", "vq1", "vq2"))
        if not (a and b):
            continue
        t = [f"{c.labels[sid]}. Research question: with an error truly introduced, final accuracy is {100 * a['effect']:+.1f} pp with the allocator (95% CI {100 * a['lo']:+.1f} to {100 * a['hi']:+.1f}, "
             f"{sig(a['p'])}), and error propagation changes by {b['effect']:+.2f} steps ({sig(b['p'])})."]
        if v1:
            e = v1["effect"]
            tier = ("not distinguishable from chance" if v1["lo"] <= .5 else "clearly above chance but weak" if e < .65 else "a modest signal" if e < .7 else "an acceptable signal" if e < .8 else "a good signal")
            t.append(f"VQ1: the trust score ranks accurate above inaccurate steps with AUC {e:.2f} (95% CI {v1['lo']:.2f}-{v1['hi']:.2f}), {tier}.")
        if v2:
            t.append(f"VQ2: the injected message gets a trust score {abs(v2['effect']):.2f} points {'lower' if v2['effect'] < 0 else 'higher'} than the original ({pe(v2['p'])}, dz={v2['dz']:.2f}, "
                     f"lower in {v2['low']} of {v2['n']} pairs).")
        m = F.get((sid, "mis_fork")) or F.get((sid, "mis_baseline"))
        if m:
            t.append(f"Mistakes: when the allocator misjudged the {'injected' if (sid, 'mis_fork') in F else 'first'} message, final accuracy with the allocator changed by {100 * m['d_mis']:+.1f} pp relative to the allocator-off run (n={m['n_mis']}, {pe(m['p_within'])}), versus {100 * m['d_ok']:+.1f} pp when it did not ({pe(m['p'])} for the contrast).")
        i = F.get((sid, "ctx_interaction"))
        if i:
            t.append(f"Prompt length: the interaction between trust and log prompt length is {i['coef']:+.3f} ({pe(i['p'])}).")
        s.p(" ".join(t))
    s.p("How to read this: a null result for the research question is only informative together with the other findings. If injected errors barely lower accuracy (see the context table in the research-question "
        "chapter), the debate already recovers on its own and the allocator has little to improve; the validation questions then show whether the allocator would be able to help if errors did propagate.")


def limitations(c: Ctx, rep: Report) -> None:
    s = rep.section("Limitations and threats to validity", 1)
    for t in ("The two cases are not a controlled experiment on the judge/agent size ratio alone: they also differ in model family, backend (cloud or local) and possibly in the number of web-search attempts that succeeded. Differences between the cases are associations.",
              "Judge and agents come from the same model family within each case, so a shared blind spot (for example confidently wrong answers) can make the judge agree with a wrong agent; the allocator's AUC is therefore a lower bound for an independent judge.",
              "Step and final accuracy use exact-match GAIA scoring after answer extraction. Steps without a FINAL ANSWER count as inaccurate in step accuracy and error propagation, but are excluded from the trust analyses.",
              "A single injection (message 1, one failure mode chosen by the injector LLM) is tested. Failure modes are not randomised; the injector picks the mode that fits, so modes differ in task mix and some modes are rare.",
              "Error propagation is right-censored at the last step: runs that never recover are counted at their last step, which is conservative for the allocator-off and allocator-on arms alike.",
              "Subgroup tests (level, failure mode, per-maxim, per-bin) are descriptive and uncorrected; only the headline tests carry a Holm adjustment.",
              "The judge's context limit is taken from the recorded num_ctx where available and otherwise assumed per backend; if the real limit was lower, the 'fill' figures understate how close the prompts came.",
              "Token and time costs exclude the shared first message and the injector call, and wall-clock time is unreliable for resumed runs."):
        s.p("- " + t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", type=Path, nargs="+", help="results folder(s); scanned recursively for <task>/<condition>.json")
    ap.add_argument("--out", type=Path, help="output folder (default: <root>/analysis for one root, ./analysis for several)")
    ap.add_argument("--high", type=float, default=4.5, help="trust >= high counts as 'high trust' (mistake analysis)")
    ap.add_argument("--low", type=float, default=3.0, help="trust < low counts as 'low trust' (mistake analysis)")
    ap.add_argument("--ctx-limit", default="", help="judge context window when the records do not store num_ctx, e.g. cloud=262144,local=32768")
    ap.add_argument("--no-pdf", action="store_true", help="write report.tex only, do not run pdflatex")
    a = ap.parse_args()
    lim = {k: int(v) for k, v in (kv.split("=") for kv in a.ctx_limit.split(",") if kv)}
    runs, steps, tasks = load(a.roots, lim)
    out = a.out or (a.roots[0] / "analysis" if len(a.roots) == 1 else Path("analysis"))
    out.mkdir(parents=True, exist_ok=True)
    runs.drop(columns=["orig_text", "inj_text", "first_text"]).to_csv(out / "runs.csv", index=False)
    steps.to_csv(out / "steps.csv", index=False)
    tasks.to_csv(out / "tasks.csv", index=False)
    core.style()
    c = Ctx(runs, steps, tasks, label_setups(runs), a.high, a.low)
    rep = Report(out, "Inter-agent trust and the resilience of LLM debate: analysis report")
    for chapter in (eda, rq, vq1, vq2, mistakes, ctxlen, head_to_head):
        try:
            chapter(c, rep)
        except Exception as e:  # one broken chapter (unexpected data) must not lose the others
            print(f"WARNING: chapter {chapter.__name__} failed: {type(e).__name__}: {e}")
            rep.section(f"{chapter.__name__} could not be completed", 1).p(f"This analysis stopped with {type(e).__name__}: {e}. The remaining chapters are unaffected.")
    limitations(c, rep)
    summary(c, rep)
    rep.sections.insert(0, rep.sections.pop())  # the summary goes first
    for m in rep.write(not a.no_pdf):
        print(m)
    print(f"{len(runs)} runs, {len(steps)} steps, {len(c.ids)} setup(s) -> {out}/ (report.tex, report.md, figures/, tables/)")


if __name__ == "__main__":
    main()