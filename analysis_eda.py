"""Exploratory data analysis (report chapter 2): what is in the dataset, before any hypothesis is tested."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy import stats

from analysis_core import COLORS, GREY, HATCH, INK, LIGHT, Ctx, Report, wilson

COND = [("baseline", "off", "baseline"), ("baseline", "on", "+ allocator"), ("fork", "off", "injected"), ("fork", "on", "injected + allocator")]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f1f0ec", "#eb6834"])  # two hues + neutral midpoint


def panels(c: Ctx, w: float = 4.6, h: float = 3.6, sharey: bool = True, n: int | None = None):
    n = n or c.n
    fig, ax = plt.subplots(1, n, figsize=(w * n, h), sharey=sharey, squeeze=False)
    return fig, ax[0]


def cond_legend(ax, color=GREY, loc="best", extra=()):
    ax.legend(handles=[Patch(fc=color, ec=INK, lw=.6, label="allocator off"), Patch(fc=color, ec=INK, lw=.6, hatch=HATCH["on"], label="allocator on"),
                       Patch(fc=color, alpha=.5, ec=INK, lw=.6, label="error injected (lighter)"), *extra], loc=loc, fontsize=7)


def eda(c: Ctx, rep: Report) -> None:
    ch = rep.section("Exploratory data analysis", 1, "What does the dataset look like before we test anything?")
    ch.p("This chapter describes the data: who the two setups are, how complete the runs are, how hard the tasks were, what the injector "
         "did, what the trust allocator's scores look like and what everything cost. Nothing here is a hypothesis test.")
    setups(c, rep)
    quality(c, rep)
    accuracy(c, rep)
    trajectory(c, rep)
    injection(c, rep)
    trust_dist(c, rep)
    if c.runs.prompt_tokens.sum() > 0:  # records written before token stats existed have no call data
        cost(c, rep)
        context(c, rep)
    else:
        rep.section("Cost and prompt lengths", 2).p("The records contain no per-call token statistics, so cost and prompt-length analyses are skipped.")
    drivers(c, rep)
    correlations(c, rep)


# ------------------------------------------------------------------------------------------
def setups(c: Ctx, rep: Report) -> None:
    s = rep.section("The two setups and the dataset", 2, "What exactly is being compared?")
    rows = []
    for sid in c.ids:
        i = c.runs[c.runs.setup_id == sid].iloc[0]
        t = c.tasks[c.tasks.setup_id == sid]
        rows.append({"case": c.short[sid], "agents": i.agent_tag, "trust judge / injector": i.judge_tag,
                     "judge / agent size": f"{i.judge_ratio:.1f}x" if i.judge_ratio == i.judge_ratio and i.judge_ratio else "?",
                     "tasks": len(t), "level 1 / 2 / 3": " / ".join(str(int((t.level == k).sum())) for k in (1, 2, 3)),
                     "with attachment": int(t.has_file.sum()), "backend(s)": ", ".join(sorted(c.runs[c.runs.setup_id == sid].backend.dropna().unique()))})
    s.table(pd.DataFrame(rows), "setups", "The experimental setups (one row per case)",
            "Case 1 stands in for a strong judge supervising equally strong agents; case 2 for a judge about three times the size of the agents it supervises.")
    s.p("Each task is run under four conditions: (1) baseline, (2) trust allocator without error, (3) injected error without allocator, (4) injected error with "
        "allocator. The injected error corrupts Agent 1's first message once per task and the very same corrupted message is used in conditions 3 and 4. "
        "Tasks come from GAIA, in three difficulty levels (1 = easiest). All injected-run analyses use only tasks where the injector found an eligible "
        "failure mode.")


def quality(c: Ctx, rep: Report) -> None:
    s = rep.section("Completeness and data quality", 2, "Which runs and tasks can be used, and why are the others dropped?")
    a, rows = c.all_runs, []
    for sid in c.ids:
        r, t = a[a.setup_id == sid], c.tasks[c.tasks.setup_id == sid]
        rows.append({"case": c.short[sid], "tasks": len(t), "runs": len(r), "failed runs": int((~r.ok).sum()),
                     "tasks with all 4 runs ok": int(t.all_four.sum()), "ineligible injections": int(((r.kind == "fork") & r.ok & ~r.eligible & (r.trust == "on")).sum()),
                     "RQ pairs (identical injection)": int((t.rq_reason == "").sum()), "VQ2 pairs": int((t.vq2_reason == "").sum()),
                     "steps without FINAL ANSWER": int(r[r.ok].n_no_answer.sum()), "unusable audits": int(r[r.ok].n_audit_unusable.sum()),
                     "budget-lean agent turns": int(r[r.ok].lean_turns.sum()), "failed LLM attempts": int(r.n_failed_calls.sum())})
    s.table(pd.DataFrame(rows), "completeness", "Run completeness and the size of the paired analyses", transpose=True)
    ex = []
    for sid in c.ids:
        t = c.tasks[c.tasks.setup_id == sid]
        for name, col in (("RQ pairs (allocator off vs on)", "rq_reason"), ("VQ2 pairs (original vs injected message)", "vq2_reason")):
            for reason, n in t[t[col] != ""][col].value_counts().items():
                ex.append({"case": c.short[sid], "analysis": name, "reason for exclusion": reason, "tasks": int(n)})
    s.table(pd.DataFrame(ex), "exclusions", "Tasks excluded from the paired analyses, with the reason",
            "A VQ2 pair needs (a) an exact text match between the clean Agent 1 message and the original stored with the injection, and (b) a valid trust score in both runs. "
            "An RQ pair needs both injected runs to share the same injection (same corrupted text and failure mode).")
    d = pd.DataFrame([{"case": c.short[sid], "condition": f"{k}_{tr}", "failed": int((~a[(a.setup_id == sid) & (a.condition == f'{k}_{tr}')].ok).sum()),
                       "of": int(((a.setup_id == sid) & (a.condition == f"{k}_{tr}")).sum())} for sid in c.ids for k, tr, _ in COND])
    s.table(d, "failed_by_condition", "Failed runs by condition")
    bk = c.r(c.ids[0]).backend.nunique() > 1 or any(c.r(i).backend.nunique() > 1 for i in c.ids)
    if bk:
        b = c.runs[(c.runs.kind == "baseline") & (c.runs.trust == "off")].groupby(["setup_id", "backend"]).final_correct.agg(["mean", "count"]).reset_index()
        b["case"] = b.setup_id.map(c.short)
        s.table(b[["case", "backend", "count", "mean"]].rename(columns={"count": "baseline runs", "mean": "baseline accuracy"}), "backend_mix",
                "Accuracy by inference backend (some results came from the cloud, some from a local cluster)",
                "A setup that mixes backends is only comparable if accuracy does not depend on the backend; large gaps here are a threat to validity.")


def accuracy(c: Ctx, rep: Report) -> None:
    s = rep.section("Accuracy by condition and GAIA level", 2, "How accurate is each condition, and how does it depend on task difficulty?")
    fig, axs = panels(c, 5.2, 3.7)
    rows = []
    for ax, sid in zip(axs, c.ids):
        groups = [("All", None), ("L1", 1), ("L2", 2), ("L3", 3)]
        w = 0.2
        for gi, (gname, lv) in enumerate(groups):
            for ci, (kind, tr, cname) in enumerate(COND):
                d = c.r(sid, kind, tr, eligible=True if kind == "fork" else None)
                d = d if lv is None else d[d.level == lv]
                p, lo, hi = wilson(int(d.final_correct.sum()), len(d))
                ax.bar(gi + (ci - 1.5) * w, p, w * 0.92, color=c.color[sid], alpha=1 if kind == "baseline" else 0.5, hatch=HATCH[tr], ec=INK, lw=.5)
                if len(d):
                    ax.errorbar(gi + (ci - 1.5) * w, p, yerr=[[p - lo], [hi - p]], color=INK, lw=.8, capsize=1.5)
                rows.append({"case": c.short[sid], "level": gname, "condition": cname, "n": len(d), "accuracy": p, "ci_low": lo, "ci_high": hi})
        n = {lv: int((c.tasks[(c.tasks.setup_id == sid)].level == lv).sum()) for lv in (1, 2, 3)}
        ax.set_xticks(range(4), [f"All\n(n={len(c.tasks[c.tasks.setup_id == sid])})"] + [f"Level {k}\n(n={n[k]})" for k in (1, 2, 3)])
        ax.set_title(c.labels[sid], fontsize=8)
        ax.set_ylim(0, 1.05)
    axs[0].set_ylabel("final-answer accuracy (95% Wilson CI)")
    cond_legend(axs[-1], loc="upper right")
    s.fig(fig, "eda_accuracy_by_level", "Final-answer accuracy per condition and GAIA level. Error bars are Wilson 95% intervals; injected bars use only tasks with an eligible injection.")
    s.table(pd.DataFrame(rows).assign(accuracy=lambda d: d.accuracy.round(3), ci_low=lambda d: d.ci_low.round(3), ci_high=lambda d: d.ci_high.round(3)),
            "accuracy_by_level", "Final-answer accuracy per condition and level")
    s.p("Read the figure as a difficulty profile: accuracy should fall from level 1 to level 3 in every condition. Bars of different conditions inside one group that "
        "are close together (overlapping intervals) are the first hint that neither the injected error nor the allocator changes the final answer much; the formal tests follow in the research-question chapter.")


def trajectory(c: Ctx, rep: Report) -> None:
    s = rep.section("How the debate unfolds step by step", 2, "Does accuracy improve over the debate, and where does the injected error show up?")
    fig, axs = panels(c, 5.0, 3.6)
    rows = []
    for ax, sid in zip(axs, c.ids):
        for (kind, tr, cname), ls, mk, al in zip(COND, ("-", "--", "-", "--"), "oo^^", (1, 1, .55, .55)):
            st = c.steps[(c.steps.setup_id == sid) & (c.steps.kind == kind) & (c.steps.trust == tr)]
            if kind == "fork":
                st = st[st.task_id.isin(c.r(sid, "fork", eligible=True).task_id)]
            g = st.groupby("step").step_correct.agg(["sum", "count"])
            ci = np.array([wilson(int(a), int(b)) for a, b in zip(g["sum"], g["count"])])
            ax.plot(g.index, ci[:, 0], ls, marker=mk, color=c.color[sid], alpha=al, label=cname, ms=4)
            ax.fill_between(g.index, ci[:, 1], ci[:, 2], color=c.color[sid], alpha=.08, lw=0)
            rows += [{"case": c.short[sid], "condition": cname, "step": int(i), "accuracy": ci[j, 0], "n": int(g["count"].iloc[j])} for j, i in enumerate(g.index)]
        ax.set_title(c.labels[sid], fontsize=8)
        ax.set_xlabel("step in the debate (0 = Agent 1's first message)")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7, loc="lower right")
    axs[0].set_ylabel("share of accurate steps")
    s.fig(fig, "eda_step_accuracy", "Step accuracy by position in the debate. Dashed = allocator on, triangles/lighter = error injected at step 0.")
    s.table(pd.DataFrame(rows).pivot_table(index=["case", "condition"], columns="step", values="accuracy").round(2), "step_accuracy", "Step accuracy by position")
    s.p("Step 0 is the (possibly corrupted) first message. A dip at step 0 in the injected lines that closes by step 1 or 2 means the other agents repair the error; "
        "a flat clean line means the debate does not add accuracy beyond what the first answer already had.")


def injection(c: Ctx, rep: Report) -> None:
    s = rep.section("The injected errors", 2, "What did the injector do, and did the corruption actually damage the message?")
    f = c.runs[(c.runs.kind == "fork") & (c.runs.trust == "on") & c.runs.eligible]
    fms = sorted(f.fm_id.dropna().unique())
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    w = 0.8 / c.n
    rows = []
    for i, sid in enumerate(c.ids):
        d = f[f.setup_id == sid]
        cnt = d.fm_id.value_counts().reindex(fms).fillna(0)
        ax[0].bar(np.arange(len(fms)) + (i - (c.n - 1) / 2) * w, cnt, w * .92, color=c.color[sid], label=c.short[sid])
        wrong = d.groupby("fm_id").inj_step_correct.apply(lambda x: 1 - x.astype(float).mean()).reindex(fms)
        ax[1].bar(np.arange(len(fms)) + (i - (c.n - 1) / 2) * w, wrong, w * .92, color=c.color[sid])
        ax[2].boxplot(d.words_changed.dropna(), positions=[i], widths=.5, patch_artist=True, boxprops=dict(fc=c.color[sid], alpha=.6), medianprops=dict(color=INK))
        rows += [{"case": c.short[sid], "failure mode": k, "injections": int(cnt[k]), "share making the step wrong": wrong[k],
                  "median words changed": d[d.fm_id == k].words_changed.median()} for k in fms]
    for a in ax[:2]:
        a.set_xticks(range(len(fms)), fms, rotation=20)
    ax[0].set_title("failure mode chosen by the injector (n tasks)")
    ax[1].set_title("share of injected messages whose answer is wrong")
    ax[1].set_ylim(0, 1.05)
    ax[2].set_xticks(range(c.n), [c.short[i] for i in c.ids])
    ax[2].set_title("share of words changed by the corruption")
    ax[0].legend(fontsize=8)
    s.fig(fig, "eda_injection", "What the injector chose and how destructive it was (one eligible injection per task).")
    s.table(pd.DataFrame(rows).round(2), "injection_by_fm", "Injection statistics by failure mode")
    lv = f.groupby(["setup_id", "level"]).size().unstack(fill_value=0)
    s.table(lv.rename(index=c.short).rename(columns=lambda x: f"level {x}").reset_index().rename(columns={"setup_id": "case"}), "injection_by_level", "Eligible injections per GAIA level")
    s.p("A failure mode only damages the task if the corrupted message ends up with a wrong final answer. FM-2.6 (inconsistent reasoning) often keeps a plausible-looking "
        "explanation around a wrong answer, FM-2.3 (drift) changes the topic. Failure modes that are chosen only a handful of times cannot support conclusions of their own.")


def trust_dist(c: Ctx, rep: Report) -> None:
    s = rep.section("The trust scores", 2, "What do the allocator's scores look like?")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    bins = np.arange(1, 5.26, .25)
    rows = []
    for i, sid in enumerate(c.ids):
        t = c.clean_on(sid, answered=False)
        ax[0].hist(t.trust_score, bins, alpha=.55, color=c.color[sid], label=c.short[sid], density=True)
        ax[1].boxplot([t[t.step == k].trust_score for k in range(6)], positions=np.arange(6) + (i - (c.n - 1) / 2) * .3, widths=.25, patch_artist=True,
                      boxprops=dict(fc=c.color[sid], alpha=.6), medianprops=dict(color=INK), flierprops=dict(ms=2))
        for m in ("quality", "quantity", "relation", "manner"):
            rows.append({"case": c.short[sid], "maxim": m, "mean score": t[f"maxim_{m}"].mean(), "share = 5": (t[f"maxim_{m}"] == 5).mean()})
        ax[2].bar(np.arange(4) + (i - (c.n - 1) / 2) * .35, [t[f"maxim_{m}"].mean() for m in ("quality", "quantity", "relation", "manner")], .33, color=c.color[sid])
        rows.append({"case": c.short[sid], "maxim": "TRUST (mean of 4)", "mean score": t.trust_score.mean(), "share = 5": (t.trust_score == 5).mean()})
    ax[0].set(title="trust score, clean runs (density)", xlabel="trust score (1-5)")
    ax[0].legend(fontsize=8)
    ax[1].set(title="trust score by step position (clean runs)", xlabel="step", xticks=range(6), xticklabels=range(6))
    ax[2].set(title="mean maxim score", xticks=range(4), xticklabels=["quality", "quantity", "relation", "manner"], ylim=(1, 5.1))
    s.fig(fig, "eda_trust_distribution", "Distribution of trust scores in clean allocator-on runs.")
    s.table(pd.DataFrame(rows).round(3), "trust_by_maxim", "Mean score per Gricean maxim (clean allocator-on steps)")
    s.p("A score distribution piled up at 5 is a ceiling: the allocator then has little room to separate good from bad steps, which limits what the validation "
        "questions can show. The maxims are the four components averaged into the trust score.")


def cost(c: Ctx, rep: Report) -> None:
    s = rep.section("Cost: tokens, calls and time", 2, "What does each condition cost, and what does the allocator add?")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    rows = []
    for j, (col, ttl, lg) in enumerate((("total", "tokens per run", True), ("llm_seconds", "LLM seconds per run", True))):
        for i, sid in enumerate(c.ids):
            data, pos = [], []
            for k, (kind, tr, cname) in enumerate(COND):
                d = c.r(sid, kind, tr)
                v = (d.prompt_tokens + d.completion_tokens) if col == "total" else d[col]
                data.append(v[v > 0]); pos.append(k + (i - (c.n - 1) / 2) * .35)
            ax[j].boxplot(data, positions=pos, widths=.3, patch_artist=True, boxprops=dict(fc=c.color[sid], alpha=.6), medianprops=dict(color=INK), flierprops=dict(ms=2))
        ax[j].set(title=ttl, xticks=range(4), xticklabels=[x[2].replace(" + ", "\n+ ").replace("injected ", "injected\n") for x in COND])
        ax[j].set_yscale("log")
        if j == 0:
            ax[j].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
            ax[j].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for i, sid in enumerate(c.ids):
        bot = np.zeros(4)
        for role, a in (("tok_agent", 1), ("tok_audit", .45)):
            v = np.array([c.r(sid, k, t)[role].mean() for k, t, _ in COND])
            ax[2].bar(np.arange(4) + (i - (c.n - 1) / 2) * .35, v, .33, bottom=bot, color=c.color[sid], alpha=a, ec=INK, lw=.4)
            bot = bot + v
        ax[2].set_xticks(range(4), ["base", "+alloc", "inj", "inj+alloc"])
    ax[2].set_title("mean tokens per run: agents (solid) vs judge (light)")
    ax[2].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    for sid in c.ids:
        for kind, tr, cname in COND:
            d = c.r(sid, kind, tr)
            rows.append({"case": c.short[sid], "condition": cname, "runs": len(d), "mean tokens (k)": (d.prompt_tokens + d.completion_tokens).mean() / 1e3,
                         "of which judge (k)": d.tok_audit.mean() / 1e3, "mean LLM minutes": d.llm_seconds.mean() / 60, "mean calls": d.n_calls.mean(),
                         "median max agent prompt (k tok)": d.max_agent_prompt.median() / 1e3})
    s.fig(fig, "eda_cost", "Cost per run. Log scale for the box plots; boxes are coloured by case.")
    tab = pd.DataFrame(rows)
    s.table(tab.round(2), "cost_by_condition", "Cost per run by condition")
    s.p("The allocator adds one judge call per message; the 'judge' share shows its token overhead relative to the agents' own work. Token counts exclude the shared first "
        "message and the injector call, which are produced once per task. Wall-clock time for resumed runs is not meaningful, so LLM seconds are used.")


def context(c: Ctx, rep: Report) -> None:
    s = rep.section("Prompt and context lengths", 2, "How long are the prompts the agents and the judge receive, relative to their context window?")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    rows = []
    for sid in c.ids:
        ag = c.runs[(c.runs.setup_id == sid)].max_agent_prompt
        ju = c.steps[(c.steps.setup_id == sid) & c.steps.audit_prompt_tokens.notna()]
        lim = c.runs[c.runs.setup_id == sid].ctx_limit.median()
        for a, v, nm in ((ax[0], ag[ag > 0], "agent"), (ax[1], ju.audit_prompt_tokens, "judge")):
            a.hist(np.log10(v.clip(lower=1)), bins=30, alpha=.55, color=c.color[sid], label=c.short[sid])
            a.axvline(np.log10(lim), color=c.color[sid], ls=":", lw=1.2)
        rows += [{"case": c.short[sid], "who": "agent (max per run)", "median": ag.median(), "p90": ag.quantile(.9), "max": ag.max(), "context limit": lim,
                  "share of runs above 50% of limit": (ag > .5 * lim).mean()},
                 {"case": c.short[sid], "who": "judge (per audit)", "median": ju.audit_prompt_tokens.median(), "p90": ju.audit_prompt_tokens.quantile(.9),
                  "max": ju.audit_prompt_tokens.max(), "context limit": lim, "share of runs above 50% of limit": (ju.audit_prompt_tokens > .5 * lim).mean()}]
    for a, t in zip(ax, ("agent: longest prompt per run", "judge: prompt per audit")):
        a.set(title=t, xlabel="prompt tokens (log10)", ylabel="count")
        a.set_xticks([2, 3, 4, 5], ["100", "1k", "10k", "100k"])
    ax[0].legend(fontsize=8, title="dotted = context limit", title_fontsize=7)
    s.fig(fig, "eda_context_lengths", "Prompt length distributions. Dotted vertical lines mark each setup's context limit.")
    s.table(pd.DataFrame(rows).round(2), "context_lengths", "Prompt length summary (tokens)",
            "Context limit = num_ctx stored in the record, otherwise a default per backend (cloud 262144, local 32768; override with --ctx-limit).")


def drivers(c: Ctx, rep: Report) -> None:
    s = rep.section("What makes a task hard?", 2, "Do attachments or question length relate to accuracy (clean baseline)?")
    fig, ax = panels(c, 4.3, 3.4, n=2)
    rows = []
    for sid in c.ids:
        b = c.r(sid, "baseline", "off")
        q = pd.qcut(b.question_chars.rank(method="first"), 3, labels=["short", "medium", "long"])
        for a, grp, gl in ((ax[0], b.has_file.map({True: "with file", False: "no file"}), "attachment"), (ax[1], q, "question length")):
            for k, (name, g) in enumerate(b.groupby(grp, observed=True)):
                p, lo, hi = wilson(int(g.final_correct.sum()), len(g))
                a.bar(k + list(c.ids).index(sid) * .38 - .19, p, .35, color=c.color[sid], ec=INK, lw=.4)
                a.errorbar(k + list(c.ids).index(sid) * .38 - .19, p, yerr=[[p - lo], [hi - p]], color=INK, lw=.8, capsize=1.5)
                rows.append({"case": c.short[sid], "factor": gl, "group": name, "n": len(g), "accuracy": p})
            a.set_xticks(range(grp.nunique()), sorted(grp.dropna().unique(), key=str) if gl == "attachment" else ["short", "medium", "long"])
    ax[0].set(title="by attachment", ylabel="baseline accuracy")
    ax[1].set(title="by question length (tertiles)")
    ax[0].legend(handles=[Patch(fc=c.color[i], label=c.short[i]) for i in c.ids], fontsize=8)
    s.fig(fig, "eda_difficulty_drivers", "Baseline accuracy by attachment and question length.")
    s.table(pd.DataFrame(rows).round(3), "difficulty_drivers", "Baseline accuracy by task features")
    s.p("Attachments (images, spreadsheets, audio) and long questions are usually harder; if they differ in share between the two setups they could confound any "
        "cross-setup comparison, but within a setup the paired designs below hold task features fixed.")


def correlations(c: Ctx, rep: Report) -> None:
    s = rep.section("How the per-run measures relate", 2, "Which run-level measures move together?")
    fig, axs = panels(c, 5.2, 4.6, n=c.n, sharey=False)
    feats = {"final_correct": "final correct", "step_acc": "step accuracy", "level": "GAIA level", "question_chars": "question length", "has_file": "has attachment",
             "n_calls": "LLM calls", "tokens": "total tokens", "llm_seconds": "LLM seconds", "n_no_answer": "steps w/o answer"}
    for ax, sid in zip(axs, c.ids):
        d = c.r(sid, "baseline", "off").copy()
        d["tokens"] = d.prompt_tokens + d.completion_tokens
        m = d[list(feats)].astype(float).corr(method="spearman")
        im = ax.imshow(m.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
        ax.set_xticks(range(len(feats)), list(feats.values()), rotation=60, ha="right")
        ax.set_yticks(range(len(feats)), list(feats.values()))
        for i in range(len(feats)):
            for j in range(len(feats)):
                ax.text(j, i, f"{m.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color=INK)
        ax.set_title(c.labels[sid], fontsize=8)
        ax.grid(False)
    fig.set_layout_engine("constrained")
    fig.colorbar(im, ax=axs, shrink=.7, label="Spearman rho")
    s.fig(fig, "eda_correlations", "Spearman correlations between per-run measures (clean baseline runs, no allocator).")
    s.p("Typical pattern: harder levels, longer runs and more tokens go with lower accuracy, and steps without a FINAL ANSWER go with token-hungry runs that hit a limit. "
        "Correlation is descriptive: it does not say which variable drives which.")
