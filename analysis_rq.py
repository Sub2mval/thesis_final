"""Chapter 4: the main research question (does the trust allocator increase resiliency?) and the case 1 vs case 2 comparison."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analysis_core import GREY, INK, Ctx, Report, boot_mean_ci, mcnemar, mde_pp, pe, pv, sign_test, wilcoxon, wilson
from analysis_eda import HATCH, panels

SUBSETS = [("all", "all paired tasks"), ("introduced", "error truly introduced (original message correct)"),
           ("effective", "effective injection (original correct, injected message wrong)")]


def subset(p: pd.DataFrame, key: str) -> pd.DataFrame:
    if key == "all":
        return p
    q = p[p.orig_correct == True]  # noqa: E712 (object column with NaN)
    return q if key == "introduced" else q[~q.inj_step_correct_off]


def forest(ax, rows: list, xlabel: str, title: str = "") -> None:
    """rows = [(label, estimate, lo, hi, colour)] drawn top to bottom, zero line marked."""
    for i, (lab, e, lo, hi, col) in enumerate(rows[::-1]):
        if e == e:
            ax.errorbar(e, i, xerr=[[e - lo], [hi - e]] if lo == lo else None, fmt="o", color=col, capsize=2, ms=5)
    ax.axvline(0, color=INK, lw=.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=7.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", visible=False)


def rq(c: Ctx, rep: Report) -> None:
    ch = rep.section("Main research question: does the trust allocator increase resiliency?", 1,
                     "Does adding the trust allocator make the debate recover faster from an injected error (step level) and end with a more accurate answer (overall)?")
    ch.p("Design: for every task the same corrupted first message is run twice, once without and once with the trust allocator, so each task is its own control and "
         "all tests are paired by task. 'Resiliency' is measured in two ways, as on the thesis slides. Step level: error propagation = the number of inaccurate steps in a row "
         "starting at the injected message (fewer = faster recovery; 0 means the injected message was still accurate). Overall: the accuracy of the final answer. "
         "Because the injection only matters when the original message was right, each result is shown for three nested groups of tasks: all paired tasks; tasks where the "
         "original message was correct (an error was truly introduced); and tasks where, in addition, the injected message is wrong (the injection was effective).")
    ch.p("Only tasks whose two injected runs share exactly the same corrupted message enter these tests (the exclusions are listed in the exploratory chapter). Differences are 'on minus off'.")
    P = {sid: c.rq_pairs(sid) for sid in c.ids}
    overall(c, rep, P)
    steplevel(c, rep, P)
    context_effects(c, rep)
    moderators(c, rep, P)
    mechanism(c, rep, P)


def overall(c, rep, P) -> None:
    s = rep.section("Overall: final-answer accuracy", 2, "Is the final answer more often correct with the allocator?")
    rows, fr = [], []
    for sid in c.ids:
        for key, name in SUBSETS:
            q = subset(P[sid], key)
            off, on = q.final_correct_off.to_numpy(bool), q.final_correct_on.to_numpy(bool)
            m = mcnemar(off, on) if len(q) else {"only_ref": 0, "only_comp": 0, "p": np.nan}
            d, lo, hi = boot_mean_ci(on.astype(float) - off.astype(float)) if len(q) else (np.nan,) * 3
            rows.append({"case": c.short[sid], "tasks": name, "pairs": len(q), "acc. off": off.mean() if len(q) else np.nan, "acc. on": on.mean() if len(q) else np.nan,
                         "diff (pp)": 100 * d, "95% CI (pp)": f"[{100 * lo:.1f}, {100 * hi:.1f}]", "only off right": m["only_ref"], "only on right": m["only_comp"],
                         "McNemar p": pv(m["p"]), "80%-power MDE (pp)": mde_pp(m["only_ref"] + m["only_comp"], len(q))})
            fr.append((sid, key, f"{c.short[sid]} - {name.split(' (')[0]}", 100 * d, 100 * lo, 100 * hi))
            rep.findings[(sid, f"rq_acc_{key}")] = {"effect": d, "lo": lo, "hi": hi, "p": m["p"], "n": len(q), "b": m["only_ref"], "c": m["only_comp"]}
    t = pd.DataFrame(rows)
    s.table(t.round({"acc. off": 2, "acc. on": 2, "diff (pp)": 1, "80%-power MDE (pp)": 1}), "rq_overall", "Final accuracy with the allocator off vs on (paired by task)",
            "MDE = approximate smallest accuracy difference this paired test could detect with 80% power given the observed number of discordant tasks.")
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(fr) + 1.2))
    forest(ax, [(lab, e, lo, hi, c.color[sid]) for sid, k, lab, e, lo, hi in fr], "accuracy difference, allocator on minus off (percentage points, 95% bootstrap CI)")
    s.fig(fig, "rq_overall_forest", "Effect of the allocator on final accuracy. Points to the right of zero favour the allocator; an interval around zero means no detectable effect.")
    fig, axs = panels(c, 4.6, 3.3, n=c.n)
    for ax, sid in zip(axs, c.ids):
        q = subset(P[sid], "all")
        tab = pd.crosstab(q.final_correct_off.map({True: "off: right", False: "off: wrong"}), q.final_correct_on.map({True: "on: right", False: "on: wrong"}))
        tab = tab.reindex(index=["off: right", "off: wrong"], columns=["on: right", "on: wrong"], fill_value=0)
        ax.imshow(tab.to_numpy(), cmap="Greys", vmin=0, vmax=max(1, tab.to_numpy().max()) * 1.6)
        for (i, j), v in np.ndenumerate(tab.to_numpy()):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=14, color=INK)
        ax.set_xticks([0, 1], tab.columns)
        ax.set_yticks([0, 1], tab.index)
        ax.grid(False)
        ax.set_title(c.labels[sid], fontsize=8)
    s.fig(fig, "rq_overall_transitions", "Paired outcomes per task (all paired tasks): the off-diagonal cells are the only ones that carry information about the allocator.")
    a = P[c.ids[0]]
    s.test("Exact McNemar test (paired binomial test on the discordant tasks), with a bootstrap CI for the accuracy difference",
           "The outcome (final answer right or wrong) is binary and measured twice on the same task, so only tasks where the two runs disagree carry information; "
           "the exact version is used because there are few such tasks. The bootstrap resamples tasks to put an interval on the accuracy difference.",
           "A p-value below 0.05 would mean the allocator changes accuracy more than chance explains; a large p-value with a narrow interval around 0 means 'no effect larger than the interval', "
           "while a wide interval means the data cannot rule out a meaningful effect either way (compare with the MDE column).",
           " | ".join(f"{c.short[i]} (all): {rep.findings[(i, 'rq_acc_all')]['effect'] * 100:+.1f} pp, {pe(rep.findings[(i, 'rq_acc_all')]['p'])}; "
                      f"error-introduced: {rep.findings[(i, 'rq_acc_introduced')]['effect'] * 100:+.1f} pp, {pe(rep.findings[(i, 'rq_acc_introduced')]['p'])}" for i in c.ids))


def steplevel(c, rep, P) -> None:
    s = rep.section("Step level: recovery from the injected error", 2, "Does the debate recover faster (fewer inaccurate steps in a row) with the allocator?")
    rows = []
    for sid in c.ids:
        for key, name in SUBSETS:
            q = subset(P[sid], key)
            if not len(q):
                continue
            off, on = q.propagation_off.to_numpy(float), q.propagation_on.to_numpy(float)
            d, lo, hi = boot_mean_ci(on - off)
            pw = wilcoxon(on, off)
            eff = q[~q.inj_step_correct_off]
            r1 = mcnemar((eff.propagation_off <= 1).to_numpy(), (eff.propagation_on <= 1).to_numpy()) if len(eff) else {"p": np.nan}
            nr = mcnemar(~eff.censored_off.to_numpy(), ~eff.censored_on.to_numpy()) if len(eff) else {"p": np.nan}
            rows.append({"case": c.short[sid], "tasks": name.split(" (")[0], "pairs": len(q), "mean propagation off": off.mean(), "on": on.mean(),
                         "diff": d, "95% CI": f"[{lo:.2f}, {hi:.2f}]", "Wilcoxon p": pv(pw), "never recovered off/on": f"{int(q.censored_off.sum())}/{int(q.censored_on.sum())}",
                         "recovered next step: McNemar p": pv(r1["p"]) if len(eff) else "n/a"})
            rep.findings[(sid, f"rq_prop_{key}")] = {"effect": d, "lo": lo, "hi": hi, "p": pw, "n": len(q)}
    s.table(pd.DataFrame(rows).round(2), "rq_propagation", "Error propagation (inaccurate steps in a row from the injected step) with the allocator off vs on",
            "'Recovered next step' is evaluated on effective injections only: share of tasks whose step right after the injected one is accurate again. Never recovered = still inaccurate at the last step (censored).")
    fig, axs = panels(c, 4.8, 3.5, n=c.n)
    for ax, sid in zip(axs, c.ids):
        q = subset(P[sid], "introduced")
        vals = [0, 1, 2, 3, 4, 5, 6]
        for k, tr in enumerate(("off", "on")):
            v = q[f"propagation_{tr}"].astype(int).clip(upper=6)
            sh = v.value_counts(normalize=True).reindex(vals, fill_value=0)
            ax.bar(np.array(vals) + (k - .5) * .38, sh, .36, color=c.color[sid], hatch=HATCH[tr], ec=INK, lw=.5, label=f"allocator {tr}")
        ax.set_xticks(vals, [str(v) if v < 6 else "6+" for v in vals])
        ax.set_xlabel("inaccurate steps in a row from the injected message")
        ax.set_title(f"{c.labels[sid]}\n(original message correct, n={len(q)})", fontsize=8)
        ax.legend(fontsize=7)
    axs[0].set_ylabel("share of tasks")
    s.fig(fig, "rq_propagation_hist", "Distribution of error propagation. A shift of mass to the left with the allocator (hatched) would mean faster recovery.")
    # recovery curves: accuracy of the steps k steps after the injected message
    fig, axs = panels(c, 4.8, 3.5, n=c.n)
    for ax, sid in zip(axs, c.ids):
        q = subset(P[sid], "introduced")
        for tr, ls, lab in (("off", "-", "injected, allocator off"), ("on", "--", "injected, allocator on")):
            st = c.steps[(c.steps.setup_id == sid) & (c.steps.kind == "fork") & (c.steps.trust == tr) & c.steps.task_id.isin(q.index)]
            g = st.groupby("rel_step").step_correct.agg(["sum", "count"])
            ci = np.array([wilson(int(a), int(b)) for a, b in zip(g["sum"], g["count"])])
            ax.plot(g.index, ci[:, 0], ls, marker="o", ms=4, color=c.color[sid], label=lab)
            ax.fill_between(g.index, ci[:, 1], ci[:, 2], color=c.color[sid], alpha=.08, lw=0)
        b = c.steps[(c.steps.setup_id == sid) & (c.steps.kind == "baseline") & (c.steps.trust == "off") & c.steps.task_id.isin(q.index)]
        ax.plot(sorted(b.step.unique()), b.groupby("step").step_correct.mean(), ":", color=GREY, label="clean baseline (no error)")
        ax.set_xlabel("steps after the injected message (0 = the injected message)")
        ax.set_ylim(0, 1.02)
        ax.set_title(c.labels[sid], fontsize=8)
        ax.legend(fontsize=7, loc="lower right")
    axs[0].set_ylabel("share of accurate steps")
    s.fig(fig, "rq_recovery_curve", "Recovery curve (tasks where the original message was correct): accuracy of the steps following the injected message.")
    s.test("Wilcoxon signed-rank test on the paired propagation counts (two-sided), plus a McNemar test on 'recovered at the next step'",
           "Propagation is a small count with many ties and a skewed distribution, so a paired rank test is safer than a t-test; the McNemar version collapses it to the one question "
           "that matters most for recovery speed (is the very next step accurate again?). Never-recovered runs are censored at the last step, which makes the means conservative.",
           "A significantly negative difference means faster recovery with the allocator. A zero difference with identical curves means the debate already repairs the error by itself, "
           "so there is nothing for the allocator to speed up.",
           " | ".join(f"{c.short[i]} (error introduced): {rep.findings[(i, 'rq_prop_introduced')]['effect']:+.2f} steps, {pe(rep.findings[(i, 'rq_prop_introduced')]['p'])}" for i in c.ids if (i, 'rq_prop_introduced') in rep.findings))


def context_effects(c, rep) -> None:
    s = rep.section("Context: what the injection and the allocator do on their own", 2, "How large are the injected error's damage and the allocator's cost, separately?")
    rows = []
    for sid in c.ids:
        f = c.runs[(c.runs.setup_id == sid)].pivot_table(index="task_id", columns=["kind", "trust"], values="final_correct", aggfunc="first")
        e = c.tasks[(c.tasks.setup_id == sid) & c.tasks.eligible & (c.tasks.rq_reason == "")].task_id
        for lab, a, b, idx in (("allocator cost without errors (baseline off vs on)", ("baseline", "off"), ("baseline", "on"), f.index),
                               ("effect of the injected error, allocator off (baseline vs fork)", ("baseline", "off"), ("fork", "off"), e),
                               ("effect of the injected error, allocator on (baseline vs fork)", ("baseline", "on"), ("fork", "on"), e)):
            g = f.loc[f.index.intersection(idx), [a, b]].dropna()
            if g.empty:
                continue
            m = mcnemar(g[a].astype(bool), g[b].astype(bool))
            rows.append({"case": c.short[sid], "comparison": lab, "tasks": m["n"], "accuracy reference": g[a].astype(float).mean(), "accuracy comparison": g[b].astype(float).mean(),
                         "diff (pp)": 100 * m["diff"], "only ref right": m["only_ref"], "only comp right": m["only_comp"], "McNemar p": pv(m["p"])})
            rep.findings[(sid, lab[:12])] = {"effect": m["diff"], "p": m["p"]}
    s.table(pd.DataFrame(rows).round(2), "rq_context", "Marginal effects (paired by task, exact McNemar)")
    s.test("Exact McNemar test (paired)", "Same task, same binary outcome, two conditions: the exact paired test is the natural comparison for each row.",
           "The second and third rows are a manipulation check: if injecting an error does not lower accuracy even without an allocator, the debate is already resilient and the "
           "main question has little room to show an effect. The first row shows whether the allocator itself helps or harms when nothing is wrong.",
           "see the table")


def moderators(c, rep, P) -> None:
    s = rep.section("Does the effect differ by task difficulty or failure mode?", 2, "Is there a subgroup where the allocator helps or hurts?")
    rows, fr = [], []
    for sid in c.ids:
        q0 = subset(P[sid], "all")
        for dim, label in (("level", "GAIA level"), ("fm_id", "failure mode")):
            for g, q in q0.groupby(dim):
                if len(q) < 3:
                    continue
                d, lo, hi = boot_mean_ci(q.final_correct_on.astype(float).to_numpy() - q.final_correct_off.astype(float).to_numpy())
                m = mcnemar(q.final_correct_off, q.final_correct_on)
                dp = (q.propagation_on - q.propagation_off).mean()
                rows.append({"case": c.short[sid], "by": label, "group": g, "pairs": len(q), "acc. off": q.final_correct_off.mean(), "acc. on": q.final_correct_on.mean(),
                             "diff (pp)": 100 * d, "McNemar p": pv(m["p"]), "d propagation": dp})
                if dim == "level":
                    fr.append((f"{c.short[sid]} - level {g} (n={len(q)})", 100 * d, 100 * lo, 100 * hi, c.color[sid]))
    s.table(pd.DataFrame(rows).round(2), "rq_moderators", "Allocator effect by GAIA level and by failure mode (descriptive; small groups)",
            "Subgroup tests are exploratory and not corrected for multiplicity; groups below three pairs are omitted.")
    if fr:
        fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(fr) + 1.2))
        forest(ax, fr, "accuracy difference, allocator on minus off (pp, 95% bootstrap CI)")
        s.fig(fig, "rq_by_level_forest", "Allocator effect on final accuracy by GAIA level (all paired tasks).")
    s.test("Per-group exact McNemar tests (descriptive)", "The same paired test as in the main analysis, repeated within a group; with few tasks per group it has little power.",
           "Treat an isolated significant subgroup with caution: with this many subgroups some will look significant by chance. Consistent direction across levels or failure modes is more informative than any single p-value.",
           "see the table")


def mechanism(c, rep, P) -> None:
    s = rep.section("Is the allocator's signal related to recovery?", 2, "Within allocator-on runs, do lower trust scores at the injected message go with faster or slower recovery?")
    fig, axs = panels(c, 4.6, 3.5, n=c.n)
    rows = []
    for ax, sid in zip(axs, c.ids):
        q = subset(P[sid], "introduced").dropna(subset=["trust_at_inj_on"])
        if len(q) < 4:
            continue
        rng = np.random.default_rng(0)
        ax.scatter(q.trust_at_inj_on, q.propagation_on + rng.uniform(-.12, .12, len(q)), s=14, color=c.color[sid], alpha=.6)
        rho = stats.spearmanr(q.trust_at_inj_on, q.propagation_on)
        ax.set(xlabel="trust score given to the injected message", title=f"{c.labels[sid]}\nrho={rho.statistic:.2f}, {pe(rho.pvalue)}")
        flagged = q.trust_at_inj_on < c.low
        rows.append({"case": c.short[sid], "pairs": len(q), "flagged (trust<%.1f)" % c.low: int(flagged.sum()), "Spearman rho (trust, propagation)": rho.statistic, "p": pv(rho.pvalue),
                     "mean propagation, flagged": q.propagation_on[flagged].mean() if flagged.any() else np.nan, "mean propagation, not flagged": q.propagation_on[~flagged].mean(),
                     "mean change vs allocator off, flagged": (q.propagation_on - q.propagation_off)[flagged].mean() if flagged.any() else np.nan})
    axs[0].set_ylabel("inaccurate steps in a row (jittered)")
    s.fig(fig, "rq_trust_vs_propagation", "Trust score at the injected message vs error propagation in allocator-on runs (tasks where the original message was correct).")
    s.table(pd.DataFrame(rows).round(2), "rq_mechanism", "Trust at the injected message and recovery")
    s.test("Spearman rank correlation", "Trust is nearly continuous and propagation a small count; Spearman needs no linearity or normality assumption.",
           "A positive correlation (higher trust, longer propagation) would mean the allocator's warning is followed by faster recovery. It does not prove the warning caused the recovery: "
           "an obviously corrupted message is also easier for the agents to catch on their own.", "see the table")


# ------------------------------------------------------------------------------------------ case 1 vs case 2
def head_to_head(c: Ctx, rep: Report) -> None:
    if c.n < 2:
        return
    ch = rep.section("Case 1 versus case 2", 1, "Does the answer depend on the judge/agent size relationship?")
    a, b = c.ids[:2]
    s = rep.section("Same tasks, two setups", 2, "How different are the two setups on the shared tasks?")
    f = c.runs[(c.runs.kind == "baseline") & (c.runs.trust == "off")].pivot(index="task_id", columns="setup_id", values="final_correct").dropna()
    if f.empty:
        s.p("The two setups have no task in common, so no paired comparison between them is possible.")
        return
    m = mcnemar(f[a].astype(bool), f[b].astype(bool))
    kappa = np.nan
    if len(f):
        po = (f[a] == f[b]).mean()
        p_exp = f[a].mean() * f[b].mean() + (1 - f[a].mean()) * (1 - f[b].mean())
        kappa = (po - p_exp) / (1 - p_exp) if p_exp < 1 else np.nan
    rows = [{"measure": "baseline accuracy", c.short[a]: f[a].astype(float).mean(), c.short[b]: f[b].astype(float).mean(), "paired tasks": len(f),
             "only case 1 right": m["only_ref"], "only case 2 right": m["only_comp"], "McNemar p": pv(m["p"]), "agreement kappa": kappa}]
    s.table(pd.DataFrame(rows).round(2), "h2h_baseline", "Baseline accuracy of the two setups on the tasks both completed")
    s.test("Exact McNemar test between setups, and Cohen's kappa for agreement", "Both setups answered the same tasks, so the comparison is paired; kappa shows whether the same tasks are hard for both.",
           "A significant McNemar p means one setup is more accurate; kappa near 1 means the same tasks are solved or failed in both, near 0 means the setups disagree as much as chance.",
           f"accuracy difference {100 * m['diff']:+.1f} pp, {pe(m['p'])}, kappa={kappa:.2f}")
    s2 = rep.section("Headline results side by side", 2, "How do the main results compare across the two cases?")
    rows = []
    for key, nm in (("rq_acc_introduced", "RQ overall: accuracy diff (pp), error-introduced tasks"), ("rq_prop_introduced", "RQ step level: propagation diff (steps)"),
                    ("vq1", "VQ1: AUC of trust for step inaccuracy"), ("vq2", "VQ2: mean trust drop injected vs original")):
        row = {"result": nm}
        for sid in c.ids:
            r = rep.findings.get((sid, key))
            if r:
                sc = 100 if "acc" in key else 1
                row[c.short[sid]] = f"{r['effect'] * sc:+.2f} [{r['lo'] * sc:+.2f}, {r['hi'] * sc:+.2f}], {pe(r['p'])}" if key != "vq1" else f"{r['effect']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}], {pe(r['p'])}"
        rows.append(row)
    s2.table(pd.DataFrame(rows), "h2h_headline", "Headline estimates with 95% intervals")
    fig, axs = plt.subplots(1, 4, figsize=(13, 2.6))
    for ax, (key, ttl, ref) in zip(axs, (("rq_acc_introduced", "RQ accuracy diff (pp)", 0), ("rq_prop_introduced", "RQ propagation diff (steps)", 0),
                                         ("vq1", "VQ1 AUC (0.5 = chance)", .5), ("vq2", "VQ2 trust drop (injected - original)", 0))):
        for i, sid in enumerate(c.ids):
            r = rep.findings.get((sid, key))
            if r:
                sc = 100 if "acc" in key else 1
                ax.errorbar(r["effect"] * sc, i, xerr=[[r["effect"] * sc - r["lo"] * sc], [r["hi"] * sc - r["effect"] * sc]], fmt="o", color=c.color[sid], capsize=3)
        ax.axvline(ref, color=INK, lw=.8)
        ax.set_yticks(range(c.n), [c.short[i] for i in c.ids])
        ax.set_ylim(-.6, c.n - .4)
        ax.set_title(ttl, fontsize=8.5)
        ax.grid(axis="y", visible=False)
    s2.fig(fig, "h2h_headline_forest", "Headline results of the two cases on separate scales (points with 95% intervals).")
    s2.p("Reading across the panels: if the intervals of the two cases overlap, the data show no clear difference between a judge of the same strength as the agents and a judge three times their size.")
    # differences between the setups, paired by task (resampling tasks)
    s3 = rep.section("Is the allocator effect different in the two cases?", 2, "Does a stronger judge change the allocator's effect, accuracy or detection power?")
    pa, pb = c.rq_pairs(a), c.rq_pairs(b)
    rows = []
    for key, nm in (("introduced", "error truly introduced"), ("all", "all paired tasks")):
        qa, qb = subset(pa, key), subset(pb, key)
        # tasks differ between setups (different task ids per case in general): compare effect sizes via independent bootstrap
        da = (qa.final_correct_on.astype(float) - qa.final_correct_off.astype(float)).to_numpy()
        db = (qb.final_correct_on.astype(float) - qb.final_correct_off.astype(float)).to_numpy()
        rng = np.random.default_rng(1)
        diff = [rng.choice(db, len(db)).mean() - rng.choice(da, len(da)).mean() for _ in range(4000)] if len(da) and len(db) else [np.nan]
        pr = stats.mannwhitneyu(da, db).pvalue if len(da) and len(db) else np.nan
        rows.append({"tasks": nm, "effect case 1 (pp)": 100 * da.mean() if len(da) else np.nan, "effect case 2 (pp)": 100 * db.mean() if len(db) else np.nan,
                     "case 2 - case 1 (pp)": 100 * (db.mean() - da.mean()) if len(da) and len(db) else np.nan,
                     "95% CI (pp)": f"[{100 * np.percentile(diff, 2.5):.1f}, {100 * np.percentile(diff, 97.5):.1f}]", "Mann-Whitney p": pv(pr)})
    s3.table(pd.DataFrame(rows).round(2), "h2h_effect_difference", "Difference between the allocator effects of the two cases")
    s3.test("Bootstrap difference of two independent effects (and Mann-Whitney on the per-task differences)",
            "The two cases have their own paired effect per task; comparing the cases is a comparison of two effect sizes, for which a bootstrap interval of the difference makes no distributional assumption.",
            "An interval containing zero means the data do not show that the allocator works differently with a stronger judge. The case comparison is observational: the cases differ in model family, size and backend, not only in the judge/agent ratio.",
            "see the table")
