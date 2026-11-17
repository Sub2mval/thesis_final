"""Chapters 5-7: validation questions (does the allocator catch errors?) and the exploratory analyses."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

from analysis_core import (GREY, INK, pe, tokens_axis, Ctx, Report, auc_cluster_boot, auc_score, boot_mean_ci, logit_cluster, mcnemar, pv, sign_test, wilcoxon,
                           wilson)
from analysis_eda import HATCH, panels
from analysis_rq import forest, subset

BINS = [1, 3, 4, 4.5, 4.75, 5, 5.01]
BIN_LABELS = ["<3", "3-4", "4-4.5", "4.5-4.75", "4.75-<5", "5"]


def z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() or 1)


# ------------------------------------------------------------------------------------------ VQ1
def vq1(c: Ctx, rep: Report) -> None:
    ch = rep.section("Validation question 1: internal (reasoning) errors", 1, "Is the trust score a good predictor of step inaccuracy?")
    ch.p("Population: every step of the clean runs with the allocator (no injected error), so every inaccurate step is a genuine mistake of the agents. A step is 'accurate' when its FINAL ANSWER "
         "is correct under the GAIA scorer, the same rule as for the final answer. Steps without any FINAL ANSWER (cut off or empty) are a set-up failure, not a reasoning error, and are excluded. "
         "The allocator never sees the ground truth, so this tests whether it can tell right from wrong from the reasoning and tool evidence alone.")
    s = rep.section("Discrimination", 2, "Does a higher trust score go with an accurate step?")
    rows, curves = [], {}
    for sid in c.ids:
        d = c.clean_on(sid)
        ok, bad = d[d.step_correct].trust_score, d[~d.step_correct].trust_score
        if len(ok) < 2 or len(bad) < 2:
            continue
        a, lo, hi = auc_cluster_boot(d, "trust_score", "step_correct")
        mw = stats.mannwhitneyu(ok, bad, alternative="greater").pvalue
        lg = logit_cluster(z(d.trust_score)[:, None], d.step_correct.astype(float).to_numpy(), d.task_id.to_numpy(), ["trust (per SD)"]).iloc[1]
        rho = stats.spearmanr(d.trust_score, d.step_correct.astype(int))
        rows.append({"case": c.short[sid], "steps": len(d), "inaccurate": len(bad), "mean trust accurate": ok.mean(), "mean trust inaccurate": bad.mean(),
                     "AUC": a, "AUC 95% CI": f"[{lo:.2f}, {hi:.2f}]", "Mann-Whitney p": pv(mw), "odds ratio per SD [cluster-robust p]": f"{lg.odds_ratio:.2f} [{pv(lg.p)}]",
                     "Spearman rho": rho.statistic})
        rep.findings[(sid, "vq1")] = {"effect": a, "lo": lo, "hi": hi, "p": mw, "p_logit": lg.p, "n": len(d)}
        thr = np.unique(d.trust_score)
        curves[sid] = [(np.mean(bad.to_numpy() < t), np.mean(ok.to_numpy() < t)) for t in np.r_[thr, thr.max() + 1]]
    s.table(pd.DataFrame(rows).round(3), "vq1_discrimination", "How well the trust score separates accurate from inaccurate steps (clean allocator-on runs)")
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for sid, pts in curves.items():
        pts = np.array([(0, 0)] + sorted(pts) + [(1, 1)])
        ax.plot(pts[:, 1], pts[:, 0], "-", color=c.color[sid], label=f"{c.short[sid]} (AUC {rep.findings[(sid, 'vq1')]['effect']:.2f})", lw=1.8)
    ax.plot([0, 1], [0, 1], ":", color=GREY, label="chance")
    ax.set(xlabel="false alarms: accurate steps with low trust", ylabel="caught: inaccurate steps with low trust", xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.legend(fontsize=8, loc="lower right")
    s.fig(fig, "vq1_roc", "ROC curve of 'low trust' as a detector of inaccurate steps. The diagonal is a score that carries no information.", .5)
    fig, axs = panels(c, 4.6, 3.4, n=c.n)
    for ax, sid in zip(axs, c.ids):
        d = c.clean_on(sid)
        bins = np.arange(1, 5.26, .25)
        ax.hist(d[d.step_correct].trust_score, bins, density=True, color=c.color[sid], alpha=.55, label="accurate steps")
        ax.hist(d[~d.step_correct].trust_score, bins, density=True, histtype="step", color=INK, lw=1.4, hatch="///", label="inaccurate steps")
        ax.set(xlabel="trust score", title=c.labels[sid])
        ax.legend(fontsize=7, loc="upper left")
    axs[0].set_ylabel("density")
    s.fig(fig, "vq1_distribution", "Trust score of accurate vs inaccurate steps. Two distributions on top of each other = no discrimination.")
    s.test("Area under the ROC curve (AUC) with a task-level bootstrap interval; Mann-Whitney U test; cluster-robust logistic regression",
           "AUC asks the right question for a score used to rank steps (the probability that an accurate step gets a higher trust score than an inaccurate one; 0.5 = chance) and needs no threshold. "
           "Steps of one task are not independent, so the interval comes from resampling whole tasks and the logistic regression uses task-clustered standard errors; the Mann-Whitney p-value ignores the clustering and is given for continuity.",
           "AUC of 0.5 means no information, 0.7 is usually called acceptable and 0.8 or more good. A significant p with an AUC close to 0.5 means a real but practically useless signal. "
           "The odds ratio is the change in the odds of an accurate step per standard deviation of trust.",
           " | ".join(f"{c.short[i]}: AUC {rep.findings[(i, 'vq1')]['effect']:.2f} [{rep.findings[(i, 'vq1')]['lo']:.2f}, {rep.findings[(i, 'vq1')]['hi']:.2f}], "
                      f"MW {pe(rep.findings[(i, 'vq1')]['p'])}, clustered {pe(rep.findings[(i, 'vq1')]['p_logit'])}" for i in c.ids if (i, 'vq1') in rep.findings))
    # calibration + thresholds
    s = rep.section("Calibration and usable thresholds", 2, "How accurate are steps at each trust level, and what would a warning threshold catch?")
    fig, axs = panels(c, 4.8, 3.5, n=c.n)
    rows, thr_rows = [], []
    for ax, sid in zip(axs, c.ids):
        d = c.clean_on(sid)
        d = d.assign(bin=pd.cut(d.trust_score, BINS, labels=BIN_LABELS, right=False))
        for k, lab in enumerate(BIN_LABELS):
            g = d[d.bin == lab]
            p, lo, hi = wilson(int(g.step_correct.sum()), len(g))
            if len(g):
                ax.bar(k, p, .7, color=c.color[sid], ec=INK, lw=.4)
                ax.errorbar(k, p, yerr=[[p - lo], [hi - p]], color=INK, lw=.8, capsize=2)
                ax.text(k, 0.02, f"n={len(g)}", ha="center", fontsize=7, color=INK, rotation=90, va="bottom")
            rows.append({"case": c.short[sid], "trust bin": lab, "steps": len(g), "share accurate": p})
        ax.axhline(d.step_correct.mean(), ls=":", color=GREY)
        ax.set_xticks(range(len(BIN_LABELS)), BIN_LABELS)
        ax.set(xlabel="trust score bin", title=c.labels[sid], ylim=(0, 1.02))
        for t in (3, 4, 4.5, 4.9, 5.0):
            fl, bad = d[d.trust_score < t], d[~d.step_correct]
            thr_rows.append({"case": c.short[sid], "warn if trust <": t, "steps flagged": len(fl), "share of inaccurate steps caught": (bad.trust_score < t).mean(),
                             "share of accurate steps flagged": (d[d.step_correct].trust_score < t).mean(), "precision (flagged that are inaccurate)": (~fl.step_correct).mean() if len(fl) else np.nan})
    axs[0].set_ylabel("share of accurate steps (dotted: overall)")
    s.fig(fig, "vq1_calibration", "Share of accurate steps per trust bin. A trustworthy score rises steeply from left to right; flat bars mean trust says little about accuracy.")
    s.table(pd.DataFrame(thr_rows).round(2), "vq1_thresholds", "What a warning threshold would catch (clean allocator-on steps)")
    s.p("The thresholds table shows the trade-off an operator faces: a stricter threshold catches more inaccurate steps but also flags more accurate ones.")
    # per maxim and per subgroup
    s = rep.section("Which part of the audit carries the signal?", 2, "Which Gricean maxim, agent, round, level and evidence condition discriminates best?")
    rows = []
    for sid in c.ids:
        d = c.clean_on(sid)
        for m in ("quality", "quantity", "relation", "manner"):
            g = d.dropna(subset=[f"maxim_{m}"])
            if g.step_correct.nunique() == 2:
                a, lo, hi = auc_cluster_boot(g, f"maxim_{m}", "step_correct", B=300)
                rows.append({"case": c.short[sid], "split": "maxim", "group": m, "steps": len(g), "AUC": a, "95% CI": f"[{lo:.2f}, {hi:.2f}]"})
        for col, nm in (("level", "GAIA level"), ("agent", "agent"), ("round", "round"), ("saw_tool_outputs", "audited agent used tools")):
            for v, g in d.groupby(col, dropna=True):
                if g.step_correct.nunique() == 2 and g.step_correct.value_counts().min() >= 8:
                    a, lo, hi = auc_cluster_boot(g, "trust_score", "step_correct", B=300)
                    rows.append({"case": c.short[sid], "split": nm, "group": v, "steps": len(g), "AUC": a, "95% CI": f"[{lo:.2f}, {hi:.2f}]"})
    s.table(pd.DataFrame(rows).round(2), "vq1_subgroups", "AUC by maxim and by subgroup (groups need at least 8 steps of each kind)")
    s.test("AUC with task-level bootstrap per maxim / subgroup", "Same discrimination measure as above, applied to each of the four audit criteria on their own and to subsets of steps.",
           "Maxims whose AUC is clearly above the overall score point to the part of the audit worth weighting more; overlapping intervals between groups mean no evidence of a real difference.", "see the table")


# ------------------------------------------------------------------------------------------ VQ2
def vq2(c: Ctx, rep: Report) -> None:
    ch = rep.section("Validation question 2: external (AiTM) errors", 1, "Is the trust score significantly lower for the injected message than for the original Agent 1 message?")
    ch.p("For each task, the injected message (Agent 1's first answer after corruption, scored in the injected run with the allocator) is compared with the original message "
         "(the same step in the clean run with the allocator). A task enters only if the clean run's first message is an exact text match of the original stored with the injection "
         "and both runs have a valid trust score; the table below counts every task that fails these requirements.")
    s = rep.section("Pair accounting", 2, "How many matched pairs are there?")
    rows = []
    for sid in c.ids:
        t = c.tasks[c.tasks.setup_id == sid]
        row = {"case": c.short[sid], "tasks": len(t), "matched pairs used": int((t.vq2_reason == "").sum())}
        for r, n in t[t.vq2_reason != ""].vq2_reason.value_counts().items():
            row[f"excluded: {r}"] = int(n)
        rows.append(row)
    s.table(pd.DataFrame(rows).fillna(0), "vq2_accounting", "Matched original/injected pairs and the reasons tasks were left out")
    s = rep.section("Is the injected message rated lower?", 2, "Does the allocator notice the tampering?")
    rows, fm_rows = [], []
    V = {sid: c.vq2_pairs(sid) for sid in c.ids}
    for sid in c.ids:
        v = V[sid]
        if len(v) < 3:
            continue
        d = (v.injected - v.original).to_numpy()
        m, lo, hi = boot_mean_ci(d)
        pw = wilcoxon(v.injected, v.original, "less")
        dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) else np.nan
        low = int((d < 0).sum())
        pl, pl_lo, pl_hi = wilson(low, len(d))
        rows.append({"case": c.short[sid], "pairs": len(v), "mean trust original": v.original.mean(), "mean trust injected": v.injected.mean(), "mean drop": m,
                     "95% CI": f"[{lo:.2f}, {hi:.2f}]", "injected lower in": f"{low}/{len(d)}", "Wilcoxon p (one-sided)": pv(pw), "Cohen dz": dz, "sign-test p": pv(sign_test(d))})
        rep.findings[(sid, "vq2")] = {"effect": m, "lo": lo, "hi": hi, "p": pw, "dz": dz, "n": len(v), "low": low}
        for fm, g in v.groupby("fm_id"):
            gd = g.injected - g.original
            fm_rows.append({"case": c.short[sid], "failure mode": fm, "pairs": len(g), "mean original": g.original.mean(), "mean injected": g.injected.mean(), "mean drop": gd.mean(),
                            "lower in": f"{int((gd < 0).sum())}/{len(g)}", "Wilcoxon p": pv(wilcoxon(g.injected, g.original, "less")) if len(g) >= 6 else "n<6"})
    s.table(pd.DataFrame(rows).round(2), "vq2_main", "Trust of the injected vs the original message (matched pairs)")
    fig, axs = panels(c, 4.0, 3.8, n=c.n)
    for ax, sid in zip(axs, c.ids):
        v = V[sid]
        for o, i in zip(v.original, v.injected):
            ax.plot([0, 1], [o, i], color=GREY, alpha=.25, lw=.8)
        ax.plot([0, 1], [v.original.mean(), v.injected.mean()], "o-", color=c.color[sid], lw=3, ms=7)
        ax.set_xticks([0, 1], ["original", "injected"])
        ax.set(title=f"{c.labels[sid]}\n(n={len(v)} pairs)", ylim=(.9, 5.1), xlim=(-.3, 1.3))
    axs[0].set_ylabel("trust score")
    s.fig(fig, "vq2_pairs", "One grey line per task from the original to the injected message; the coloured line is the mean.")
    s.test("Wilcoxon signed-rank test (one-sided: injected lower than original) on the matched pairs; exact sign test as a robustness check; Cohen's dz as effect size",
           "The two scores come from the same task, so the test is paired; trust scores are bounded and bunched at the top, so ranks are safer than a t-test. "
           "The one-sided alternative matches the hypothesis ('lower'); the sign test uses only the direction of each difference.",
           "A small p-value says the injected message is rated lower than chance explains. dz is the mean drop in units of the spread of the drops (0.8 or more is large). "
           "A significant drop shows that the allocator reacts to the tampering; it does not say that a warning is always raised (see thresholds below).",
           " | ".join(f"{c.short[i]}: drop {rep.findings[(i, 'vq2')]['effect']:+.2f}, {pe(rep.findings[(i, 'vq2')]['p'])}, dz={rep.findings[(i, 'vq2')]['dz']:.2f}" for i in c.ids if (i, 'vq2') in rep.findings))
    s = rep.section("By failure mode", 2, "Which kinds of injected error does the allocator catch?")
    s.table(pd.DataFrame(fm_rows).round(2), "vq2_by_fm", "Trust drop by failure mode (groups with fewer than 6 pairs get no test)")
    fig, axs = panels(c, 4.8, 3.6, n=c.n)
    for ax, sid in zip(axs, c.ids):
        v = V[sid]
        fms = [f for f, g in v.groupby("fm_id") if len(g) >= 2]
        for k, fm in enumerate(fms):
            g = v[v.fm_id == fm]
            b1 = ax.boxplot(g.original, positions=[k - .18], widths=.3, patch_artist=True, boxprops=dict(fc="white", ec=INK), medianprops=dict(color=INK), flierprops=dict(ms=2))
            b2 = ax.boxplot(g.injected, positions=[k + .18], widths=.3, patch_artist=True, boxprops=dict(fc=c.color[sid], alpha=.7), medianprops=dict(color=INK), flierprops=dict(ms=2))
        ax.set_xticks(range(len(fms)), [f"{f}\n(n={int((v.fm_id == f).sum())})" for f in fms])
        ax.set(title=c.labels[sid], ylim=(.9, 5.1))
        ax.legend(handles=[Patch(fc="white", ec=INK, label="original"), Patch(fc=c.color[sid], alpha=.7, label="injected")], fontsize=7, loc="lower left")
    axs[0].set_ylabel("trust score")
    s.fig(fig, "vq2_by_fm", "Trust scores of original and injected messages by failure mode (modes with at least 2 pairs).")
    s.test("Wilcoxon signed-rank test per failure mode (descriptive)", "The same paired test as above within each mode; modes the injector picked rarely have too few pairs for any test.",
           "Differences between modes are descriptive: the injector chose the mode per task, so modes also differ in the kinds of tasks they were applied to.", "see the table")
    # detection thresholds + tampering vs wrongness + size of the edit
    s = rep.section("Detection thresholds, tampering versus wrongness, size of the edit", 2, "How often would a warning threshold fire, and what does the score react to?")
    rows = []
    fig, axs = panels(c, 4.6, 3.5, n=c.n, sharey=False)
    for ax, sid in zip(axs, c.ids):
        v = V[sid]
        if len(v) < 3:
            continue
        for t in (3, 4, 4.5, 4.9):
            rows.append({"case": c.short[sid], "warn if trust <": t, "injected messages flagged": (v.injected < t).mean(), "original messages flagged (false alarms)": (v.original < t).mean()})
        paired_auc = float(((v.original > v.injected).sum() + .5 * (v.original == v.injected).sum()) / len(v))
        rep.findings[(sid, "vq2_auc")] = paired_auc
        ax.scatter(v.words_changed, v.original - v.injected, s=14, color=c.color[sid], alpha=.6)
        rho = stats.spearmanr(v.words_changed, v.original - v.injected) if len(v) > 3 else None
        ax.set(xlabel="share of words changed by the corruption", title=f"{c.labels[sid]}\nrho={rho.statistic:.2f}, {pe(rho.pvalue)}" if rho else c.labels[sid])
    axs[0].set_ylabel("trust drop (original - injected)")
    s.table(pd.DataFrame(rows).round(2), "vq2_thresholds", "Share of messages that would trigger a warning at a given threshold")
    s.fig(fig, "vq2_edit_size", "Does a bigger rewrite produce a bigger trust drop?")
    rows = []
    for sid in c.ids:
        v = V[sid]
        for lab, g in (("answer wrong", v[~v.inj_step_correct.astype(bool)]), ("answer still correct", v[v.inj_step_correct.astype(bool)])):
            rows.append({"case": c.short[sid], "injected message": lab, "pairs": len(g), "mean trust injected": g.injected.mean(), "mean drop": (g.original - g.injected).mean()})
    s.table(pd.DataFrame(rows).round(2), "vq2_tamper_vs_wrong", "Does the allocator react to the answer being wrong or to the message being altered?",
            "If the trust drop is as large when the corrupted message still reaches the correct answer, the allocator reacts to inconsistency in the text, not to the correctness of the answer.")
    s.test("Spearman correlation (edit size vs trust drop); descriptive threshold and subgroup tables", "Edit size is a continuous share and the drop a bounded score; a rank correlation needs no linearity.",
           "A positive correlation means larger rewrites are penalised more. Threshold tables are descriptive operating points, not tests.", "see figures and tables")


# ------------------------------------------------------------------------------------------ exploratory A
def mistakes(c: Ctx, rep: Report) -> None:
    ch = rep.section("Exploratory analysis 1: allocator mistakes", 1, "Is a trust allocator that makes a mistake worse than having no allocator at all?")
    ch.p(f"A mistake is either false reassurance (trust of at least {c.high} on a step whose answer is inaccurate) or a false alarm (trust below {c.low} on an accurate step). "
         "The agents see the score and the allocator's reasons before their next turn, so a wrong score can mislead them. For every task the allocator-on run is compared with the paired "
         "allocator-off run: if mistakes made the allocator harmful, tasks with mistakes should do worse with the allocator than without it.")
    s = rep.section("How often does the allocator make a mistake?", 2, "")
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(10, 0.55 * c.n * 2 + 1.6), sharex=True)
    for sid in c.ids:
        d = c.clean_on(sid)
        cl = pd.cut(d.trust_score, [0, c.low, c.high, 5.1], right=False, labels=["low", "medium", "high"])
        for acc, ax_i in ((True, 0), (False, 1)):
            g = cl[d.step_correct == acc].value_counts(normalize=True).reindex(["high", "medium", "low"]).fillna(0)
            left = 0
            for lab, a in zip(g.index, (1, .55, .25)):
                axs[ax_i].barh(c.short[sid], g[lab], left=left, color=c.color[sid], alpha=a, ec=INK, lw=.4, hatch={"high": "", "medium": "", "low": "///"}[lab])
                axs[ax_i].text(left + g[lab] / 2, c.short[sid], f"{lab} {g[lab]:.0%}" if g[lab] > .08 else "", ha="center", va="center", fontsize=7)
                left += g[lab]
        for acc in (True, False):
            g = d[d.step_correct == acc]
            rows.append({"case": c.short[sid], "steps": "accurate" if acc else "inaccurate", "n": len(g), f"high trust (>={c.high})": int((g.trust_score >= c.high).sum()),
                         "medium": int(((g.trust_score < c.high) & (g.trust_score >= c.low)).sum()), f"low (<{c.low})": int((g.trust_score < c.low).sum())})
    axs[0].set_title("accurate steps: how they were scored")
    axs[1].set_title("inaccurate steps: how they were scored")
    s.fig(fig, "mis_confusion", "Trust level given to accurate (left) and inaccurate (right) steps. Solid = high trust, lighter = medium, hatched = low.")
    s.table(pd.DataFrame(rows), "mis_counts", "Trust level by step accuracy (clean allocator-on runs)")
    # ---- A. mistakes at the FIRST message: identical in the off and on arm, so the grouping cannot depend on the allocator's later influence
    s = rep.section("Mistakes at the first message: worse than no allocator?", 2, "When the allocator misjudges Agent 1's first message, is the final outcome worse than in the paired run without allocator?")
    s.p("Message 1 is the same text in the allocator-off and allocator-on run of a task (clean runs: the shared first answer; injected runs: the shared corrupted answer), and it is scored before the "
        "two runs can diverge. Grouping tasks by whether the allocator misjudged this message therefore compares like with like: any difference between the on and off arm in the 'mistake' group "
        "can be a cost of the mistake, and is not produced by the grouping itself.")
    rows, fr = [], []
    for sid in c.ids:
        for kind in ("baseline", "fork"):
            runs = c.r(sid, kind)
            runs = runs[runs.eligible] if kind == "fork" else runs
            pv_ = runs.pivot(index="task_id", columns="trust", values=["final_correct", "propagation"] if kind == "fork" else ["final_correct"]).dropna(subset=[("final_correct", "off"), ("final_correct", "on")])
            pv_.columns = [f"{a}_{b}" for a, b in pv_.columns]
            pv_ = pv_.apply(pd.to_numeric, errors="coerce").astype(float)
            st0 = c.steps[(c.steps.setup_id == sid) & (c.steps.kind == kind) & (c.steps.trust == "on") & c.steps.trust_score.notna() & c.steps.has_answer]
            st0 = st0[st0.injected] if kind == "fork" else st0[st0.step == 0]
            st0 = st0.set_index("task_id")
            keep = pv_.index.intersection(st0.index)
            if kind == "fork":
                keep = keep.intersection(c.tasks[(c.tasks.setup_id == sid) & (c.tasks.rq_reason == "")].task_id)
            pv_, st0 = pv_.loc[keep], st0.loc[keep]
            fr_ = (st0.trust_score >= c.high) & ~st0.step_correct
            fa_ = (st0.trust_score < c.low) & st0.step_correct
            grp = pd.Series(np.where(fr_, "false reassurance", np.where(fa_, "false alarm", "no mistake")), index=pv_.index)
            for name in ("no mistake", "false reassurance", "false alarm"):
                q = pv_[grp == name]
                if len(q) < 2:
                    continue
                off, on = q.final_correct_off.astype(bool), q.final_correct_on.astype(bool)
                m = mcnemar(off, on)
                d, lo, hi = boot_mean_ci(on.astype(float).to_numpy() - off.astype(float).to_numpy())
                rows.append({"case": c.short[sid], "runs": "clean" if kind == "baseline" else "injected", "allocator at message 1": name, "n": len(q), "acc. off": off.mean(), "acc. on": on.mean(),
                             "diff (pp)": 100 * d, "95% CI (pp)": f"[{100 * lo:.1f}, {100 * hi:.1f}]", "only off right / only on right": f"{m['only_ref']}/{m['only_comp']}", "McNemar p": pv(m["p"])})
                fr.append((f"{c.short[sid]} {'clean' if kind == 'baseline' else 'injected'}: {name} (n={len(q)})", 100 * d, 100 * lo, 100 * hi, c.color[sid]))
            mis = grp != "no mistake"
            if mis.sum() >= 3 and (~mis).sum() >= 3:
                dd = (pv_.final_correct_on - pv_.final_correct_off)
                rep.findings[(sid, f"mis_{kind}")] = {"p": stats.mannwhitneyu(dd[mis], dd[~mis]).pvalue, "d_mis": dd[mis].mean(), "d_ok": dd[~mis].mean(), "n_mis": int(mis.sum()),
                                                      "p_within": mcnemar(pv_.final_correct_off[mis].astype(bool), pv_.final_correct_on[mis].astype(bool))["p"]}
    s.table(pd.DataFrame(rows).round(2), "mis_first_message", "Allocator on minus off, by whether the allocator misjudged the first message",
            f"False reassurance = trust >= {c.high} on a wrong message; false alarm = trust < {c.low} on a correct one; anything else (including medium scores) = no mistake. Tasks with failed or unusable audits are left out.")
    if fr:
        fig, ax = plt.subplots(figsize=(7.8, 0.4 * len(fr) + 1.2))
        forest(ax, fr, "accuracy difference, allocator on minus off (pp, 95% bootstrap CI)")
        s.fig(fig, "mis_forest", "Allocator effect on final accuracy by the allocator's verdict on the first message. If a mistaken verdict made the allocator worse than none, the mistake rows would sit left of zero.")
    s.test("Exact McNemar test within each group (on vs off, paired by task); Mann-Whitney test of the per-task change between mistake and no-mistake tasks",
           "Within a group the two arms are paired runs of the same task, so McNemar is the exact paired comparison. The groups are defined by the allocator's verdict on a message that both arms share, so the "
           "group contrast is not distorted by what happened later in the debate. Mistake groups are small, which limits power.",
           "A significantly negative difference in a mistake group would mean a mistaken allocator is worse than none. A difference near zero means mistakes were not costly here; with small groups read the "
           "interval: it shows how large a harm the data can rule out.",
           " | ".join(f"{c.short[i]} {k}: mistakes {rep.findings[(i, f'mis_{k}')]['d_mis'] * 100:+.1f} pp (n={rep.findings[(i, f'mis_{k}')]['n_mis']}, {pe(rep.findings[(i, f'mis_{k}')]['p_within'])}) vs no mistake {rep.findings[(i, f'mis_{k}')]['d_ok'] * 100:+.1f} pp, "
                      f"difference {pe(rep.findings[(i, f'mis_{k}')]['p'])}" for i in c.ids for k in ("baseline", "fork") if (i, f"mis_{k}") in rep.findings))
    # ---- B. mistakes anywhere in the debate: descriptive only, because the grouping depends on the allocator-on run itself
    s = rep.section("Mistakes anywhere in the debate (descriptive)", 2, "Do tasks in which the allocator misjudged some step end worse than their allocator-off twin?")
    rows = []
    for sid in c.ids:
        for kind in ("baseline", "fork"):
            runs = c.r(sid, kind)
            runs = runs[runs.eligible] if kind == "fork" else runs
            pv_ = runs.pivot(index="task_id", columns="trust", values=["final_correct"]).dropna()
            pv_.columns = [f"{a}_{b}" for a, b in pv_.columns]
            st = c.steps[(c.steps.setup_id == sid) & (c.steps.kind == kind) & (c.steps.trust == "on") & c.steps.trust_score.notna() & c.steps.has_answer]
            st = st.assign(mis=((st.trust_score >= c.high) & ~st.step_correct) | ((st.trust_score < c.low) & st.step_correct))
            anym = st.groupby("task_id").mis.any().reindex(pv_.index).fillna(False).astype(bool)
            for name, mask in (("no mistake in the run", ~anym), ("at least one mistake", anym)):
                q = pv_[mask]
                if len(q) < 2:
                    continue
                off, on = q.final_correct_off.astype(bool), q.final_correct_on.astype(bool)
                m = mcnemar(off, on)
                rows.append({"case": c.short[sid], "runs": "clean" if kind == "baseline" else "injected", "tasks": name, "n": len(q), "acc. off": off.mean(), "acc. on": on.mean(),
                             "diff (pp)": 100 * m["diff"], "only off right / only on right": f"{m['only_ref']}/{m['only_comp']}", "McNemar p": pv(m["p"])})
    s.table(pd.DataFrame(rows).round(2), "mis_any", "Allocator on minus off, by whether the allocator-on run contained any mistake")
    s.p("Caution: here the groups are defined from the allocator-on run itself. A run with a false reassurance contains an inaccurate step, which makes its final answer more likely wrong, whereas its "
        "allocator-off twin is not affected by this selection. A negative difference in the 'at least one mistake' rows is therefore expected even if the allocator's mistakes cost nothing; "
        "only the first-message analysis above is a fair test of harm.")


# ------------------------------------------------------------------------------------------ exploratory B
def ctxlen(c: Ctx, rep: Report) -> None:
    ch = rep.section("Exploratory analysis 2: judge performance and prompt length", 1,
                     "Does the trust allocator perform worse as its input approaches the judge model's context limit?")
    ch.p("The judge's input is the task, the whole debate so far and the audited agent's tool outputs, so it grows with the position in the debate and with tool-heavy tasks. "
         "Prompt length (tokens of the judge call, as recorded) is compared with how well the trust score tracks step accuracy. Because later steps both have longer prompts and are "
         "different in kind, the model below controls for the step position. The context limit is the recorded num_ctx, or the backend default (see the exploratory chapter).")
    s = rep.section("How close does the judge get to its limit?", 2, "")
    rows = []
    for sid in c.ids:
        a = c.steps[(c.steps.setup_id == sid) & c.steps.audit_prompt_tokens.notna()]
        rows.append({"case": c.short[sid], "audits": len(a), "median prompt (tokens)": a.audit_prompt_tokens.median(), "p95": a.audit_prompt_tokens.quantile(.95),
                     "max": a.audit_prompt_tokens.max(), "context limit": a.ctx_limit.median(), "max share of limit": a.audit_fill.max(),
                     "audits above 25% of limit": (a.audit_fill > .25).mean(), "above 50%": (a.audit_fill > .5).mean(), "above 75%": (a.audit_fill > .75).mean()})
    tab = pd.DataFrame(rows)
    s.table(tab.round(3), "ctx_limit_usage", "Judge prompt length relative to the context limit")
    far = all(r["max share of limit"] < .5 for r in rows)
    s.p("In this data no judge call used more than half of its context window, so the question can only be answered inside the observed range, not for prompts at the limit itself."
        if far else "Some judge calls used more than half of the context window; the upper bins below therefore speak to behaviour close to the limit.")
    s = rep.section("Discrimination by prompt length", 2, "")
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.7))
    rows, lg_rows = [], []
    for sid in c.ids:
        d = c.clean_on(sid)
        d = d[d.audit_prompt_tokens.notna()].copy()
        if len(d) < 50:
            continue
        d["bin"] = pd.qcut(d.audit_prompt_tokens, 5, duplicates="drop")
        xs, aucs, los, his, tr = [], [], [], [], []
        for b, g in d.groupby("bin", observed=True):
            a, lo, hi = auc_cluster_boot(g, "trust_score", "step_correct", B=300) if g.step_correct.nunique() == 2 and g.step_correct.value_counts().min() >= 5 else (np.nan,) * 3
            xs.append(g.audit_prompt_tokens.median())
            aucs.append(a); los.append(lo); his.append(hi); tr.append(g.trust_score.mean())
            rows.append({"case": c.short[sid], "prompt tokens (bin)": str(b), "median fill of limit": g.audit_fill.median(), "steps": len(g), "step accuracy": g.step_correct.mean(),
                         "mean trust": g.trust_score.mean(), "AUC": a, "AUC 95% CI": f"[{lo:.2f}, {hi:.2f}]",
                         "high trust given inaccurate": (g[~g.step_correct].trust_score >= c.high).mean()})
        aucs, los, his = map(np.array, (aucs, los, his))
        axs[0].errorbar(xs, aucs, yerr=[aucs - los, his - aucs], fmt="o-", color=c.color[sid], capsize=3, label=c.short[sid])
        axs[1].plot(xs, tr, "o-", color=c.color[sid], label=c.short[sid])
        X = np.column_stack([z(d.trust_score), z(np.log(d.audit_prompt_tokens)), z(d.trust_score) * z(np.log(d.audit_prompt_tokens)), z(d.step)])
        lg = logit_cluster(X, d.step_correct.astype(float).to_numpy(), d.task_id.to_numpy(), ["trust", "log prompt length", "trust x log prompt length", "step position"])
        lg.insert(0, "case", c.short[sid])
        lg_rows.append(lg)
        it = lg[lg.term == "trust x log prompt length"].iloc[0]
        rep.findings[(sid, "ctx_interaction")] = {"coef": it.coef, "p": it.p, "or": it.odds_ratio}
        axs[2].plot(xs, [1 - d[d.bin == b].step_correct.mean() for b in sorted(d.bin.unique())], "o-", color=c.color[sid])
    axs[0].axhline(.5, color=GREY, ls=":")
    for a, t in zip(axs, ("AUC of trust by judge prompt length", "mean trust by prompt length", "share of inaccurate steps by prompt length")):
        a.set(title=t, xlabel="judge prompt tokens (median of quintile bin)")
        tokens_axis(a)
    axs[0].legend(fontsize=8)
    s.fig(fig, "ctx_auc_by_length", "Judge discrimination (AUC, bootstrap 95% CI) and mean trust across quintiles of prompt length. A downward AUC trend would mean the judge degrades with longer input.")
    s.table(pd.DataFrame(rows).round(2), "ctx_bins", "Judge performance by prompt-length quintile (clean allocator-on steps)")
    if lg_rows:
        s.table(pd.concat(lg_rows).round(3), "ctx_logit", "Does the predictive power of trust change with prompt length? (logistic regression of step accuracy, task-clustered SEs)",
                "A negative 'trust x log prompt length' coefficient means trust predicts step accuracy less well when the prompt is longer. The step position is controlled for.")
    s.test("Logistic regression of step accuracy on trust, log prompt length, their interaction and step position, with task-clustered standard errors; AUC per length bin",
           "The interaction term is exactly the quantity of interest (does trust predict accuracy less well when the prompt is longer?), and clustering by task respects that steps of one task are correlated. "
           "The binned AUCs show the same thing without a model; both are needed because the model can hide non-monotone patterns.",
           "A significantly negative interaction would support 'the judge degrades with long inputs'. A non-significant interaction is not proof that length is harmless, especially if prompts never come close to the limit. "
           "Prompt length also grows with the debate position and with tool-heavy tasks, so the effect is associational.",
           " | ".join(f"{c.short[i]}: interaction coef {rep.findings[(i, 'ctx_interaction')]['coef']:+.3f} (OR {rep.findings[(i, 'ctx_interaction')]['or']:.2f}), {pe(rep.findings[(i, 'ctx_interaction')]['p'])}"
                      for i in c.ids if (i, "ctx_interaction") in rep.findings))
    s = rep.section("Unusable audits and prompt length", 2, "Does the judge fail to produce a usable score more often on long prompts?")
    rows = []
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for sid in c.ids:
        a = c.steps[(c.steps.setup_id == sid) & (c.steps.trust == "on") & c.steps.has_trust_entry & c.steps.audit_prompt_tokens.notna()].copy()
        if len(a) < 50:
            continue
        a["bin"] = pd.qcut(a.audit_prompt_tokens, 5, duplicates="drop")
        a["fail"] = a.trust_score.isna()
        g = a.groupby("bin", observed=True).agg(tokens=("audit_prompt_tokens", "median"), n=("fail", "size"), fail=("fail", "mean"))
        ax.plot(g.tokens, g.fail, "o-", color=c.color[sid], label=c.short[sid])
        rho = stats.spearmanr(a.audit_prompt_tokens, a.fail)
        rows.append({"case": c.short[sid], "audits": len(a), "unusable": int(a.fail.sum()), "share unusable": a.fail.mean(), "Spearman rho (tokens, unusable)": rho.statistic, "p": pv(rho.pvalue)})
    ax.set(xlabel="judge prompt tokens (bin median)", ylabel="share of audits without a usable score")
    tokens_axis(ax)
    ax.legend(fontsize=8)
    s.fig(fig, "ctx_unusable", "Share of audits that returned no usable score, by prompt length.", .6)
    s.table(pd.DataFrame(rows).round(3), "ctx_unusable_table", "Unusable audits vs prompt length")
    s.test("Spearman rank correlation between prompt length and audit failure", "Failure is binary and rare, length continuous; a rank correlation is a simple, assumption-light screen.",
           "A positive correlation would mean long prompts cause more failures. Failures this rare give little power, so only a clear trend is informative.", "see the table")
