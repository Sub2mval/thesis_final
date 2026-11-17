"""Shared machinery for analyze.py: statistical helpers, the plot style, and the Report object that collects sections
(text, figures, tables, test write-ups) and renders them to LaTeX and Markdown."""

from __future__ import annotations

import math
import re
import subprocess
import shutil
import warnings
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ------------------------------------------------------------------------------------------ plot style
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # one colour per setup (validated colour-blind-safe set); case 1 = blue, case 2 = orange
INK, GREY, LIGHT = "#2b2b29", "#8a8a85", "#d9d8d3"
HATCH = {"off": "", "on": "////"}  # allocator off = solid, on = hatched (identity never rests on colour alone)


def style() -> None:
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GREY,
                         "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
                         "grid.color": LIGHT, "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
                         "axes.prop_cycle": matplotlib.cycler(color=COLORS)})


def tokens_axis(ax) -> None:
    """Log x-axis with plain '4k / 10k' tick labels instead of matplotlib's 4x10^3."""
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v / 10 ** int(np.floor(np.log10(v))) in (2, 5) else ""))
    ax.tick_params(axis="x", which="minor", labelsize=7)


# ------------------------------------------------------------------------------------------ statistics
def pv(p) -> str:
    return "n/a" if p is None or p != p else ("<0.001" if p < 0.001 else f"{p:.3f}")


def pe(p) -> str:
    """p-value inside a sentence: 'p<0.001' or 'p=0.035'."""
    return "p n/a" if p is None or p != p else ("p<0.001" if p < 0.001 else f"p={p:.3f}")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion and Wilson 95% interval (better than the normal interval for small n or p near 0/1)."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    c, h = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, c - h), min(1.0, c + h)


def mcnemar(ref, comp) -> dict:
    """Exact McNemar test on paired booleans (ref = reference condition, comp = comparison condition)."""
    ref, comp = np.asarray(ref, bool), np.asarray(comp, bool)
    b, c = int((ref & ~comp).sum()), int((~ref & comp).sum())
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {"n": len(ref), "only_ref": b, "only_comp": c, "p": p, "diff": float(comp.mean() - ref.mean()) if len(ref) else np.nan}


def sign_test(d) -> float:
    """Two-sided exact sign test on paired differences (zeros dropped)."""
    d = np.asarray(d, float)
    d = d[d != 0]
    return stats.binomtest(int((d > 0).sum()), len(d), 0.5).pvalue if len(d) else 1.0


def wilcoxon(x, y, alt: str = "two-sided") -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # scipy warns when ties make the normal approximation degenerate
            return float(stats.wilcoxon(np.asarray(x, float), np.asarray(y, float), alternative=alt).pvalue)
    except ValueError:  # every difference is zero
        return np.nan


def boot_mean_ci(d, B: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of paired differences with a percentile bootstrap 95% CI (resampling tasks)."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return (float(d.mean()) if len(d) else np.nan), np.nan, np.nan
    m = np.random.default_rng(seed).choice(d, (B, len(d))).mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mde_pp(n_discordant: int, n: int) -> float:
    """Approximate smallest difference in accuracy (percentage points) an exact McNemar test detects with 80% power
    (alpha = .05 two-sided), given the observed number of discordant pairs: |b - c| ~ 2.8 * sqrt(discordant)."""
    return 100 * 2.8 * math.sqrt(max(n_discordant, 1)) / n if n else np.nan


def auc_score(pos, neg) -> float:
    """P(score of a 'pos' item > score of a 'neg' item), ties count 1/2 (= ROC AUC)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return np.nan
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_cluster_boot(df: pd.DataFrame, score: str, label: str, cluster: str = "task_id", B: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """AUC of `score` for label==True vs label==False with a bootstrap over clusters (steps of one task are not independent)."""
    groups = [(g[score].to_numpy(float)[g[label].to_numpy(bool)], g[score].to_numpy(float)[~g[label].to_numpy(bool)]) for _, g in df.groupby(cluster)]
    a = auc_score(np.concatenate([g[0] for g in groups]), np.concatenate([g[1] for g in groups]))
    rng, out = np.random.default_rng(seed), []
    for _ in range(B):
        pick = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        v = auc_score(np.concatenate([g[0] for g in pick]), np.concatenate([g[1] for g in pick]))
        if v == v:
            out.append(v)
    return a, (float(np.percentile(out, 2.5)) if out else np.nan), (float(np.percentile(out, 97.5)) if out else np.nan)


def holm(ps: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate over the headline tests)."""
    idx = sorted((i for i, p in enumerate(ps) if p == p), key=lambda i: ps[i])
    out, run = [np.nan] * len(ps), 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (len(idx) - r) * ps[i]))
        out[i] = run
    return out


def logit_cluster(X: np.ndarray, y: np.ndarray, groups: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Logistic regression (Newton-Raphson) with cluster-robust standard errors (clusters = tasks). -> coef, SE, z, p, odds ratio."""
    X1 = np.column_stack([np.ones(len(y)), X])
    b = np.zeros(X1.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-np.clip(X1 @ b, -30, 30)))
        H = X1.T @ (X1 * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(b))
        step = np.linalg.solve(H, X1.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-9:
            break
    p = 1 / (1 + np.exp(-np.clip(X1 @ b, -30, 30)))
    H = X1.T @ (X1 * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(b))
    S = pd.DataFrame(X1 * (y - p)[:, None]).groupby(groups).sum().to_numpy()
    G = len(S)
    Hi = np.linalg.inv(H)
    V = Hi @ (S.T @ S) @ Hi * G / max(G - 1, 1)
    se = np.sqrt(np.diag(V))
    z = b / se
    return pd.DataFrame({"term": ["intercept"] + names, "coef": b, "se": se, "z": z, "p": 2 * stats.norm.sf(np.abs(z)), "odds_ratio": np.exp(b)})


# ------------------------------------------------------------------------------------------ report object
_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
_UNI = {"≥": r"$\geq$", "≤": r"$\leq$", "→": r"$\rightarrow$", "×": r"$\times$", "±": r"$\pm$", "≈": r"$\approx$", "−": "-", "Δ": r"$\Delta$",
        "α": r"$\alpha$", "ρ": r"$\rho$", "–": "--", "—": "---", "’": "'", "“": "``", "”": "''", "→": r"$\rightarrow$", "·": r"$\cdot$"}


def tex(s) -> str:
    """Escape text for LaTeX; $...$ stays as math, and a few unicode symbols are mapped."""
    parts = str(s).split("$")
    out = []
    for i, seg in enumerate(parts):
        if i % 2:  # inside $...$
            out.append("$" + seg + "$")
        else:
            seg = "".join(_TEX.get(c, c) for c in seg)
            out.append("".join(_UNI.get(c, c) for c in seg))
    return "".join(out) if len(parts) % 2 else tex(str(s).replace("$", r"\$"))


def cell(v) -> str:
    if isinstance(v, (float, np.floating)):
        return "" if v != v else f"{v:.2f}"
    return "" if v is None else str(v)


class Section:
    def __init__(self, rep: "Report", title: str, level: int, question: str = "") -> None:
        self.rep, self.title, self.level, self.question, self.blocks = rep, title, level, question, []

    def p(self, text: str) -> None:
        self.blocks.append(("p", text))

    def fig(self, fig, name: str, caption: str, width: float = 0.95) -> None:
        path = self.rep.fig_dir / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        self.blocks.append(("fig", path, caption, width))

    def table(self, df: pd.DataFrame, name: str, caption: str, note: str = "", transpose: bool = False) -> None:
        """transpose = rows become columns (for tables with many measures and few rows, which would be unreadably small otherwise)."""
        if df is None or len(df) == 0:
            self.blocks.append(("p", f"({caption}: no data)"))
            return
        if transpose:
            df = df.set_index(df.columns[0]).T.rename_axis("measure").reset_index()
        df.to_csv(self.rep.tab_dir / f"{name}.csv", index=not isinstance(df.index, pd.RangeIndex))
        self.blocks.append(("tab", df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df, caption, note))

    def test(self, name: str, why: str, read: str, result: str) -> None:
        """name = the test; why = why it fits this data (1-2 sentences); read = how to interpret it; result = the numbers."""
        self.blocks.append(("test", name, why, read, result))


class Report:
    def __init__(self, out: Path, title: str) -> None:
        self.out, self.title, self.sections = out, title, []
        self.fig_dir, self.tab_dir = out / "figures", out / "tables"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.tab_dir.mkdir(parents=True, exist_ok=True)
        self.findings: dict = {}  # filled by the sections, read by the executive summary

    def section(self, title: str, level: int = 1, question: str = "") -> Section:
        s = Section(self, title, level, question)
        self.sections.append(s)
        return s

    # ---- LaTeX
    def latex(self) -> str:
        o = [r"\documentclass[10pt,a4paper]{article}", r"\usepackage[T1]{fontenc}", r"\usepackage[utf8]{inputenc}", r"\IfFileExists{lmodern.sty}{\usepackage{lmodern}}{}",
             r"\usepackage[margin=2.1cm]{geometry}", r"\usepackage{graphicx,booktabs,float,caption,amsmath,array,xcolor}",
             r"\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black]{hyperref}", r"\captionsetup{font=small,labelfont=bf}",
             r"\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}", r"\title{" + tex(self.title) + "}", r"\date{\today}", r"\begin{document}",
             r"\maketitle", r"\tableofcontents", r"\clearpage"]
        for s in self.sections:
            o.append("\\" + ["section", "subsection", "subsubsection"][s.level - 1] + "{" + tex(s.title) + "}")
            if s.question:
                o.append(r"\textit{Question: " + tex(s.question) + "}\n")
            for b in s.blocks:
                if b[0] == "p":
                    o.append(tex(b[1]) + "\n")
                elif b[0] == "fig":
                    o += [r"\begin{figure}[H]\centering", rf"\includegraphics[width={b[3]}\linewidth]{{figures/{b[1].name}}}", r"\caption{" + tex(b[2]) + "}", r"\end{figure}"]
                elif b[0] == "tab":
                    full = b[1]
                    first, rest = full.columns[0], list(full.columns[1:])
                    parts = [rest[k:k + 6] for k in range(0, len(rest), 6)] or [[]]  # at most 7 columns per printed table, so text stays readable
                    for k, cs in enumerate(parts):
                        df = full[[first] + cs]
                        num = lambda col: all(re.fullmatch(r"[<>+\-~]?[\d.,]+( ?(pp|%|x|k))?|n/a|", cell(v).strip()) for v in col)
                        spec = "".join("r" if num(df[c]) else "l" for c in df.columns)
                        cap = b[2] + (f" ({k + 1}/{len(parts)})" if len(parts) > 1 else "")
                        o += [r"\begin{table}[H]\centering\small", r"\caption{" + tex(cap) + "}", r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{%",
                              rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(tex(c) for c in df.columns) + r" \\", r"\midrule"]
                        o += [" & ".join(tex(cell(v)) for v in r) + r" \\" for r in df.itertuples(index=False)]
                        o += [r"\bottomrule", r"\end{tabular}}"] + ([r"\\[2pt]{\footnotesize " + tex(b[3]) + "}"] if b[3] and k == len(parts) - 1 else []) + [r"\end{table}"]
                else:
                    _, name, why, read, res = b
                    o += [r"\begin{quote}\small", r"\textbf{Test: }" + tex(name) + r"\\", r"\textbf{Why this test: }" + tex(why) + r"\\",
                          r"\textbf{How to read it: }" + tex(read) + r"\\", r"\textbf{Result: }" + tex(res), r"\end{quote}"]
        o.append(r"\end{document}")
        return "\n".join(o)

    # ---- Markdown (same content, for quick reading / diffs)
    def markdown(self) -> str:
        o = [f"# {self.title}", ""]
        for s in self.sections:
            o += ["#" * (s.level + 1) + " " + s.title, ""] + ([f"*Question: {s.question}*", ""] if s.question else [])
            for b in s.blocks:
                if b[0] == "p":
                    o += [b[1].replace("$", ""), ""]
                elif b[0] == "fig":
                    o += [f"![{b[2]}](figures/{b[1].name})", "", f"*{b[2]}*", ""]
                elif b[0] == "tab":
                    df = b[1]
                    o += [f"**{b[2]}**", "", "| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
                    o += ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)] + ([f"*{b[3]}*"] if b[3] else []) + [""]
                else:
                    o += [f"> **Test:** {b[1]}  \n> **Why:** {b[2]}  \n> **How to read it:** {b[3]}  \n> **Result:** {b[4]}".replace("$", ""), ""]
        return "\n".join(o)

    def write(self, compile_pdf: bool = True) -> list[str]:
        (self.out / "report.tex").write_text(self.latex(), encoding="utf-8")
        (self.out / "report.md").write_text(self.markdown(), encoding="utf-8")
        msgs = []
        if compile_pdf:
            if shutil.which("pdflatex") is None:
                msgs.append("pdflatex not found: report.tex is complete. Install LaTeX (sudo apt install texlive-latex-recommended texlive-latex-extra) or upload report.tex plus figures/ to Overleaf")
            else:
                for _ in range(2):  # twice: table of contents
                    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=self.out, capture_output=True, text=True)
                msgs.append("report.pdf written" if r.returncode == 0 else "pdflatex failed (see report.log); report.tex is complete")
        return msgs


# ------------------------------------------------------------------------------------------ analysis context
class Ctx:
    """The three tables plus the per-setup views every section needs. Failed runs stay out of every analysis (counted in data quality)."""

    def __init__(self, runs: pd.DataFrame, steps: pd.DataFrame, tasks: pd.DataFrame, labels: dict, high: float, low: float) -> None:
        self.all_runs, self.tasks, self.labels, self.high, self.low = runs, tasks, labels, high, low
        self.runs = runs[runs.ok].copy()
        key = ["setup_id", "task_id", "kind", "trust"]
        good = self.runs[key]
        self.steps = steps.merge(good, on=key, how="inner")
        self.ids = list(labels)
        self.color = {s: COLORS[i % len(COLORS)] for i, s in enumerate(self.ids)}
        self.short = {s: f"Case {i + 1}" for i, s in enumerate(self.ids)}
        self.n = len(self.ids)

    def r(self, sid: str, kind: str | None = None, trust: str | None = None, eligible: bool | None = None) -> pd.DataFrame:
        d = self.runs[self.runs.setup_id == sid]
        d = d[d.kind == kind] if kind else d
        d = d[d.trust == trust] if trust else d
        return d[d.eligible == eligible] if eligible is not None else d

    def clean_on(self, sid: str, answered: bool = True) -> pd.DataFrame:
        """Steps of clean runs with the allocator and a usable trust score (VQ1's population)."""
        s = self.steps[(self.steps.setup_id == sid) & (self.steps.kind == "baseline") & (self.steps.trust == "on") & self.steps.trust_score.notna()]
        return s[s.has_answer] if answered else s

    def rq_pairs(self, sid: str, strict: bool = True) -> pd.DataFrame:
        """One row per task with BOTH injected runs: columns <var>_off / <var>_on. strict = the two runs share the very same injection."""
        t = self.tasks[self.tasks.setup_id == sid].set_index("task_id")
        keep = t.index[(t.rq_reason == "")] if strict else t.index[t.eligible | (t.rq_reason == "injections differ between allocator off/on")]
        f = self.r(sid, "fork", eligible=True)
        f = f[f.task_id.isin(keep)]
        cols = ["final_correct", "propagation", "censored", "inj_step_correct", "trust_at_inj", "step_acc", "tok_audit", "prompt_tokens", "completion_tokens"]
        p = f.pivot(index="task_id", columns="trust", values=cols).dropna(subset=[("final_correct", "off"), ("final_correct", "on")])
        p.columns = [f"{a}_{b}" for a, b in p.columns]
        p = p.apply(pd.to_numeric, errors="coerce")  # pivot of mixed bool/None columns comes back as object
        meta = f[f.trust == "on"].set_index("task_id")[["level", "fm_id", "words_changed", "answer_changed", "has_file"]]
        p = p.join(meta, how="left").join(t[["orig_correct"]], how="left")
        for c in ("final_correct_off", "final_correct_on", "inj_step_correct_off", "inj_step_correct_on", "censored_off", "censored_on"):
            p[c] = p[c].astype(float).astype(bool)
        p["orig_correct"] = p.orig_correct.map(lambda v: bool(v) if v == v and v is not None else np.nan)
        return p

    def vq2_pairs(self, sid: str) -> pd.DataFrame:
        """Matched original / injected pairs: exact text match with the stored original and a valid trust score on both sides."""
        t = self.tasks[(self.tasks.setup_id == sid) & (self.tasks.vq2_reason == "")].set_index("task_id")
        f = self.r(sid, "fork", "on", eligible=True).set_index("task_id")
        f = f.loc[f.index.intersection(t.index)]
        return pd.DataFrame({"original": t.trust_orig.astype(float), "injected": t.trust_inj.astype(float)}).join(
            f[["fm_id", "level", "words_changed", "answer_changed", "inj_step_correct"]], how="inner")
