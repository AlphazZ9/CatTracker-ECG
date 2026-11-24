# -*- coding: utf-8 -*-
"""
ecg.io.group_compare
--------------------
The analyses of the Group tab on a group table (ecg.io.group_data): the
animals split by their group, their sex or not at all, and for each kind of
data (Pre/Post segments, Time course, Whole recording, Event-locked):

* the mean ± SEM (n) of each group;
* within each group, the paired test of the condition against the
  reference (Wilcoxon signed-rank, exact for small n, and the paired t);
* between the groups, on the change (or the value for a whole recording):
  Mann-Whitney and Welch's t for two groups, Kruskal-Wallis and one-way
  ANOVA for more;
* the two-way mixed ANOVA (group between, segment / time within, the
  Greenhouse-Geisser correction for time), on the animals with every value;
* Holm over the metrics tested, within each family of tests.

The experimental unit is the animal. The results carry the per-animal
values, the tables for GraphPad Prism and the data of the figure.

Nothing here may import from ecg.ui.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ecg.core.metric_families import family_of, family_rank, is_descriptive
from ecg.io.safe_write import replacing

from ecg.io.group_data import (
    SHEET_EVENTS, SHEET_SEGMENTS, SHEET_TIME, SHEET_WHOLE, TC_BIN_S, TC_METRICS, KEY,
    ALIGN_EXACT, ALIGN_PAST_SPAN, TC_ANCHOR_SPAN_S, anchor_lines, bin_edges, binned_values,
    onsets_named, stim_onsets, SHEET_ANNOT,
)
from ecg.io.group_stats import (
    DEFAULT_METRICS, _ttest, _wilcoxon, check_rows, consecutive_pairs, holm, mann_whitney_p,
    metric_names,
    segment_tags, smallest_mann_whitney_p, smallest_wilcoxon_p,
)

BY_NONE = "No groups"
BY_GROUP = "Group"
BY_SEX = "Sex"
BY_CHOICES = (BY_NONE, BY_GROUP, BY_SEX)

AN_SEGMENTS = "Pre/Post segments"
AN_TIME = "Time course"
AN_WHOLE = "Whole recording"
AN_EVENTS = "Event-locked"
ANALYSES = (AN_SEGMENTS, AN_TIME, AN_WHOLE, AN_EVENTS)

ALL_LABEL = "All animals"
CONTROL_NAMES = ("control", "ctrl", "ctl", "vehicle", "veh", "saline", "sham", "wt",
                 "wild", "témoin", "temoin", "sal")
SEX_NAMES = {"M": "Males", "F": "Females"}


# ── groups ───────────────────────────────────────────────────────────────

def animal_groups(animals: "pd.DataFrame", by: str) -> "tuple[dict[tuple[str, str], str], list[str], list[str]]":
    """({(subject, file): group name} of the animals used, the group names in
    order, notes on the animals left out). *by*: BY_NONE, BY_GROUP, BY_SEX."""
    out: "dict[tuple[str, str], str]" = {}
    notes: "list[str]" = []
    unticked: "list[str]" = []
    # 'subject (file)' when the subject has several recordings, so a note
    # says which one was left out.
    names = _names([(str(r["subject"]), str(r["file"])) for _, r in animals.iterrows()])
    for _, r in animals.iterrows():
        key = (str(r["subject"]), str(r["file"]))
        if not bool(r.get("use", True)):
            unticked.append(names[key])
            continue
        if by == BY_SEX:
            sex = str(r.get("sex", "") or "")
            if sex not in SEX_NAMES:
                notes.append(f"{names[key]}: sex unknown, left out")
                continue
            out[key] = SEX_NAMES[sex]
        elif by == BY_GROUP:
            g = str(r.get("group", "") or "").strip()
            if not g:
                notes.append(f"{names[key]}: no group, left out")
                continue
            out[key] = g
        else:
            out[key] = ALL_LABEL
    notes = list(dict.fromkeys(notes))
    if unticked:
        notes.append(f"Unticked in Animals, left out: {', '.join(dict.fromkeys(unticked))}")
    # Two recordings of one subject are two units of the tests: say so, the
    # user may have meant one animal.
    files: "dict[str, list[str]]" = {}
    for s, f in out:
        files.setdefault(s, []).append(f)
    twice = [f"{s} ({', '.join(fs)})" for s, fs in files.items() if len(fs) > 1]
    if twice:
        notes.append("One subject, several recordings, each counted as a separate animal in "
                     f"the tests: {'; '.join(twice)}")
    if by == BY_SEX:
        order = [n for n in ("Males", "Females") if n in out.values()]
    else:
        # The control group first (the reference of the figure and tables).
        order = sorted(set(out.values()),
                       key=lambda s: (not s.casefold().startswith(CONTROL_NAMES), s.casefold()))
    return out, order, notes


# ── statistics ───────────────────────────────────────────────────────────

def _num(v: Any) -> float:
    """*v* as a float, NaN when it is not a finite number."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f if math.isfinite(f) else math.nan


def _need_animals(gmap: "Mapping", by: str, animals: "Optional[pd.DataFrame]" = None) -> None:
    """No animal left once grouped: say why (rather than "no data"), none
    ticked in *animals* first (then no group or sex is not the reason)."""
    if gmap:
        return
    if animals is not None and not any(bool(r.get("use", True)) for _, r in animals.iterrows()):
        raise ValueError("No animal is ticked in Animals.")
    if by == BY_GROUP:
        raise ValueError("No animal has a group: type one in Animals, or choose No groups.")
    if by == BY_SEX:
        raise ValueError("No animal has a sex (M / F): set it in Animals, or choose No groups.")
    raise ValueError("No animal is ticked in Animals.")


def describe(v: "Sequence[float]") -> "tuple[float, float, int]":
    """(mean, SEM, n) of the finite values of *v*."""
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n == 0:
        return math.nan, math.nan, 0
    sem = float(np.std(a, ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    return float(np.mean(a)), sem, n


def between_test(samples: "Sequence[Sequence[float]]") -> "dict[str, Any]":
    """Rank and parametric tests between independent groups: Mann-Whitney
    and Welch's t for two, Kruskal-Wallis and one-way ANOVA for more."""
    from scipy import stats
    xs = [np.asarray(s, dtype=float) for s in samples]
    xs = [x[np.isfinite(x)] for x in xs]
    out: "dict[str, Any]" = {"rank test": "", "p": math.nan, "param test": "", "p param": math.nan}
    if len(xs) < 2 or any(len(x) < 1 for x in xs) or sum(len(x) for x in xs) < 3:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(xs) == 2:
            out["rank test"] = "Mann-Whitney"
            try:
                out["p"] = mann_whitney_p(xs[0], xs[1])
            except ValueError:
                pass
            out["param test"] = "Welch t"
            if all(len(x) >= 2 for x in xs):
                out["p param"] = float(stats.ttest_ind(xs[0], xs[1], equal_var=False).pvalue)
        else:
            out["rank test"] = "Kruskal-Wallis"
            try:
                out["p"] = float(stats.kruskal(*xs).pvalue)
            except ValueError:
                pass
            out["param test"] = "one-way ANOVA"
            if all(len(x) >= 2 for x in xs):
                out["p param"] = float(stats.f_oneway(*xs).pvalue)
    for k in ("p", "p param"):
        if not math.isfinite(out[k]):
            out[k] = math.nan
    return out


def within_test(ref: "Sequence[float]", cond: "Sequence[float]") -> "dict[str, Any]":
    """Paired tests of *cond* vs *ref* (same animals, in order): n, Wilcoxon
    p and paired t p on the finite pairs."""
    a = np.asarray(ref, dtype=float)
    b = np.asarray(cond, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = b[ok] - a[ok]
    _, p = _wilcoxon(d)
    return {"n": int(ok.sum()), "p": p, "p t": _ttest(d)}


def hedges_g(x: "Sequence[float]", y: "Sequence[float]") -> float:
    """Hedges' g of *y* against *x* (independent groups): the difference of
    the means over the pooled SD, with the small-sample correction."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return math.nan
    sp = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    # No spread up to rounding (identical values give ~1e-16, not 0): no
    # effect size, rather than g = 1e15.
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-300)
    if not sp > 1e-12 * scale:
        return math.nan
    j = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    return float(j * (b.mean() - a.mean()) / sp)


def cohen_dz(d: "Sequence[float]") -> float:
    """Cohen's dz of paired differences *d*: mean over SD."""
    a = np.asarray(d, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return math.nan
    sd = a.std(ddof=1)
    return float(a.mean() / sd) if sd > 0 else math.nan


def posthoc(samples: "Mapping[str, Sequence[float]]") -> "list[dict]":
    """Every pair of groups, Mann-Whitney, Holm over the pairs: after a
    Kruskal-Wallis on more than two groups."""
    from scipy import stats
    names = list(samples)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = np.asarray(samples[names[i]], dtype=float)
            b = np.asarray(samples[names[j]], dtype=float)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            p = math.nan
            if len(a) and len(b) and len(a) + len(b) >= 3:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        p = mann_whitney_p(a, b)
                    except ValueError:
                        pass
            rows.append({"pair": f"{names[j]} vs {names[i]}", "p": p,
                         "g": hedges_g(a, b)})
    adj = holm([r["p"] for r in rows])
    for r, q in zip(rows, adj):
        r["p Holm"] = q
    return rows


def _between_extras(row: dict, samples: "Mapping[str, Sequence[float]]",
                    order: "Sequence[str]") -> None:
    """Hedges' g (two groups: the second against the first, the control) or
    the post-hoc pairs (more than two) of a between-group test, into *row*."""
    if len(order) == 2:
        row["Hedges g"] = hedges_g(samples[order[0]], samples[order[1]])
    elif len(order) > 2:
        ph = posthoc({g: samples[g] for g in order})
        row["post-hoc"] = ph
        row["post-hoc (Mann-Whitney, Holm)"] = "; ".join(
            f"{r['pair']}: p = {fmt_p(r['p Holm'])}" for r in ph)


def _between_text(row: "Mapping[str, Any]") -> str:
    """The p of a between-group test with its test, and g for two groups."""
    text = f"{fmt_p(row.get('between p', math.nan))} ({row.get('between test', '')})"
    g = row.get("Hedges g", math.nan)
    if isinstance(g, float) and math.isfinite(g):
        text += f" · g = {g:.2f}"
    return text


def _posthoc_display(row: "Mapping[str, Any]", order: "Sequence[str]") -> "list[dict]":
    out = []
    for r in row.get("post-hoc") or []:
        d = {"Test": f"    post-hoc {r['pair']}", "p": f"{fmt_p(r['p'])} (Mann-Whitney)",
             "p Holm": fmt_p(r["p Holm"]), "_p": r["p Holm"], "_metric": row.get("Metric")}
        for g in order:
            d[g] = ""
        out.append(d)
    return out


def _gg_epsilon(y: np.ndarray, g: np.ndarray) -> float:
    """Greenhouse-Geisser epsilon of the within factor (columns of *y*) from
    the covariance pooled within the groups *g*."""
    n, k = y.shape
    if k < 3:
        return 1.0
    levels = list(dict.fromkeys(g.tolist()))
    resid = np.vstack([y[g == lv] - y[g == lv].mean(axis=0) for lv in levels])
    dfe = n - len(levels)
    if dfe < 1:
        return math.nan
    s = resid.T @ resid / dfe
    # Orthonormal contrasts orthogonal to the mean (Helmert).
    c = np.zeros((k - 1, k))
    for i in range(1, k):
        c[i - 1, :i] = 1.0
        c[i - 1, i] = -i
        c[i - 1] /= np.linalg.norm(c[i - 1])
    m = c @ s @ c.T
    tr = float(np.trace(m))
    tr2 = float(np.trace(m @ m))
    if tr2 <= 0:
        return math.nan
    return max(1.0 / (k - 1), min(1.0, tr * tr / ((k - 1) * tr2)))


def mixed_anova(y: "np.ndarray", groups: "Sequence[str]") -> "pd.DataFrame":
    """Two-way mixed ANOVA on *y* (animals × levels of the within factor,
    no missing value) with *groups* the between factor (one group: the
    one-way repeated-measures ANOVA). One row per effect: Source, df1, df2,
    F, p, epsilon (Greenhouse-Geisser, within effects) and p GG."""
    from scipy import stats
    y = np.asarray(y, dtype=float)
    g = np.asarray(list(groups), dtype=object)
    n, k = y.shape
    levels = list(dict.fromkeys(g.tolist()))
    a = len(levels)
    rows: "list[dict]" = []
    if n - a < 1 or k < 2:
        return pd.DataFrame(rows, columns=["Source", "df1", "df2", "F", "p", "epsilon", "p GG"])
    grand = y.mean()
    subj = y.mean(axis=1)
    lev = y.mean(axis=0)
    ss_subj = k * float(((subj - grand) ** 2).sum())
    ss_t = n * float(((lev - grand) ** 2).sum())
    ss_total = float(((y - grand) ** 2).sum())
    ss_g = 0.0
    ss_gt = 0.0
    for lv in levels:
        yg = y[g == lv]
        ng = len(yg)
        mg = yg.mean()
        ss_g += k * ng * (mg - grand) ** 2
        ss_gt += ng * float(((yg.mean(axis=0) - mg - lev + grand) ** 2).sum())
    ss_sg = ss_subj - ss_g
    ss_err = ss_total - ss_subj - ss_t - ss_gt
    df_sg, df_t = n - a, k - 1
    df_err = df_sg * df_t
    eps = _gg_epsilon(y, g)

    def _row(src: str, ss: float, df1: float, sse: float, df2: float, within: bool) -> dict:
        f = (ss / df1) / (sse / df2) if df1 > 0 and df2 > 0 and sse > 0 else math.nan
        p = float(stats.f.sf(f, df1, df2)) if math.isfinite(f) else math.nan
        e = eps if within else math.nan
        p_gg = (float(stats.f.sf(f, df1 * e, df2 * e))
                if within and math.isfinite(f) and math.isfinite(e) else math.nan)
        return {"Source": src, "df1": df1, "df2": df2, "F": f, "p": p, "epsilon": e,
                "p GG": p_gg}

    if a > 1:
        rows.append(_row("Group", ss_g, a - 1, ss_sg, df_sg, False))
    rows.append(_row("Within", ss_t, df_t, ss_err, df_err, True))
    if a > 1:
        rows.append(_row("Group × within", ss_gt, (a - 1) * df_t, ss_err, df_err, True))
    return pd.DataFrame(rows)


def fmt_p(p: float) -> str:
    if p is None or not math.isfinite(p):
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def fmt_mean(m: float, sem: float, n: int, decimals: int = 2) -> str:
    if not math.isfinite(m):
        return "—"
    s = f"{m:.{decimals}f}"
    if math.isfinite(sem):
        s += f" ± {sem:.{decimals}f}"
    return f"{s} ({n})"


def _decimals(values: "Sequence[float]") -> int:
    a = np.abs(np.asarray(values, dtype=float))
    a = a[np.isfinite(a)]
    if not len(a):
        return 2
    m = float(np.median(a))
    return 0 if m >= 100 else 1 if m >= 10 else 2 if m >= 1 else 3


# ── results ──────────────────────────────────────────────────────────────

@dataclass
class GroupResult:
    analysis: str
    by: str
    groups: "list[str]"
    # Results table of the tab: Test, one column per group, p, p Holm (text).
    display: "pd.DataFrame"
    # Every number behind it (Excel).
    stats: "pd.DataFrame"
    # One row per animal: subject, file, group, the values used.
    per_animal: "pd.DataFrame"
    anova: "pd.DataFrame" = field(default_factory=pd.DataFrame)
    notes: "list[str]" = field(default_factory=list)
    prism: "list[tuple]" = field(default_factory=list)
    figure: "dict[str, Any]" = field(default_factory=dict)
    settings: "list[tuple[str, Any]]" = field(default_factory=list)

    @property
    def n_animals(self) -> int:
        return int(len(self.per_animal))


def _row_title(subject: str, sex: str) -> str:
    return f"{subject} ({sex})" if sex in ("M", "F") else subject


def _twice(keys: "Sequence[tuple[str, str]]") -> "set[str]":
    """The subjects of *keys* that have more than one recording."""
    subjects = [k[0] for k in keys]
    return {s for s in subjects if subjects.count(s) > 1}


def _names(keys: "Sequence[tuple[str, str]]") -> "dict[tuple[str, str], str]":
    """A unique name per recording: its subject, 'subject (file)' when the
    subject has several recordings (the curves of the figure)."""
    twice = _twice(keys)
    return {k: f"{k[0]} ({k[1]})" if k[0] in twice else k[0] for k in keys}


def _titles(keys: "Sequence[tuple[str, str]]", sexes: "Mapping") -> "dict[tuple[str, str], str]":
    """A unique Prism row / column title per recording: the subject with its
    sex, and its file when the subject has several recordings (one title
    each, or a table keyed by title would keep only one of them)."""
    twice = _twice(keys)
    return {k: _row_title(k[0], sexes.get(k, "")) + (f" ({k[1]})" if k[0] in twice else "")
            for k in keys}


def _sexes(animals: "pd.DataFrame") -> "dict[tuple[str, str], str]":
    return {(str(r["subject"]), str(r["file"])): str(r.get("sex", "") or "")
            for _, r in animals.iterrows()}


def _small_n_note(ns: "Sequence[int]",
                  sizes: "Sequence[Sequence[int]]" = ()) -> "Optional[str]":
    """The smallest p the rank tests can reach when n is small: the Wilcoxon
    with the smallest of *ns* (pairs within a group), the Mann-Whitney with
    the two-group sizes *sizes* that reach the least (between groups)."""
    parts = []
    ns = [n for n in ns if n > 0]
    if ns and min(ns) <= 5:
        n = min(ns)
        parts.append(f"n = {n}: the smallest Wilcoxon p is {smallest_wilcoxon_p(n):.3g}")
    pairs = [(int(z[0]), int(z[1])) for z in sizes if len(z) == 2 and min(z) > 0]
    if pairs:
        n1, n2 = max(pairs, key=lambda z: smallest_mann_whitney_p(*z))
        if min(n1, n2) <= 5:
            parts.append(f"{n1} vs {n2}: the smallest Mann-Whitney p is "
                         f"{smallest_mann_whitney_p(n1, n2):.3g}")
    if not parts:
        return None
    return "; ".join(parts) + "; see also the paired / Welch t (Statistics… workbook)"


def _finite_n(v: "Sequence[float]") -> int:
    return int(np.isfinite(np.asarray(v, dtype=float)).sum())


def _column_table(title: str, cols: "Mapping[str, Sequence[float]]") -> tuple:
    n = max((len(v) for v in cols.values()), default=0)
    frame = pd.DataFrame({k: list(v) + [math.nan] * (n - len(v)) for k, v in cols.items()})
    return (title, frame)


def _paired_table(title: str, rows: "Sequence[str]", cols: "Mapping[str, Sequence[float]]") -> tuple:
    frame = pd.DataFrame({k: list(v) for k, v in cols.items()}, index=list(rows))
    return (title, frame)


# The p column of a row shown but not tested (Data analysed: beats,
# duration, blocks used).
DESCRIPTIVE = "descriptive (data analysed, not tested)"


def _holm_column(df: "pd.DataFrame", col: str, out: str) -> None:
    """Holm over the rows tested: a row not tested (column Tested False, a
    Data analysed row) has no p and is not counted in the family."""
    if col in df.columns and len(df):
        p = [float(x) for x in df[col]]
        if "Tested" in df.columns:
            p = [x if bool(t) else math.nan for x, t in zip(p, df["Tested"])]
        df[out] = holm(p)


def _two_condition(analysis: str, by: str, order: "list[str]", gmap: "dict",
                   sexes: "dict", values: "dict[str, dict[tuple, tuple[float, float]]]",
                   names: "tuple[str, str]", notes: "list[str]",
                   short: "Mapping[str, str]", within: str = "Segment") -> GroupResult:
    """Shared by Pre/Post segments and Event-locked: per metric and animal a
    (reference, condition) pair; within-group paired tests, between-group
    tests on Δ, mixed ANOVA group × condition."""
    na, nb = names
    stat_rows: "list[dict]" = []
    disp: "list[dict]" = []
    anova_rows: "list[pd.DataFrame]" = []
    per_animal: "dict[tuple, dict]" = {}
    prism: "list[tuple]" = []
    ns: "list[int]" = []
    sizes: "list[list[int]]" = []
    titles = _titles(list(gmap), sexes)
    for metric, vals in values.items():
        keys = [k for k in gmap if k in vals]
        # Beats, Duration, Blocks used: how much data, not an outcome. Shown
        # per group, never tested, so not in the Holm family either.
        desc = is_descriptive(metric)
        row: "dict[str, Any]" = {"Metric": metric, "Tested": not desc}
        dec = _decimals([v for k in keys for v in vals[k]])
        dec_d = _decimals([vals[k][1] - vals[k][0] for k in keys])
        deltas: "dict[str, list[float]]" = {}
        for g in order:
            ks = [k for k in keys if gmap[k] == g]
            a = [vals[k][0] for k in ks]
            b = [vals[k][1] for k in ks]
            d = [y - x for x, y in zip(a, b)]
            deltas[g] = d
            for lab, v in ((na, a), (nb, b), ("Δ", d)):
                m, s, n = describe(v)
                row[f"{g} {lab} mean"], row[f"{g} {lab} SEM"], row[f"{g} {lab} n"] = m, s, n
            w = within_test(a, b)
            row[f"{g} n pairs"] = w["n"]
            row[f"{g} Wilcoxon p"] = math.nan if desc else w["p"]
            row[f"{g} paired t p"] = math.nan if desc else w["p t"]
            row[f"{g} dz"] = math.nan if desc else cohen_dz(d)
            if not desc:
                ns.append(w["n"])
            for k in ks:
                pa = per_animal.setdefault(k, {"subject": k[0], "file": k[1], "group": g})
                pa[f"{short.get(metric, metric)} {na}"] = vals[k][0]
                pa[f"{short.get(metric, metric)} {nb}"] = vals[k][1]
            prism.append(_paired_table(
                f"{metric}{' · ' + g if len(order) > 1 else ''}",
                [titles[k] for k in ks], {na: a, nb: b}))
        if len(order) > 1:
            if desc:
                row["between test"], row["between p"] = "descriptive", math.nan
                row["between param test"], row["between param p"] = "", math.nan
            else:
                if len(order) == 2:
                    sizes.append([_finite_n(deltas[g]) for g in order])
                bt = between_test([deltas[g] for g in order])
                row["between test"], row["between p"] = bt["rank test"], bt["p"]
                row["between param test"], row["between param p"] = (bt["param test"],
                                                                      bt["p param"])
                _between_extras(row, deltas, order)
            prism.append(_column_table(f"Δ {metric} ({nb} − {na})", deltas))
        complete = [k for k in keys if all(math.isfinite(x) for x in vals[k])]
        if not desc and len(complete) >= len(order) + 1:
            y = np.array([vals[k] for k in complete], dtype=float)
            an = mixed_anova(y, [gmap[k] for k in complete])
            if len(an):
                an.insert(0, "Metric", metric)
                an["Source"] = an["Source"].replace({"Within": within,
                                                     "Group × within": f"Group × {within.lower()}"})
                anova_rows.append(an)
        stat_rows.append(row)
        row["_dec"], row["_dec_d"] = dec, dec_d
    stats = pd.DataFrame(stat_rows)
    for g in order:
        _holm_column(stats, f"{g} Wilcoxon p", f"{g} Wilcoxon p (Holm)")
    if len(order) > 1:
        _holm_column(stats, "between p", "between p (Holm)")
    for _, r in stats.iterrows():
        dec = int(r["_dec"])
        desc = not bool(r["Tested"])
        d1 = {"Test": f"Δ {r['Metric']} ({nb} − {na})", "_metric": r["Metric"]}
        if len(order) > 1:
            for g in order:
                d1[g] = fmt_mean(r[f"{g} Δ mean"], r[f"{g} Δ SEM"], int(r[f"{g} Δ n"]),
                                 int(r["_dec_d"]))
            if desc:
                d1["p"], d1["p Holm"], d1["_p"] = DESCRIPTIVE, "", math.nan
            else:
                d1["p"] = _between_text(r)
                d1["p Holm"] = fmt_p(r["between p (Holm)"])
                d1["_p"] = r["between p (Holm)"]
            disp.append(d1)
            disp.extend(_posthoc_display(r, order))
        d2 = {"Test": f"{r['Metric']}: {nb} vs {na}", "_metric": r["Metric"]}
        for g in order:
            d2[g] = (f"{fmt_mean(r[f'{g} {na} mean'], r[f'{g} {na} SEM'], int(r[f'{g} {na} n']), dec)}"
                     f" → {fmt_mean(r[f'{g} {nb} mean'], r[f'{g} {nb} SEM'], int(r[f'{g} {nb} n']), dec)}"
                     + ("" if desc else f"   p = {fmt_p(r[f'{g} Wilcoxon p (Holm)'])}"))
        d2["p"] = DESCRIPTIVE if desc else "within group: Wilcoxon (Holm)"
        d2["p Holm"] = ""
        d2["_p"] = min([r[f"{g} Wilcoxon p (Holm)"] for g in order
                        if math.isfinite(r[f"{g} Wilcoxon p (Holm)"])], default=math.nan)
        disp.append(d2)
    stats = stats.drop(columns=["_dec", "_dec_d", "post-hoc"], errors="ignore")
    note = _small_n_note(ns, sizes)
    if note:
        notes.append(note)
    anova = pd.concat(anova_rows, ignore_index=True) if anova_rows else pd.DataFrame()
    pa = pd.DataFrame(list(per_animal.values()))
    return GroupResult(analysis=analysis, by=by, groups=order, display=_display(disp, order),
                       stats=stats, per_animal=pa, anova=anova, notes=notes, prism=prism)


def _display(rows: "list[dict]", order: "list[str]", metric: str = "") -> "pd.DataFrame":
    """The results table. Columns starting with "_" are for the screen only
    (the p for the colour, the metric of the row for the family headers);
    *metric* is the metric of rows that do not name one (one-metric analyses)."""
    cols = ["Test", *order, "p", "p Holm", "_p", "_metric"]
    df = pd.DataFrame(rows, columns=cols)
    if metric:
        df["_metric"] = df["_metric"].where(df["_metric"].notna(), metric)
    return df


def display_by_family(display: "pd.DataFrame") -> "list[tuple[str, pd.DataFrame]]":
    """The results table cut into sections for the screen: (header, rows),
    one section per metric family (Heart rate, HRV time domain...) in family
    order, the metrics in their own order inside. With all pairs, each pair
    keeps its place and is cut on its own ("Heart rate · B vs A"). A table of
    a single metric is one section without a header ("", display)."""
    d = display.reset_index(drop=True)
    if "_metric" not in d.columns:
        return [("", d)]
    metrics = [m for m in d["_metric"] if isinstance(m, str) and m]
    if len(set(metrics)) < 2:
        return [("", d)]
    pairs = d["_pair"] if "_pair" in d.columns else pd.Series([""] * len(d))
    out: "list[tuple[str, pd.DataFrame]]" = []
    for pair in dict.fromkeys(pairs):
        block = d.loc[pairs == pair]
        fams = [family_of(m) if isinstance(m, str) else "" for m in block["_metric"]]
        # A row without a metric stays with the row above it.
        for i in range(1, len(fams)):
            if not fams[i]:
                fams[i] = fams[i - 1]
        seen = list(dict.fromkeys(fams))
        for fam in sorted(seen, key=family_rank):
            rows = block.loc[[f == fam for f in fams]]
            head = fam or "Other"
            out.append((f"{head} · {pair}" if pair else head, rows))
    return out


# ── the four analyses ────────────────────────────────────────────────────

PAIRS_ONE = "One pair"
PAIRS_ALL = "All pairs"
POINCARE = "Poincaré (group)"


def analyse_segments(sheets: "Mapping[str, pd.DataFrame]", animals: "pd.DataFrame", by: str,
                     a: str, b: str, metrics: "Optional[Sequence[str]]" = None,
                     names: "tuple[str, str]" = ("", ""), pairs: str = PAIRS_ONE,
                     leave_out: bool = True) -> GroupResult:
    """Pre/Post segments of the Group sheet, per metric (*metrics* None:
    DEFAULT_METRICS the table holds; "all": all): segment *b* vs *a*, or
    with *pairs* PAIRS_ALL every consecutive pair of segments (B vs A, D vs
    C...; Holm within each pair). Each row is checked for the pair (segment
    missing, quality alert, subject on several rows, another template or
    duration: ecg.io.group_stats.check_rows); with *leave_out* those rows are
    left out, and listed either way."""
    df = sheets.get(SHEET_SEGMENTS)
    if df is None or df.empty:
        raise ValueError("The group table has no segments (Compare segments → Add to group table).")
    df = df.reset_index(drop=True)
    tags = segment_tags(df)
    gmap, order, notes = animal_groups(animals, by)
    _need_animals(gmap, by, animals)
    if pairs == PAIRS_ALL:
        todo, odd = consecutive_pairs(tags)
        if not todo:
            raise ValueError("The table holds fewer than two segments.")
        if odd:
            notes.append(f"Segment {odd} has no partner: not tested")
    else:
        if a not in tags or b not in tags or a == b:
            raise ValueError("Choose two different segments of the table.")
        todo = [(a, b)]
    multi = len(todo) > 1
    sexes = _sexes(animals)
    parts: "list[GroupResult]" = []
    fig_values: "dict[str, dict]" = {}
    fig_names: "dict[str, tuple[str, str]]" = {}
    poincare: "dict[str, tuple]" = {}
    for pa, pb in todo:
        tag = f" · {pb} vs {pa}" if multi else ""
        pnotes: "list[str]" = []
        checks = check_rows(df, pa, pb)
        doubtful = {(c.subject, c.file): c for c in checks if not c.ok}
        listed = [f"{c.subject or '?'} ({'; '.join(c.problems)})" for k, c in doubtful.items()
                  if k in gmap and not all(p.startswith("no segment") for p in c.problems)]
        if listed:
            pnotes.append(f"{'Left out' if leave_out else 'To check (kept)'}{tag}: "
                          + ", ".join(listed))
        available = metric_names(df, pa, pb)
        if metrics is None:
            names_m = [m for m in DEFAULT_METRICS if m in available] or available
        elif metrics == "all":
            names_m = available
        else:
            names_m = [m for m in metrics if m in available]
        values: "dict[str, dict]" = {}
        used_rows = []
        for i, r in df.iterrows():
            key = (str(r["subject"]), str(r["file"]))
            if key not in gmap or (leave_out and key in doubtful):
                continue
            used_rows.append(i)
            for m in names_m:
                va, vb = _num(r.get(f"{m} {pa}")), _num(r.get(f"{m} {pb}"))
                if math.isfinite(va) and math.isfinite(vb):
                    values.setdefault(m, {})[key] = (float(va), float(vb))
        values = {m: values.get(m, {}) for m in names_m}
        missing = [k[0] for k in gmap
                   if not any(k in v for v in values.values()) and k not in doubtful]
        if missing:
            pnotes.append(f"No {pa} / {pb} segments: {', '.join(missing)}")
        na, nb = (names[0] or pa, names[1] or pb) if not multi else (pa, pb)
        part = _two_condition(AN_SEGMENTS, by, order, gmap, sexes, values, (na, nb),
                              pnotes, {})
        if multi:
            part.display["Test"] = [f"[{pb} vs {pa}] {t}" for t in part.display["Test"]]
            part.display["_pair"] = f"{pb} vs {pa}"
            for frame in (part.stats, part.anova):
                if len(frame):
                    frame.insert(0, "Pair", f"{pb} vs {pa}")
            part.prism = [(f"{t[0]}{tag}",) + tuple(t[1:]) for t in part.prism]
        parts.append(part)
        for m, v in values.items():
            fig_values[f"{m}{tag}"] = v
            fig_names[f"{m}{tag}"] = (na, nb)
        poincare[f"{POINCARE}{tag}"] = (df.loc[used_rows], pa, pb, (na, nb))
    if len(parts) == 1:
        res = parts[0]
        res.notes = notes + res.notes
    else:
        per = parts[0].per_animal
        for p_ in parts[1:]:
            if len(p_.per_animal):
                per = p_.per_animal if not len(per) else per.merge(
                    p_.per_animal, on=["subject", "file", "group"], how="outer")
        res = GroupResult(
            analysis=AN_SEGMENTS, by=by, groups=order,
            display=pd.concat([p_.display for p_ in parts], ignore_index=True),
            stats=pd.concat([p_.stats for p_ in parts], ignore_index=True),
            per_animal=per,
            anova=pd.concat([p_.anova for p_ in parts if len(p_.anova)], ignore_index=True)
            if any(len(p_.anova) for p_ in parts) else pd.DataFrame(),
            notes=notes + [n for p_ in parts for n in p_.notes if n not in notes],
            prism=[t for p_ in parts for t in p_.prism])
        # One note on the smallest p a Wilcoxon can reach, not one per pair.
        seen: "list[str]" = []
        res.notes = [n for n in res.notes if not (n in seen or seen.append(n))]
    first = next(iter(fig_values), "")
    res.figure = {"kind": "paired", "values": fig_values, "names_by": fig_names,
                  "gmap": gmap, "names": fig_names.get(first, ("A", "B")), "metric": first,
                  "poincare": poincare}
    res.settings = [("Analysis", AN_SEGMENTS), ("By", by),
                    ("Segments", PAIRS_ALL + ": " + ", ".join(f"{pb} vs {pa}" for pa, pb in todo)
                     if multi else f"{todo[0][1]} vs {todo[0][0]}"),
                    ("Rows to check", "left out" if leave_out else "kept (listed)")]
    return res


def whole_metrics(sheets: "Mapping[str, pd.DataFrame]") -> "list[str]":
    df = sheets.get(SHEET_WHOLE)
    if df is None:
        return []
    return [c for c in df.columns if c not in ("subject", "sex", "group", "use", "file")]


def analyse_whole(sheets: "Mapping[str, pd.DataFrame]", animals: "pd.DataFrame", by: str,
                  metrics: "Optional[Sequence[str]]" = None) -> GroupResult:
    """Whole recording: each metric compared between the groups."""
    df = sheets.get(SHEET_WHOLE)
    if df is None or df.empty:
        raise ValueError("The group table has no whole-recording metrics "
                         "(Group tab → Add open recording, after Analyze).")
    gmap, order, notes = animal_groups(animals, by)
    _need_animals(gmap, by, animals)
    available = whole_metrics(sheets)
    if metrics is None:
        names_m = [m for m in DEFAULT_METRICS if m in available] or available[:8]
    elif metrics == "all":
        names_m = available
    else:
        names_m = [m for m in metrics if m in available]
    rows, disp, prism = [], [], []
    per_animal: "dict[tuple, dict]" = {}
    values: "dict[str, dict]" = {}
    sizes: "list[list[int]]" = []
    for m in names_m:
        vals = {}
        for _, r in df.iterrows():
            key = (str(r["subject"]), str(r["file"]))
            v = _num(r.get(m))
            if key in gmap and math.isfinite(v):
                vals[key] = float(v)
                per_animal.setdefault(key, {"subject": key[0], "file": key[1],
                                            "group": gmap[key]})[m] = float(v)
        values[m] = vals
        desc = is_descriptive(m)
        row: "dict[str, Any]" = {"Metric": m, "Tested": not desc}
        by_g = {g: [vals[k] for k in gmap if gmap[k] == g and k in vals] for g in order}
        for g in order:
            mm, s, n = describe(by_g[g])
            row[f"{g} mean"], row[f"{g} SEM"], row[f"{g} n"] = mm, s, n
        if len(order) > 1 and desc:
            row.update({"between test": "descriptive", "between p": math.nan,
                        "between param test": "", "between param p": math.nan})
        elif len(order) > 1:
            if len(order) == 2:
                sizes.append([len(by_g[g]) for g in order])
            bt = between_test([by_g[g] for g in order])
            row.update({"between test": bt["rank test"], "between p": bt["p"],
                        "between param test": bt["param test"], "between param p": bt["p param"]})
            _between_extras(row, by_g, order)
        rows.append(row)
        prism.append(_column_table(m, by_g))
    stats = pd.DataFrame(rows)
    if len(order) > 1:
        _holm_column(stats, "between p", "between p (Holm)")
    for _, r in stats.iterrows():
        dec = _decimals([r[f"{g} mean"] for g in order])
        d = {"Test": str(r["Metric"]), "_metric": str(r["Metric"])}
        for g in order:
            d[g] = fmt_mean(r[f"{g} mean"], r[f"{g} SEM"], int(r[f"{g} n"]), dec)
        if len(order) > 1 and not bool(r["Tested"]):
            d["p"], d["p Holm"], d["_p"] = DESCRIPTIVE, "", math.nan
        elif len(order) > 1:
            d["p"] = _between_text(r)
            d["p Holm"] = fmt_p(r["between p (Holm)"])
            d["_p"] = r["between p (Holm)"]
        else:
            d["p"], d["p Holm"], d["_p"] = "descriptive", "", math.nan
        disp.append(d)
        disp.extend(_posthoc_display(r, order))
    if len(order) < 2:
        notes.append("One group: descriptive only (choose By Group or Sex to compare)")
    note = _small_n_note([], sizes)
    if note:
        notes.append(note)
    res = GroupResult(analysis=AN_WHOLE, by=by, groups=order, display=_display(disp, order),
                      stats=stats.drop(columns=["post-hoc"], errors="ignore"), per_animal=pd.DataFrame(list(per_animal.values())),
                      notes=notes, prism=prism)
    res.figure = {"kind": "dots", "values": values, "gmap": gmap,
                  "metric": names_m[0] if names_m else ""}
    res.settings = [("Analysis", AN_WHOLE), ("By", by)]
    return res


def event_labels(sheets: "Mapping[str, pd.DataFrame]") -> "list[str]":
    df = sheets.get(SHEET_EVENTS)
    if df is None or df.empty or "label" not in df.columns:
        return []
    return sorted({str(v) for v in df["label"] if str(v) not in ("", "nan")}, key=str.casefold)


def event_metrics(sheets: "Mapping[str, pd.DataFrame]") -> "list[str]":
    df = sheets.get(SHEET_EVENTS)
    if df is None:
        return []
    return [c[:-4] for c in df.columns if str(c).endswith(" pre") and f"{c[:-4]} post" in df.columns]


def _truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() not in ("false", "0", "no", "")
    if isinstance(v, float) and not math.isfinite(v):
        return False
    return bool(v)


def _config_text(method: Any, window: Any) -> str:
    m = "" if method is None or (isinstance(method, float) and not math.isfinite(method)) \
        else str(method).strip()
    w = _num(window)
    return " · ".join(p for p in (m or "default", f"{w:g} s" if math.isfinite(w) else "") if p)


def event_configs(sheets: "Mapping[str, pd.DataFrame]") -> "list[str]":
    """The event-locked configurations of the table ("robust · 15 s"), the
    most used first: one recording can hold the same events analysed with
    several methods or windows, which must not be pooled."""
    df = sheets.get(SHEET_EVENTS)
    if df is None or df.empty:
        return []
    texts = [_config_text(m, w) for m, w in zip(df.get("method", [None] * len(df)),
                                                 df.get("window_s", [None] * len(df)))]
    counts = pd.Series(texts).value_counts()
    return sorted(counts.index, key=lambda t: (-int(counts[t]), t))


def analyse_events(sheets: "Mapping[str, pd.DataFrame]", animals: "pd.DataFrame", by: str,
                   label: str = "", metrics: "Optional[Sequence[str]]" = None,
                   config: "Optional[str]" = None) -> GroupResult:
    """Event-locked: per animal the mean of its events named *label* ('' :
    all) that are valid for each metric, in one configuration (*config*, one
    of event_configs(); None: the most used), post vs pre within each group
    and Δ between groups."""
    df = sheets.get(SHEET_EVENTS)
    if df is None or df.empty:
        raise ValueError("The group table has no event-locked results (save them to the event "
                         "registry in Time course → Autonomic, then Add open recording).")
    gmap, order, notes = animal_groups(animals, by)
    _need_animals(gmap, by, animals)
    ev = df
    configs = event_configs(sheets)
    config = config if config in configs else configs[0]
    ev = ev.loc[[_config_text(m, w) == config for m, w in
                 zip(ev.get("method", [None] * len(ev)), ev.get("window_s", [None] * len(ev)))]]
    if len(configs) > 1:
        notes.append(f"Configuration {config} only (the table also holds "
                     f"{', '.join(c for c in configs if c != config)})")
    if label:
        ev = ev.loc[ev["label"].astype(str) == label]
    available = event_metrics(sheets)
    names_m = list(metrics) if metrics not in (None, "all") else available
    names_m = [m for m in names_m if m in available]
    if metrics is None:
        pref = [m for m in ("rCSI", "rCPI", "HR", "SD1", "SD2") if m in names_m]
        names_m = pref or names_m[:5]
    values: "dict[str, dict]" = {}
    for m in names_m:
        vals = {}
        for (s, f), sub in ev.groupby(list(KEY), sort=False):
            key = (str(s), str(f))
            if key not in gmap:
                continue
            pre = pd.to_numeric(sub[f"{m} pre"], errors="coerce").to_numpy(float)
            post = pd.to_numeric(sub[f"{m} post"], errors="coerce").to_numpy(float)
            # The series' own validity (tables made before it: the event's).
            vcol = f"{m} valid" if f"{m} valid" in sub.columns else "valid"
            valid = (np.array([_truthy(v) for v in sub[vcol]], dtype=bool)
                     if vcol in sub.columns else np.ones(len(sub), bool))
            ok = np.isfinite(pre) & np.isfinite(post) & valid
            if ok.any():
                vals[key] = (float(pre[ok].mean()), float(post[ok].mean()))
        values[m] = vals
    notes.append("Per animal: the mean of its events" + (f" “{label}”" if label else ""))
    res = _two_condition(AN_EVENTS, by, order, gmap, _sexes(animals), values, ("pre", "post"),
                         notes, {}, within="Pre/post")
    res.figure = {"kind": "paired", "values": values, "gmap": gmap, "names": ("pre", "post"),
                  "metric": names_m[0] if names_m else ""}
    res.settings = [("Analysis", AN_EVENTS), ("By", by), ("Events", label or "all"),
                    ("Configuration", config)]
    return res


def _ordinal(n: int) -> str:
    """1st, 2nd, 3rd, 4th..."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _bins_mean(v: np.ndarray) -> float:
    """Mean of the per-bin values *v*; NaN unless at least half are valid."""
    ok = np.isfinite(v)
    if not len(v) or 2 * int(ok.sum()) < len(v):
        return math.nan
    return float(v[ok].mean())


def time_bin_size(bin_s: float) -> "tuple[float, str]":
    """(bin size, note): *bin_s* (s) on the table's TC_BIN_S grid -- a
    non-multiple rounded to the nearest one, the note saying so ('' when it
    was one). A size below TC_BIN_S, or not a number, is refused."""
    try:
        b = float(bin_s)
    except (TypeError, ValueError):
        raise ValueError(f"Bins: {bin_s!r} is not a number of seconds.") from None
    if not math.isfinite(b) or b < TC_BIN_S - 1e-9:
        raise ValueError(f"Bins: the size must be at least {TC_BIN_S:g} s "
                         f"(the table keeps {TC_BIN_S:g} s sums), not {bin_s:g}.")
    r = max(TC_BIN_S, round(b / TC_BIN_S) * TC_BIN_S)
    if abs(r - b) > 1e-9:
        return r, (f"Bins of {b:g} s rounded to {r:g} s (a multiple of {TC_BIN_S:g} s, "
                   "the table's sums)")
    return r, ""


def _on_grid(t_s: float) -> float:
    """*t_s* (s) rounded to the nearest multiple of TC_BIN_S."""
    return round(t_s / TC_BIN_S) * TC_BIN_S


def time_span(start_min: float, end_min: float) -> "tuple[float, float, list[str]]":
    """(from, to) s of a time course on the table's TC_BIN_S grid, and the
    notes saying so: a 'from' or 'to' between two fine bins would make an end
    bin that no fine bin is centred in (empty for every animal) or one that
    takes a fine bin past it. Each stays at least TC_BIN_S from 0."""
    a = min(_on_grid(start_min * 60), -TC_BIN_S)
    b = max(_on_grid(end_min * 60), TC_BIN_S)
    notes = [f"'{what}' {v:g} min moved to {r / 60:g} min (on the table's {TC_BIN_S:g} s bins)"
             for what, v, r in (("from", start_min, a), ("to", end_min, b))
             if abs(r - v * 60) > 1e-6]
    return a, b, notes


def analyse_time(sheets: "Mapping[str, pd.DataFrame]", animals: "pd.DataFrame", by: str,
                 stim: str, metric: str = "HR", bin_s: float = 60.0,
                 start_min: float = -10.0, end_min: float = 30.0,
                 window_min: "tuple[float, float]" = (0.0, 5.0), delta: bool = True,
                 occurrence: int = 0, min_coverage: float = 0.5) -> GroupResult:
    """Time course aligned on the annotation *stim* of each animal: *metric*
    (HR, RR, SDNN, RMSSD) in bins of *bin_s* (a multiple of TC_BIN_S,
    anchored on 0) from *start_min* to *end_min* min (moved onto the
    TC_BIN_S grid, time_span). Per animal, the
    baseline is the mean of its bins before 0 (Δ from it with *delta*) and
    the test window value the mean of its bins centred in *window_min*,
    each NaN unless at least half of those bins are valid; the window
    against the baseline within each group, its Δ between groups; the mixed
    ANOVA group × time on the bins."""
    tc = sheets.get(SHEET_TIME)
    if tc is None or tc.empty:
        raise ValueError("The group table has no time course (Group tab → Add open recording, "
                         "after Analyze).")
    if metric not in TC_METRICS:
        raise ValueError(f"Unknown metric {metric}")
    if not (start_min < 0 < end_min):
        raise ValueError("The time course must start before 0 and end after it.")
    w0, w1 = window_min
    if not (w1 > w0):
        raise ValueError("The test window must end after it starts.")
    bin_s, bin_note = time_bin_size(bin_s)
    start_s, end_s, span_notes = time_span(start_min, end_min)
    start_min, end_min = start_s / 60, end_s / 60
    edges = bin_edges(start_s, end_s, bin_s)
    starts = edges[:-1]
    centres = (edges[:-1] + edges[1:]) / 2
    pre = centres < 0
    in_win = (centres >= w0 * 60) & (centres < w1 * 60)
    if not in_win.any():
        raise ValueError(f"No bin is centred in the test window {w0:g}–{w1:g} min: widen it, "
                         f"use smaller bins or extend 'to'.")
    # The window as tested: the edges of the bins centred in it, which the
    # labels show (0–5 min with 2 min bins is 0–4 min).
    wb = np.flatnonzero(in_win)
    u0, u1 = edges[wb[0]] / 60, edges[wb[-1] + 1] / 60
    win_txt = f"{u0:g}–{u1:g}"
    gmap, order, notes = animal_groups(animals, by)
    _need_animals(gmap, by, animals)
    if bin_note:
        notes.append(bin_note)
    notes.extend(span_notes)
    if edges[1] - edges[0] < bin_s - 1e-9:
        notes.append(f"First bin cut by 'from': {edges[0] / 60:g} to {edges[1] / 60:g} min")
    if edges[-1] - edges[-2] < bin_s - 1e-9:
        notes.append(f"Last bin cut by 'to': {edges[-2] / 60:g} to {edges[-1] / 60:g} min")
    if w0 < start_min or w1 > end_min:
        notes.append(f"Test window {w0:g}–{w1:g} min goes past the time course ({start_min:g} "
                     f"to {end_min:g} min): only its bins inside it are used")
    if w0 < 0:
        notes.append(f"Test window {w0:g}–{w1:g} min starts before 0: it overlaps the "
                     f"baseline ({start_min:g}–0 min)")
    onsets = stim_onsets(sheets.get(SHEET_ANNOT))
    sexes = _sexes(animals)
    names = _names(list(gmap))
    titles = _titles(list(gmap), sexes)
    unit = TC_METRICS[metric]
    series: "dict[tuple, np.ndarray]" = {}
    base_v: "dict[tuple, float]" = {}
    win_v: "dict[tuple, float]" = {}
    no_stim, no_tc, approx, past, short = [], [], [], [], []
    several: "dict[int, list[str]]" = {}
    for (s, f), sub in tc.groupby(list(KEY), sort=False):
        key = (str(s), str(f))
        if key not in gmap:
            continue
        hits = onsets_named(onsets.get(key, []), stim)
        t0 = hits[occurrence] if 0 <= occurrence < len(hits) else None
        if t0 is None:
            no_stim.append(names[key])
            continue
        if len(hits) > 1:
            several.setdefault(len(hits), []).append(names[key])
        lines, how = anchor_lines(sub, t0, (start_s, end_s))
        if how != ALIGN_EXACT:
            (past if how == ALIGN_PAST_SPAN else approx).append(names[key])
        rel = lines.assign(t_s=lines["t_s"].to_numpy(float) - t0)
        v = binned_values(rel, metric, edges, min_coverage)
        # Baseline and window from the same bins as the curve, so a Δ curve
        # averages 0 before the time 0 and the test compares like with like.
        base, win = _bins_mean(v[pre]), _bins_mean(v[in_win])
        for what, val, part in (("baseline", base, v[pre]), ("test window", win, v[in_win])):
            if not math.isfinite(val):
                need = (len(part) + 1) // 2
                short.append(f"{names[key]}: only {_finite_n(part)} of its {len(part)} {what} "
                             f"bins have data (at least {need} needed) — left out of the "
                             "baseline, window and Δ means and the tests")
        series[key] = v - base if delta else v
        base_v[key], win_v[key] = base, win
    for k in gmap:
        if k not in series and k[0] not in no_stim and not ((tc["subject"] == k[0])
                                                          & (tc["file"] == k[1])).any():
            no_tc.append(names[k])
    if no_stim:
        notes.append(f"No “{stim}” annotation, left out: {', '.join(no_stim)}")
    if no_tc:
        notes.append(f"No time course, left out: {', '.join(no_tc)}")
    if not series:
        raise ValueError(f"No animal has both a time course and a “{stim}” annotation.")
    if approx:
        notes.append(f"Aligned on “{stim}” within ±{TC_BIN_S / 2:g} s only (the table has no "
                     f"bins laid out on it: added before exact alignment or before “{stim}” "
                     f"was annotated, or left out to fit Excel's row limit): "
                     f"{', '.join(approx)}. Re-add these recordings (“Add open "
                     "recording” / Import a project) for exact alignment")
    if past:
        lo, hi = (x / 3600 for x in TC_ANCHOR_SPAN_S)
        notes.append(f"Aligned on “{stim}” within ±{TC_BIN_S / 2:g} s only: 'from' / 'to' go "
                     f"past the {lo:g} to +{hi:g} h kept exact around each annotation: "
                     f"{', '.join(past)}")
    for count, who in sorted(several.items()):
        notes.append(f"{count} “{stim}” annotations, the {_ordinal(occurrence + 1)} is used: "
                     f"{', '.join(who)}")
    notes.extend(short)
    notes.append(f"Per animal, baseline: the mean of its {int(pre.sum())} bins before 0; test "
                 f"window: the mean of its {len(wb)} bins in {win_txt} min")
    t_min = centres / 60.0
    label = f"{'Δ ' if delta else ''}{metric} ({unit})"
    disp, rows, prism = [], [], []
    curves: "dict[str, dict]" = {}
    ns = []
    deltas: "dict[str, list[float]]" = {}
    row: "dict[str, Any]" = {"Metric": f"{metric} ({unit})",
                             "window (min)": win_txt, "baseline (min)": f"{start_min:g}–0"}
    for g in order:
        ks = [k for k in series if gmap[k] == g]
        mat = np.array([series[k] for k in ks]) if ks else np.empty((0, len(starts)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean = np.nanmean(mat, axis=0) if len(ks) else np.full(len(starts), np.nan)
            cnt = np.sum(np.isfinite(mat), axis=0) if len(ks) else np.zeros(len(starts))
            sd = np.nanstd(mat, axis=0, ddof=1) if len(ks) > 1 else np.full(len(starts), np.nan)
        sem = np.where(cnt > 1, sd / np.sqrt(np.maximum(cnt, 1)), np.nan)
        # One curve per recording (a subject with two files keeps both), and
        # n the animals that add at least one value to the mean.
        curves[g] = {"mean": mean, "sem": sem,
                     "n": int(sum(bool(np.isfinite(series[k]).any()) for k in ks)),
                     "animals": {names[k]: series[k] for k in ks}}
        b = [base_v[k] for k in ks]
        w = [win_v[k] for k in ks]
        d = [y - x for x, y in zip(b, w)]
        deltas[g] = d
        # The means over the same animals (both values), so window minus
        # baseline is the Δ.
        both = [i for i, x in enumerate(d) if math.isfinite(x)]
        for lab, v in (("baseline", [b[i] for i in both]), ("window", [w[i] for i in both]),
                       ("Δ", d)):
            m, s, n = describe(v)
            row[f"{g} {lab} mean"], row[f"{g} {lab} SEM"], row[f"{g} {lab} n"] = m, s, n
        wt = within_test(b, w)
        row[f"{g} Wilcoxon p"], row[f"{g} paired t p"] = wt["p"], wt["p t"]
        ns.append(wt["n"])
        frame = pd.DataFrame({titles[k]: series[k] for k in ks})
        frame.insert(0, "Time (min)", t_min)
        prism.append((f"{label} · time course{' · ' + g if len(order) > 1 else ''}",
                      frame, "Time (min)"))
        prism.append(_paired_table(
            f"{metric} ({unit}) baseline vs {win_txt} min"
            f"{' · ' + g if len(order) > 1 else ''}",
            [titles[k] for k in ks],
            {f"baseline ({start_min:g}–0 min)": b, f"{win_txt} min": w}))
    sizes = [[_finite_n(deltas[g]) for g in order]] if len(order) == 2 else []
    if len(order) > 1:
        bt = between_test([deltas[g] for g in order])
        row.update({"between test": bt["rank test"], "between p": bt["p"],
                    "between param test": bt["param test"], "between param p": bt["p param"]})
        _between_extras(row, deltas, order)
        prism.append(_column_table(f"Δ {metric} ({unit}) {win_txt} min vs baseline", deltas))
    for g in order:
        row[f"{g} dz"] = cohen_dz(deltas[g])
    rows.append(row)
    stats = pd.DataFrame(rows).drop(columns=["post-hoc"], errors="ignore")
    dec = _decimals([row.get(f"{g} Δ mean", math.nan) for g in order])
    if len(order) > 1:
        d1 = {"Test": f"Δ {metric} {win_txt} min ({unit})"}
        for g in order:
            d1[g] = fmt_mean(row[f"{g} Δ mean"], row[f"{g} Δ SEM"], int(row[f"{g} Δ n"]), dec)
        d1["p"] = _between_text(row)
        d1["p Holm"] = ""
        d1["_p"] = row["between p"]
        disp.append(d1)
        disp.extend(_posthoc_display(row, order))
    d2 = {"Test": f"Within group, {win_txt} min vs baseline"}
    for g in order:
        d2[g] = f"p = {fmt_p(row[f'{g} Wilcoxon p'])}  (t: {fmt_p(row[f'{g} paired t p'])})"
    d2["p"], d2["p Holm"] = "Wilcoxon (paired t)", ""
    d2["_p"] = min([row[f"{g} Wilcoxon p"] for g in order
                    if math.isfinite(row[f"{g} Wilcoxon p"])], default=math.nan)
    disp.append(d2)
    # Mixed ANOVA on the animals with every bin.
    complete = [k for k in series if np.all(np.isfinite(series[k]))]
    dropped = [names[k] for k in series if k not in complete]
    anova = pd.DataFrame()
    groups_c = [gmap[k] for k in complete]
    if len(complete) >= len(set(groups_c)) + 1 and len(starts) >= 2 and complete:
        anova = mixed_anova(np.array([series[k] for k in complete]), groups_c)
        anova["Source"] = anova["Source"].replace({"Within": "Time",
                                                   "Group × within": "Group × time"})
        anova.insert(0, "Metric", label)
        for _, a in anova.iterrows():
            p = a["p GG"] if math.isfinite(a["p GG"]) else a["p"]
            eps = f", GG ε = {a['epsilon']:.2f}" if math.isfinite(a["epsilon"]) else ""
            d = {"Test": f"RM ANOVA {a['Source']}"}
            d[order[0]] = (f"F({a['df1']:g}, {a['df2']:g}) = {a['F']:.2f}{eps}"
                           if math.isfinite(a["F"]) else "—")
            for g in order[1:]:
                d[g] = ""
            d["p"], d["p Holm"], d["_p"] = fmt_p(p), "", p
            disp.append(d)
        if len(starts) > len(complete):
            # More levels than animals: the covariance of the bins cannot be
            # estimated and the GG epsilon sits at its floor.
            notes.append(f"The ANOVA has {len(starts)} time levels for {len(complete)} animals: "
                         "larger bins (fewer levels than animals) make it more reliable")
    if dropped:
        notes.append(f"Left out of the ANOVA (an empty bin): {', '.join(dropped)}")
    note = _small_n_note(ns, sizes)
    if note:
        notes.append(note)
    notes.append(f"A bin covered less than {min_coverage:.0%} by valid RR is empty for that animal")
    res = GroupResult(analysis=AN_TIME, by=by, groups=order,
                      display=_display(disp, order, metric=metric),
                      stats=stats, per_animal=pd.DataFrame(
                          [{"subject": k[0], "file": k[1], "group": gmap[k],
                            "baseline": base_v[k], "window": win_v[k],
                            **{f"{t:g} min": v for t, v in zip(t_min, series[k])}}
                           for k in series]),
                      anova=anova, notes=notes, prism=prism)
    res.figure = {"kind": "time", "t_min": t_min, "curves": curves, "label": label,
                  "window": (u0, u1), "stim": stim, "bin_s": bin_s}
    res.settings = [("Analysis", AN_TIME), ("By", by), ("Time 0", stim),
                    ("Metric", label), ("Bins (s)", bin_s),
                    ("From / to (min)", f"{start_min:g} / {end_min:g}"),
                    ("Test window (min)", win_txt if win_txt == f"{w0:g}–{w1:g}"
                     else f"{w0:g}–{w1:g} (bins {win_txt})")]
    return res


# ── statistics workbook ──────────────────────────────────────────────────

def _with_family(stats: "pd.DataFrame") -> "pd.DataFrame":
    """*stats* with a Family column after Metric (Heart rate, HRV time
    domain...), to sort or filter the metrics in Excel."""
    if "Metric" not in stats.columns or "Family" in stats.columns or not len(stats):
        return stats
    out = stats.copy()
    out.insert(list(out.columns).index("Metric") + 1, "Family",
               [family_of(m) for m in out["Metric"]])
    return out


def write_results(path: str, res: GroupResult, table_name: str = "") -> None:
    """The statistics of *res* as an Excel workbook: Results (the table of
    the tab), Statistics (every number), ANOVA, Per animal, Settings."""
    settings = [("Group table", table_name)] + list(res.settings) + \
        [("Groups", ", ".join(res.groups)), ("Animals", res.n_animals)] + \
        [("Note", n) for n in res.notes]
    with replacing(path) as tmp, \
            pd.ExcelWriter(tmp, engine="openpyxl") as xw:
        res.display.drop(columns=[c for c in res.display.columns if str(c).startswith("_")]
                         ).to_excel(xw, sheet_name="Results", index=False)
        _with_family(res.stats).to_excel(xw, sheet_name="Statistics", index=False)
        if len(res.anova):
            res.anova.to_excel(xw, sheet_name="ANOVA", index=False)
        res.per_animal.to_excel(xw, sheet_name="Per animal", index=False)
        pd.DataFrame(settings, columns=["Setting", "Value"]).to_excel(
            xw, sheet_name="Settings", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(60, max(10, width + 2))
