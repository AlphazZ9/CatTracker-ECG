# -*- coding: utf-8 -*-
"""
ecg.io.group_stats
------------------
Group statistics on a Compare segments group table (ecg.io.group_table: one
row per recording, columns "<metric> <segment>"): the rows to check, then
for two segments A and B a paired test per metric across animals -- the
experimental unit is the animal, never the beat or the block.

Per metric: n animals with both values, medians, median and mean of the
paired differences B − A, the two-sided Wilcoxon signed-rank test (exact
for small n, scipy) and the paired t-test, and Holm-adjusted p-values over
the metrics tested.

All pairs (all_pairs_stats): the same tests for every consecutive pair of
segments of the table (B vs A, D vs C, F vs E: Pre / Post 1, 5, 10 min),
Holm within each pair and optionally across all pairs × metrics, with a
check that each segment carries the same label in every animal.
Nothing here may import from ecg.ui.
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ecg.core.metric_families import is_descriptive
from ecg.io.safe_write import replacing

from ecg.io.subject_name import normalise_sex

# Per-segment columns of the group table that are not metrics.
META_PREFIXES = ("start (s)", "end (s)", "valid RR (%)", "Poincaré pairs (n)",
                 "abrupt RR jumps (n)", "QC", "label", "index")
# Per-recording columns that are not metrics either; "sex" (M / F / empty,
# from the Recording sheet) is absent from tables made before it. The
# statistics run on every animal whatever its sex.
ID_COLUMNS = ("subject", "sex", "file", "template")
DURATION_TOL = 0.02          # durations within 2 % of the group's median

# Metrics tested by default: one per distinct question, so that the Holm
# family is not inflated by equivalent tests. Left out on purpose: RMSSD
# (= SD1 · √2 up to the mean difference), LF n.u. and HF n.u. (HF n.u. =
# 100 − LF n.u., LF/HF a monotone function of it), pNN6 (often 0 in
# anaesthetised mice); "All metrics" tests every column.
DEFAULT_METRICS = ("Mean HR (60000/mean NN) (bpm)", "SDNN (ms)", "Poincaré SD1 (ms)",
                   "Poincaré SD2 (ms)", "LF/HF")
DEFAULT_METRICS_NOTE = ("RMSSD ≈ SD1·√2, LF n.u. / HF n.u. ≡ LF/HF and pNN6 are not tested "
                        "by default (equivalent or often 0): tick All metrics to test them")


def read_group_table(path: str) -> "pd.DataFrame":
    """The "Group" sheet of a group table written by ecg.io.group_table."""
    df = pd.read_excel(path, sheet_name="Group")
    for c in ("subject", "file"):
        if c not in df.columns:
            raise ValueError(f"{os.path.basename(path)} is not a group table (no '{c}' column)")
    return df


def segment_tags(df: "pd.DataFrame") -> "list[str]":
    """The segments of the table that hold at least one value, in column order."""
    tags = []
    for c in df.columns:
        if c.startswith("start (s) "):
            tag = c[len("start (s) "):]
            if df[c].notna().any():
                tags.append(tag)
    return tags


def metric_names(df: "pd.DataFrame", a: str, b: str) -> "list[str]":
    """Metric names (label with unit) that the table holds for both *a* and *b*."""
    out = []
    tags = [c[len("start (s) "):] for c in df.columns if c.startswith("start (s) ")]
    for c in df.columns:
        if not c.endswith(f" {a}"):
            continue
        # A column of a longer tag ending the same way ("SDNN (ms) Post A"
        # for tag "A" when "Post A" is a tag) is that tag's, not a metric.
        if any(t != a and len(t) > len(a) and c.endswith(f" {t}") for t in tags):
            continue
        name = c[: -len(a) - 1]
        if name in ID_COLUMNS or name.startswith(META_PREFIXES):
            continue
        if f"{name} {b}" in df.columns and name not in out:
            out.append(name)
    return out


@dataclass
class RowCheck:
    subject: str
    file: str
    problems: "list[str]" = field(default_factory=list)
    sex: str = ""               # M / F / '' (shown in the reports, never tested)

    @property
    def ok(self) -> bool:
        return not self.problems


def subject_text(v: object) -> str:
    """A subject cell as text: '' for an empty cell (NaN), '243' for 243.0."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def sex_text(v: object) -> str:
    """A sex cell as text: 'M', 'F', '' for an empty cell, else as written."""
    return normalise_sex(subject_text(v))


def check_rows(df: "pd.DataFrame", a: str, b: str) -> "list[RowCheck]":
    """Per row, what makes it doubtful for the A vs B comparison: a segment
    missing, a quality alert, a subject shared with another row (the unit is
    the animal), another template, or a duration other than the group's."""
    checks = []
    both = pd.Series(True, index=df.index)
    for tag in (a, b):
        col = f"start (s) {tag}"
        both &= df[col].notna() if col in df.columns else False
    # Only rows of this comparison count: another protocol's row of the same
    # animal, or its template, says nothing about this one.
    subj = df["subject"].map(subject_text)
    dup = subj.where(both).duplicated(keep=False) & both & (subj != "")
    templates = df["template"].fillna("").astype(str) if "template" in df.columns else None
    tpl_both = templates[both & (templates != "")] if templates is not None else pd.Series([], dtype=str)
    counts = tpl_both.value_counts()
    # A clear majority only: a tie names no reference.
    common_tpl = (counts.index[0] if len(counts) and (len(counts) == 1 or
                                                      counts.iloc[0] > counts.iloc[1]) else "")

    def dur(tag: str) -> "pd.Series":
        s, e = f"start (s) {tag}", f"end (s) {tag}"
        if s in df.columns and e in df.columns:
            return pd.to_numeric(df[e], errors="coerce") - pd.to_numeric(df[s], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    durs = {tag: dur(tag) for tag in (a, b)}
    med = {tag: float(np.nanmedian(d)) if d.notna().any() else math.nan for tag, d in durs.items()}
    for i, row in df.iterrows():
        rc = RowCheck(subject_text(row.get("subject")), subject_text(row.get("file")),
                      sex=sex_text(row.get("sex")))
        if not rc.subject:
            rc.problems.append("no subject")
        elif dup.iloc[i]:
            rc.problems.append(f"subject {rc.subject} on several rows")
        for tag in (a, b):
            if pd.isna(row.get(f"start (s) {tag}")):
                rc.problems.append(f"no segment '{tag}'")
                continue
            qc = row.get(f"QC {tag}")
            if isinstance(qc, str) and qc.strip() and qc.strip().lower() != "ok":
                rc.problems.append(f"{tag}: {qc.strip()}")
            d, m = durs[tag].iloc[i], med[tag]
            if np.isfinite(d) and np.isfinite(m) and m > 0 and abs(d - m) > DURATION_TOL * m:
                rc.problems.append(f"{tag} lasts {d:g} s, not {m:g} s like the group")
        if (common_tpl and templates is not None and both.iloc[i]
                and templates.iloc[i] != common_tpl):
            rc.problems.append(f"template '{templates.iloc[i] or 'none'}', not '{common_tpl}'")
        checks.append(rc)
    return checks


def holm(p: "Sequence[float]") -> "list[float]":
    """Holm-Bonferroni adjusted p-values (NaN kept, not counted)."""
    p = [float(x) for x in p]
    idx = [i for i, x in enumerate(p) if np.isfinite(x)]
    m = len(idx)
    adj = [math.nan] * len(p)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda k: p[k])):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


# Up to this many arrangements, a rank test with tied values is worked out
# exactly by enumerating them. scipy falls back on the normal approximation
# there, which with 3 vs 3 animals can give p = 0.047 where no split of 6
# animals goes below 0.1 (smallest_mann_whitney_p, quoted in the results).
_EXACT_MAX = 20000


def _has_ties(v: "np.ndarray") -> bool:
    return len(np.unique(v)) < len(v)


def _exact_two_sided(stat_all: "np.ndarray", obs: float) -> float:
    """Two-sided p over every arrangement: twice the smaller one-sided share
    (as scipy's exact tests), with a small tolerance for the half ranks."""
    s = np.asarray(stat_all, dtype=float)
    lo = np.mean(s <= obs + 1e-9)
    hi = np.mean(s >= obs - 1e-9)
    return float(min(1.0, 2.0 * min(lo, hi)))


def mann_whitney_p(x: "Sequence[float]", y: "Sequence[float]") -> float:
    """Two-sided Mann-Whitney p of two independent samples: scipy's, except
    with tied values in small samples, where it is the exact permutation p
    (every split of the pooled values into groups of these sizes, mid-ranks),
    so it never goes below smallest_mann_whitney_p."""
    from itertools import combinations
    from scipy import stats
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    n1, n2 = len(a), len(b)
    pooled = np.concatenate([a, b])
    if n1 and n2 and _has_ties(pooled) and math.comb(n1 + n2, n1) <= _EXACT_MAX:
        ranks = stats.rankdata(pooled)
        idx = np.array(list(combinations(range(n1 + n2), n1)), dtype=int)
        return _exact_two_sided(ranks[idx].sum(axis=1), float(ranks[:n1].sum()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def _wilcoxon(d: "np.ndarray") -> "tuple[float, float]":
    from scipy import stats
    if len(d) < 2 or np.all(d == 0):
        return math.nan, math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = stats.wilcoxon(d, alternative="two-sided")
    w, p = float(r.statistic), float(r.pvalue)
    # Tied |differences|: the exact sign-flip p, the zeros dropped as scipy's
    # default does (older scipy uses the normal approximation there), so p
    # never goes below smallest_wilcoxon_p.
    nz = np.asarray(d, dtype=float)
    nz = nz[nz != 0]
    if len(nz) and _has_ties(np.abs(nz)) and 2 ** len(nz) <= _EXACT_MAX:
        ranks = stats.rankdata(np.abs(nz))
        signs = (np.arange(2 ** len(nz))[:, None] >> np.arange(len(nz))) & 1
        p = _exact_two_sided((signs * ranks).sum(axis=1), float(ranks[nz > 0].sum()))
    return w, p


def _ttest(d: "np.ndarray") -> float:
    from scipy import stats
    if len(d) < 2 or np.std(d, ddof=1) == 0:
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.ttest_1samp(d, 0.0).pvalue)


def paired_stats(df: "pd.DataFrame", a: str, b: str,
                 metrics: "Optional[Sequence[str]]" = None) -> "pd.DataFrame":
    """One row per metric: n animals with both values, median A, median B,
    median and mean of B − A, Wilcoxon W and p, paired t p, and Holm-adjusted
    Wilcoxon p over the metrics tested. *metrics* None: the DEFAULT_METRICS
    the table holds (counts and durations are not outcomes, and every extra
    metric makes Holm stricter); "all": every metric of the table."""
    available = metric_names(df, a, b)
    if metrics is None:
        names = [m for m in DEFAULT_METRICS if m in available] or available
    elif metrics == "all":
        names = available
    else:
        names = list(metrics)
    rows = []
    for name in names:
        ca, cb = f"{name} {a}", f"{name} {b}"
        if ca not in df.columns or cb not in df.columns:
            continue
        va = pd.to_numeric(df[ca], errors="coerce").to_numpy(float)
        vb = pd.to_numeric(df[cb], errors="coerce").to_numpy(float)
        ok = np.isfinite(va) & np.isfinite(vb)
        va, vb = va[ok], vb[ok]
        d = vb - va
        # Beats, Duration, Blocks used: described, not tested (no p, so not
        # in the Holm family either).
        tested = not is_descriptive(name)
        w, p_w = _wilcoxon(d) if tested else (math.nan, math.nan)
        rows.append({
            "Metric": name, "n": int(ok.sum()),
            # Wilcoxon leaves out zero differences: this is its n.
            "n tested (Δ ≠ 0)": int((d != 0).sum()),
            f"Median {a}": float(np.median(va)) if len(va) else math.nan,
            f"Median {b}": float(np.median(vb)) if len(vb) else math.nan,
            "Median Δ (B − A)": float(np.median(d)) if len(d) else math.nan,
            "Mean Δ (B − A)": float(np.mean(d)) if len(d) else math.nan,
            "n up / down": f"{int((d > 0).sum())} / {int((d < 0).sum())}",
            "Wilcoxon W": w, "Wilcoxon p": p_w,
            "Paired t p": _ttest(d) if tested else math.nan, "Tested": tested,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["Wilcoxon p (Holm)"] = holm(out["Wilcoxon p"].tolist())
    return out


def smallest_wilcoxon_p(n: int) -> float:
    """The smallest two-sided p an exact Wilcoxon signed-rank test can give
    with *n* pairs (2 / 2^n): below this the test cannot reach significance."""
    return 2.0 / (2 ** n) if n > 0 else math.nan


def smallest_mann_whitney_p(n1: int, n2: int) -> float:
    """The smallest two-sided p an exact Mann-Whitney test can give with
    groups of *n1* and *n2* (2 / C(n1 + n2, n1): the two groups fully apart)."""
    if n1 < 1 or n2 < 1:
        return math.nan
    return min(1.0, 2.0 / math.comb(n1 + n2, n1))


def _report_frames(df: "pd.DataFrame", a: str, b: str, stats: "pd.DataFrame",
                   checks: "Sequence[RowCheck]", excluded: "Sequence[str]" = (),
                   note: str = "") -> "dict[str, pd.DataFrame]":
    """The four tables of a single-pair report: Statistics, About, Rows
    checked, Paired values (one row per animal and metric tested)."""
    info = [("Segments compared", f"{a} (A) vs {b} (B)"),
            ("Test", "two-sided Wilcoxon signed-rank (exact for small n); paired t for "
                     "comparison; Holm correction over the metrics"),
            ("Unit", "the animal: one row per recording, one recording per animal"),
            ("Rows left out", ", ".join(excluded) if excluded else "none")]
    tested = (stats.loc[stats["Tested"].astype(bool)] if len(stats) and "Tested" in stats.columns
              else stats)
    n = int(tested["n tested (Δ ≠ 0)"].max()) if len(tested) else 0
    if n:
        info.append(("Smallest possible Wilcoxon p",
                     f"{smallest_wilcoxon_p(n):.3g} (n = {n} animals with a non-zero difference; "
                     "per metric: column n tested)"))
    info.append(("Metrics tested", ", ".join(tested["Metric"]) if len(tested) else "none"))
    shown = [m for m in (stats["Metric"] if len(stats) else []) if m not in set(tested["Metric"])]
    if shown:
        info.append(("Described, not tested", ", ".join(shown) + " (the data analysed: no p, "
                     "not counted in Holm)"))
    if note:
        info.append(("Note", note))
    # The sex beside the subject when the table has it (tables made before
    # the Sex field have none: the columns are then as before).
    has_sex = "sex" in df.columns
    paired = []
    for name in stats["Metric"] if len(stats) else []:
        for _i, r in df.iterrows():
            row = {"Metric": name, "subject": r.get("subject")}
            if has_sex:
                row["sex"] = sex_text(r.get("sex"))
            row.update({"file": r.get("file"), a: r.get(f"{name} {a}"),
                        b: r.get(f"{name} {b}")})
            paired.append(row)
    checked_cols = ["subject"] + (["sex"] if has_sex else []) + ["file", "status", "problems"]
    return {
        "Statistics": stats,
        "About": pd.DataFrame(info, columns=["Item", "Value"]),
        "Rows checked": pd.DataFrame(
            [{"subject": c.subject, "sex": c.sex, "file": c.file,
              "status": "ok" if c.ok else "check",
              "problems": "; ".join(c.problems)} for c in checks],
            columns=checked_cols),
        "Paired values": pd.DataFrame(paired),
    }


def _fit_columns(xw) -> None:
    for ws in xw.book.worksheets:
        for col in ws.columns:
            width = max((len(str(c.value)) for c in col[:200] if c.value is not None), default=8)
            ws.column_dimensions[col[0].column_letter].width = min(width + 3, 70)


def write_group_report(path: str, df: "pd.DataFrame", a: str, b: str,
                       stats: "pd.DataFrame", checks: "Sequence[RowCheck]",
                       excluded: "Sequence[str]" = (), note: str = "") -> None:
    """Excel report: Statistics, Rows checked, then the paired values (one
    row per animal) of every metric tested."""
    frames = _report_frames(df, a, b, stats, checks, excluded, note)
    with replacing(path) as tmp, \
            pd.ExcelWriter(tmp, engine="openpyxl") as xw:
        for name, frame in frames.items():
            frame.to_excel(xw, sheet_name=name, index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
        _fit_columns(xw)


# ── All pairs: B vs A, D vs C, F vs E ──────────────────────────────────────

def consecutive_pairs(tags: "Sequence[str]") -> "tuple[list[tuple[str, str]], Optional[str]]":
    """(A, B), (C, D), (E, F)... of *tags* in order, and the odd last tag
    left without a partner (None when the count is even)."""
    tags = list(tags)
    pairs = [(tags[i], tags[i + 1]) for i in range(0, len(tags) - 1, 2)]
    return pairs, (tags[-1] if len(tags) % 2 else None)


def _norm_label(v: object) -> str:
    """A label for comparison: case and spacing do not count."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return " ".join(str(v).split()).casefold()


def segment_labels(df: "pd.DataFrame", tag: str) -> "tuple[str, list[Optional[str]]]":
    """The label of segment *tag* in the group -- the clear majority of the
    rows that have the segment and a label, as typed; '' when the table
    stores no label (tables written before the "label <tag>" columns) or on a
    tie -- and the label of each row (None when the row has no segment or no
    label stored)."""
    has = (df[f"start (s) {tag}"].notna() if f"start (s) {tag}" in df.columns
           else pd.Series(False, index=df.index))
    col = f"label {tag}"
    if col not in df.columns:
        return "", [None] * len(df)
    # An empty cell -- a row added before the labels were kept, or an empty
    # label (Excel reads "" back as empty) -- is unknown, not a mismatch.
    raw: "list[Optional[str]]" = [
        None if not h or pd.isna(v) or not str(v).strip() else " ".join(str(v).split())
        for v, h in zip(df[col], has)]
    seen: "dict[str, list[str]]" = {}
    for v in raw:
        if v is not None:
            seen.setdefault(_norm_label(v), []).append(v)
    ranked = sorted(seen.values(), key=len, reverse=True)
    if not ranked or (len(ranked) > 1 and len(ranked[0]) == len(ranked[1])):
        return "", raw
    return ranked[0][0], raw


def label_problems(df: "pd.DataFrame", tags: "Sequence[str]"
                   ) -> "tuple[list[list[str]], list[str]]":
    """Per row, the segments of *tags* whose label is not the group's
    ("A is 'Pre 5min', not 'Pre 1min' like the group"), and notes on the
    tags whose labels cannot be checked (none stored, or no majority)."""
    probs: "list[list[str]]" = [[] for _ in range(len(df))]
    notes: "list[str]" = []
    for tag in tags:
        if f"label {tag}" not in df.columns:
            notes.append(f"{tag}: no label stored (table written before labels were kept), "
                         "labels not checked")
            continue
        ref, raw = segment_labels(df, tag)
        if not ref:
            kinds = sorted({v for v in raw if v is not None})
            if len(kinds) > 1:
                notes.append(f"{tag}: no majority label ({', '.join(repr(k) for k in kinds)}), "
                             "labels not checked")
            continue
        for i, v in enumerate(raw):
            if v is not None and _norm_label(v) != _norm_label(ref):
                probs[i].append(f"{tag} is '{v}', not '{ref}' like the group")
    return probs, notes


def _all_tags(df: "pd.DataFrame") -> "list[str]":
    return [c[len("start (s) "):] for c in df.columns if str(c).startswith("start (s) ")]


def _tag_columns(df: "pd.DataFrame", tag: str, tags: "Sequence[str]") -> "dict[str, str]":
    """{name: column} of segment *tag* ("SDNN (ms)": "SDNN (ms) Pre"), without
    the columns of a longer tag ending the same way."""
    out = {}
    for c in df.columns:
        c = str(c)
        if not c.endswith(f" {tag}"):
            continue
        if any(t != tag and len(t) > len(tag) and c.endswith(f" {t}") for t in tags):
            continue
        name = c[: -len(tag) - 1]
        if name not in ID_COLUMNS:
            out[name] = c
    return out


@dataclass
class SegmentLayout:
    """The segments of a group table in the order of the recordings: the
    group's segment at each place (1st, 2nd...) and, per row, which of its
    own segments sits at each place."""
    tags: "list[str]"                       # the group's segment at each place
    rows: "list[dict[int, str]]"            # per row: place -> the row's own tag
    notes: "list[str]" = field(default_factory=list)
    extra: "list[list[str]]" = field(default_factory=list)   # per row: tags not placed


def _row_order(df: "pd.DataFrame", i: int, tags: "Sequence[str]") -> "list[str]":
    """The segments of row *i* in their order in the recording: by the
    "index <tag>" stored since the all-pairs mode, by start time otherwise."""
    row = df.iloc[i]
    keyed = []
    for t in tags:
        st = pd.to_numeric(row.get(f"start (s) {t}"), errors="coerce")
        if pd.isna(st):
            continue
        ix = pd.to_numeric(row.get(f"index {t}"), errors="coerce")
        keyed.append(((float(ix) if pd.notna(ix) else math.inf), float(st), t))
    return [k[2] for k in sorted(keyed)]


def segment_layout(df: "pd.DataFrame") -> SegmentLayout:
    """Which segments of the rows go together. Each row's segments are taken
    in their own order; the group's segment at each place is the one most
    rows hold there (case and spacing do not count; on a tie the first in
    the table). A row's segment is put at the group's place of the same
    name wherever it sits, otherwise at its own place: a segment of another
    name there ("Post 1 min" where the group has "Post 1") is a row to check
    (all_pairs_stats), never a pair of its own."""
    tags = _all_tags(df)
    col_rank = {t: k for k, t in enumerate(tags)}
    orders = [_row_order(df, i, tags) for i in range(len(df))]
    counts: "dict[int, int]" = {}
    for o in orders:
        if o:
            counts[len(o)] = counts.get(len(o), 0) + 1
    if not counts:
        return SegmentLayout([], [{} for _ in orders], [], [[] for _ in orders])
    top = max(counts.values())
    n_places = max(k for k, v in counts.items() if v == top)
    notes: "list[str]" = []
    if len(counts) > 1:
        notes.append(f"the rows do not all hold {n_places} segments ("
                     + ", ".join(f"{v} row(s) with {k}" for k, v in sorted(counts.items())) + ")")
    group: "list[str]" = []
    for p in range(n_places):
        by: "dict[str, list[str]]" = {}
        for o in orders:
            if len(o) > p:
                by.setdefault(_norm_label(o[p]), []).append(o[p])
        used = {_norm_label(g) for g in group}
        ranked = sorted(by.values(), key=lambda v: (_norm_label(v[0]) in used, -len(v),
                                                    min(col_rank[t] for t in v)))
        win = ranked[0]
        exact = sorted(set(win), key=lambda t: (-win.count(t), col_rank[t]))[0]
        if (len(ranked) > 1 and len(ranked[1]) == len(win)
                and _norm_label(ranked[1][0]) not in used):
            tied = [r[0] for r in ranked if len(r) == len(win) and _norm_label(r[0]) not in used]
            notes.append(f"segment {p + 1}: " + " / ".join(repr(t) for t in tied)
                         + f" are held by as many rows each; '{exact}' is used, "
                         "the other rows are rows to check")
        group.append(exact)
    gnorm = [_norm_label(g) for g in group]
    rows, extra = [], []
    for o in orders:
        place: "dict[int, str]" = {}
        left = []
        for t in o:                               # the same segment, wherever it sits
            q = next((q for q, g in enumerate(gnorm) if g == _norm_label(t) and q not in place),
                     None)
            if q is None:
                left.append(t)
            else:
                place[q] = t
        rest = []
        for t in left:                            # another name at its own place
            p = o.index(t)
            if p < n_places and p not in place:
                place[p] = t
            else:
                rest.append(t)
        rows.append(place)
        extra.append(rest)
    return SegmentLayout(group, rows, notes, extra)


def _placed_frame(df: "pd.DataFrame", layout: SegmentLayout, places: "Sequence[int]",
                  ) -> "pd.DataFrame":
    """*df* with each row's segment at *places* under the group's tag of that
    place, so that check_rows and paired_stats read it as usual."""
    tags = _all_tags(df)
    moves = [(i, layout.rows[i][p], layout.tags[p]) for i in range(len(df)) for p in places
             if p in layout.rows[i] and layout.rows[i][p] != layout.tags[p]]
    if not moves:
        return df
    work = df.copy()
    for i, own, grp in moves:
        for col in _tag_columns(df, grp, tags).values():   # what the row had under it
            if work[col].dtype != object:
                work[col] = work[col].astype(object)
            work.iloc[i, work.columns.get_loc(col)] = math.nan
        for name, col in _tag_columns(df, own, tags).items():
            dst = f"{name} {grp}"
            v = df.iloc[i][col]
            if dst not in work.columns:
                work[dst] = pd.Series([math.nan] * len(work), index=work.index, dtype=object)
            elif work[dst].dtype != object:
                work[dst] = work[dst].astype(object)
            work.iloc[i, work.columns.get_loc(dst)] = v
    return work


@dataclass
class PairResult:
    """One pair of an all-pairs run: the rows used and their paired tests."""
    a: str
    b: str
    label_a: str                    # the group's label of each segment ('' unknown)
    label_b: str
    used: "pd.DataFrame"
    stats: "pd.DataFrame"
    checks: "list[RowCheck]"
    keep: "list[bool]"
    left_out: "list[str]"
    notes: "list[str]" = field(default_factory=list)

    @property
    def name_a(self) -> str:
        return self.label_a or self.a

    @property
    def name_b(self) -> str:
        return self.label_b or self.b

    @property
    def title(self) -> str:
        """'Post Stim 1min vs Pre Stim 1min'; with the tags when they are not
        the labels ('Post vs Pre (B vs A)'), as repeated labels would give
        every pair the same name."""
        return pair_title(self.a, self.b, self.label_a, self.label_b)


def pair_title(a: str, b: str, label_a: str = "", label_b: str = "") -> str:
    """The name of the pair B vs A (PairResult.title): its labels, with the
    tags when they are not the labels."""
    names = f"{label_b or b} vs {label_a or a}"
    tags = f"{b} vs {a}"
    return names if names == tags else f"{names} ({tags})"


@dataclass
class AllPairsResult:
    pairs: "list[PairResult]"
    odd: "Optional[str]"            # the last segment left without a partner
    holm_across: bool
    notes: "list[str]" = field(default_factory=list)


HOLM_ALL = "Wilcoxon p (Holm, all pairs)"


def all_pairs_stats(df: "pd.DataFrame", metrics: "Optional[Sequence[str]]" = None,
                    exclude: bool = True, holm_across: bool = False,
                    tags: "Optional[Sequence[str]]" = None) -> AllPairsResult:
    """paired_stats for every consecutive pair of segments (B vs A, D vs C,
    F vs E...; an odd last segment is left out with a note) on the rows of
    each pair that pass check_rows and a name / label check: a row whose
    segment there has another name or label than the group's (another
    window) is a row to check, left out with the others when *exclude*
    (kept but listed otherwise). The pairs follow each row's own order of
    segments (segment_layout), so that segments of a few rows named
    otherwise never make a pair of their own; *tags* given: those tags,
    paired in that order. Holm within each pair (column "Wilcoxon p
    (Holm)", as for one pair) and, when *holm_across*, over all pairs ×
    metrics (HOLM_ALL)."""
    if tags is not None:
        tags = list(tags)
        layout = SegmentLayout(tags, [{p: t for p, t in enumerate(tags)
                                       if pd.notna(df.iloc[i].get(f"start (s) {t}"))}
                                      for i in range(len(df))], [], [[] for _ in range(len(df))])
    else:
        layout = segment_layout(df)
    pairs, odd = consecutive_pairs(layout.tags)
    notes = list(layout.notes)
    if odd is not None:
        notes.append(f"segment {odd} has no partner (odd number of segments): not tested")
    for i, rest in enumerate(layout.extra):
        if rest:
            r = df.iloc[i]
            who = subject_text(r.get("subject")) or subject_text(r.get("file")) or f"row {i + 1}"
            notes.append(f"{who}: segment(s) " + ", ".join(repr(t) for t in rest)
                         + " fit no segment of the group: not tested")
    out = []
    for j, (a, b) in enumerate(pairs):
        places = (2 * j, 2 * j + 1)
        work = _placed_frame(df, layout, places)
        checks = check_rows(work, a, b)
        lprobs, lnotes = label_problems(work, (a, b))
        pnotes = []
        for i, (c, extra) in enumerate(zip(checks, lprobs)):
            renamed = []
            for p in places:
                own, grp = layout.rows[i].get(p), layout.tags[p]
                if own is None or own == grp:
                    continue
                if _norm_label(own) == _norm_label(grp):
                    pnotes.append(f"{c.subject or c.file}: '{own}' taken as '{grp}' "
                                  "(case or spacing)")
                else:
                    renamed.append(grp)
                    c.problems.append(f"segment {p + 1} is '{own}', not '{grp}' like the group")
            # The name check says it already: no second line from the labels.
            c.problems.extend(x for x in extra
                              if not any(x.startswith(f"{g} is '") for g in renamed))
        missing = [any(p.startswith("no segment") for p in c.problems) for c in checks]
        keep = [(c.ok or not exclude) and not m for c, m in zip(checks, missing)]
        used = work.loc[keep].reset_index(drop=True)
        stats = paired_stats(used, a, b, metrics) if len(used) else pd.DataFrame()
        left_out = [c.subject or c.file for c, k, m in zip(checks, keep, missing)
                    if not k and not m]
        out.append(PairResult(a, b, segment_labels(work, a)[0], segment_labels(work, b)[0],
                              used, stats, checks, keep, left_out, lnotes + pnotes))
    if holm_across:
        flat = [(k, i) for k, pr in enumerate(out) for i in range(len(pr.stats))]
        adj = holm([float(out[k].stats["Wilcoxon p"].iloc[i]) for k, i in flat])
        for pr in out:
            if len(pr.stats):
                pr.stats[HOLM_ALL] = math.nan
        for (k, i), v in zip(flat, adj):
            out[k].stats.iloc[i, out[k].stats.columns.get_loc(HOLM_ALL)] = v
    return AllPairsResult(out, odd, holm_across, notes)


def all_pairs_summary(res: AllPairsResult) -> "pd.DataFrame":
    """One row per pair × metric tested: n, median Δ (B − A), Wilcoxon p and
    Holm within the pair (and across all pairs when asked)."""
    cols = ["Pair", "A", "B", "Comparison", "Metric", "n", "Median Δ (B − A)", "Wilcoxon p",
            "Wilcoxon p (Holm)"] + ([HOLM_ALL] if res.holm_across else [])
    rows = []
    for k, pr in enumerate(res.pairs, 1):
        for _i, r in pr.stats.iterrows():
            rows.append([k, pr.a, pr.b, pr.title, r["Metric"], int(r["n"]),
                         r["Median Δ (B − A)"], r["Wilcoxon p"], r["Wilcoxon p (Holm)"]]
                        + ([r.get(HOLM_ALL, math.nan)] if res.holm_across else []))
    return pd.DataFrame(rows, columns=cols)


def _sheet_name(k: int, pr: PairResult, taken: "set[str]") -> str:
    """'1 B vs A' made a valid, unique Excel sheet name (31 characters at
    most; 'Pair 1' when the tags are too long, rather than a label cut)."""
    bad = set("[]:*?/\\")
    base = "".join("-" if ch in bad else ch for ch in f"{k} {pr.b} vs {pr.a}")
    if len(base) > 31:
        base = f"Pair {k}"
    name, i = base, 2
    while name.lower() in taken or name.lower() == "summary":
        suffix = f" ({i})"
        name, i = base[: 31 - len(suffix)] + suffix, i + 1
    taken.add(name.lower())
    return name


def write_all_pairs_report(path: str, res: AllPairsResult, note: str = "") -> "list[str]":
    """Excel report of an all-pairs run: a Summary sheet (pairs × metrics:
    n, median Δ, Wilcoxon p, Holm, then an About block), then one sheet per
    pair holding the four tables of the single-pair report (Statistics,
    About, Rows checked, Paired values) one below the other. Returns the
    sheet names of the pairs."""
    summary = all_pairs_summary(res)
    about = [("Pairs", "; ".join(f"{k}: {pr.title}" for k, pr in enumerate(res.pairs, 1))
              or "none"),
             ("Test", "per pair and metric: two-sided Wilcoxon signed-rank across animals "
                      "(exact for small n); paired t on the pair sheets"),
             ("Holm", "within each pair, over its metrics"
                      + (f"; and across all pairs × metrics (column '{HOLM_ALL}')"
                         if res.holm_across else "")),
             ("Labels", "each segment must carry the group's label in every row; a row "
                        "whose label differs is a row to check")]
    about += [("Note", n) for n in res.notes]
    for pr in res.pairs:
        about += [("Note", f"{pr.b} vs {pr.a}: {n}") for n in pr.notes]
    if note:
        about.append(("Note", note))
    names = []
    with replacing(path) as tmp, \
            pd.ExcelWriter(tmp, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        pd.DataFrame(about, columns=["Item", "Value"]).to_excel(
            xw, sheet_name="Summary", index=False, startrow=len(summary) + 3)
        xw.book["Summary"].freeze_panes = "A2"
        taken: "set[str]" = set()
        for k, pr in enumerate(res.pairs, 1):
            name = _sheet_name(k, pr, taken)
            names.append(name)
            frames = _report_frames(pr.used, pr.a, pr.b, pr.stats, pr.checks, pr.left_out,
                                    "; ".join(x for x in (*pr.notes, note) if x))
            row = 0
            for title, frame in frames.items():
                pd.DataFrame([[title]]).to_excel(xw, sheet_name=name, index=False,
                                                 header=False, startrow=row)
                frame.to_excel(xw, sheet_name=name, index=False, startrow=row + 1)
                row += len(frame) + 4
        _fit_columns(xw)
    return names
