# -*- coding: utf-8 -*-
"""
ecg.io.group_data
-----------------
The group table as a whole experiment: one workbook, one animal per
recording, read by the Group tab.

Sheets (each keyed by subject and file; adding a recording again replaces
its lines in every sheet it writes, never in the others):

* ``Animals``: subject, sex, group (free text, set in the Group tab), use
  (left out of every analysis when false) and file -- the list of animals.
* ``Group``: the Compare segments row (ecg.io.group_table), unchanged.
* ``Whole``: the whole-recording metrics, one column per metric.
* ``TimeCourse``: the valid RR intervals summed in TC_BIN_S bins (count,
  sums and sums of squares of RR, HR and successive differences): enough to
  recompute the HR (60000 / mean RR), mean RR, SDNN and RMSSD of any coarser
  bin exactly, so the bin size and the time 0 stay free in the Group tab
  without the RR series of every animal in the workbook. The fine bins are laid out once
  per anchor (``anchor_s``: the recording start, 0, over the whole
  recording, and the start of each named annotation, over TC_ANCHOR_SPAN_S
  around it), as anchor + k * TC_BIN_S, so a bin edge that is a multiple
  of TC_BIN_S from the chosen time 0 is exact. Anchors TC_BIN_S apart (or a
  multiple of it) share the same fine bins and are stored once. A table
  made before ``anchor_s`` has the recording-start bins only.
* ``Annotations``: the recording's annotations (the time 0 of a time course).
* ``Events``: the event-locked results of the recording (one line per
  event: pre, post and Δ of each metric), from the event registry.

A table made before these sheets (only ``Group``) still opens: its animals
come from the ``Group`` sheet.

Nothing here may import from ecg.ui.
"""
from __future__ import annotations

import math
import os
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ecg.io.safe_write import replacing

from ecg.io.group_stats import sex_text, subject_text
from ecg.io.group_table import match_existing_tags
from ecg.io.subject_name import normalise_sex

SHEET_ANIMALS = "Animals"
SHEET_SEGMENTS = "Group"
SHEET_WHOLE = "Whole"
SHEET_TIME = "TimeCourse"
SHEET_ANNOT = "Annotations"
SHEET_EVENTS = "Events"
# The order of the sheets in the workbook.
SHEETS = (SHEET_ANIMALS, SHEET_SEGMENTS, SHEET_WHOLE, SHEET_TIME, SHEET_ANNOT, SHEET_EVENTS)
KEY = ("subject", "file")
EXCEL_MAX_ROWS = 1_048_576          # rows of an Excel sheet, header included

# Fine bin of the TimeCourse sheet (s): a coarse bin of the Group tab takes
# the fine bins whose centre falls in it -- exact when the fine bins are
# anchored on the time 0 (anchor_s), within TC_BIN_S / 2 otherwise.
TC_BIN_S = 5.0
TC_COLUMNS = ("anchor_s", "t_s", "n_rr", "sum_rr", "sumsq_rr", "sum_hr", "sumsq_hr",
              "n_diff", "sumsq_diff")
# Two anchors whose offset is a multiple of TC_BIN_S to within this (s) share
# their fine bins; a time 0 matches an anchor to within it too.
ANCHOR_TOL_S = 1e-3
# The span (s, from the anchor) over which an annotation's own fine bins are
# laid out: a time course rarely goes further, and a grid over a whole
# day-long recording per annotation would pass Excel's row limit. Past it the
# recording-start grid is used (within TC_BIN_S / 2, with a note).
TC_ANCHOR_SPAN_S = (-2 * 3600.0, 4 * 3600.0)

# Time-course metrics: name -> unit.
TC_METRICS = {"HR": "bpm", "RR": "ms", "SDNN": "ms", "RMSSD": "ms"}


# ── reading ──────────────────────────────────────────────────────────────

def _key_text(df: "pd.DataFrame") -> "pd.DataFrame":
    """Subject and file as text (a numeric subject read as 243.0 is '243')."""
    df = df.copy()
    if "subject" in df.columns:
        df["subject"] = [subject_text(v) for v in df["subject"]]
    if "file" in df.columns:
        df["file"] = ["" if (isinstance(v, float) and not math.isfinite(v)) else str(v)
                      for v in df["file"]]
    return df


def read_sheets(path: str, only: "Optional[Sequence[str]]" = None) -> "dict[str, pd.DataFrame]":
    """Every sheet of the group table at *path*: the known ones (SHEETS) with
    subject and file as text, and any other sheet the user added (notes, a
    Prism import...) as it is, written back by write_sheets. A workbook
    without a subject / file column in a known sheet is refused. *only*:
    just those of the sheets (the ones the workbook has), for a reader that
    needs a few -- the TimeCourse sheet of a big table takes most of the
    time to read; never written back."""
    if only is None:
        xl = pd.read_excel(path, sheet_name=None)
    else:
        with pd.ExcelFile(path) as book:
            xl = {name: book.parse(name) for name in book.sheet_names if name in only}
    out = {name: _key_text(df) for name, df in xl.items() if name in SHEETS}
    if only is not None and not out:
        return {}                   # none of those sheets: the caller says what is missing
    if not out or not any(all(c in df.columns for c in KEY) for df in out.values()):
        raise ValueError(f"{os.path.basename(path)} is not a group table "
                         "(no subject / file columns)")
    for name, df in out.items():
        for c in KEY:
            if c not in df.columns:
                raise ValueError(f"{os.path.basename(path)}: sheet {name} has no '{c}' column")
    out.update({name: df for name, df in xl.items() if name not in SHEETS})
    return out


def _bool(v: Any, default: bool = True) -> bool:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return default
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("", "nan"):
            return default
        return t not in ("0", "false", "no", "non", "n")
    return bool(v)


def animals_frame(sheets: "Mapping[str, pd.DataFrame]") -> "pd.DataFrame":
    """The animals of the table: subject, sex, group, use, file, and which
    data each holds (``has_segments``, ``has_whole``, ``has_time``,
    ``has_events``: number of events). From the Animals sheet, completed by
    the recordings of the other sheets (an older table has only Group)."""
    rows: "dict[tuple[str, str], dict]" = {}

    def _row(s: str, f: str) -> dict:
        return rows.setdefault((s, f), {"subject": s, "sex": "", "group": "", "use": True,
                                        "file": f})

    an = sheets.get(SHEET_ANIMALS)
    if an is not None:
        for _, r in an.iterrows():
            d = _row(r["subject"], r["file"])
            d["sex"] = sex_text(r.get("sex", ""))
            g = r.get("group", "")
            d["group"] = "" if g is None or (isinstance(g, float) and not math.isfinite(g)) \
                else subject_text(g)
            d["use"] = _bool(r.get("use", True))
    for name in (SHEET_SEGMENTS, SHEET_WHOLE, SHEET_TIME, SHEET_EVENTS):
        df = sheets.get(name)
        if df is None:
            continue
        for (s, f), sub in df.groupby(list(KEY), sort=False):
            d = _row(s, f)
            if not d["sex"] and "sex" in sub.columns:
                d["sex"] = sex_text(sub["sex"].iloc[0])
            if not d["group"] and "group" in sub.columns:
                d["group"] = subject_text(sub["group"].iloc[0])
    out = pd.DataFrame(list(rows.values()),
                       columns=["subject", "sex", "group", "use", "file"])
    if out.empty:
        return out.assign(has_segments=pd.Series(dtype=bool), has_whole=pd.Series(dtype=bool),
                          has_time=pd.Series(dtype=bool), has_events=pd.Series(dtype=int))

    def _has(name: str) -> "list[bool]":
        df = sheets.get(name)
        keys = set() if df is None else set(zip(df["subject"], df["file"]))
        return [(s, f) in keys for s, f in zip(out["subject"], out["file"])]

    out["has_segments"] = _has(SHEET_SEGMENTS)
    out["has_whole"] = _has(SHEET_WHOLE)
    out["has_time"] = _has(SHEET_TIME)
    ev = sheets.get(SHEET_EVENTS)
    counts = ({} if ev is None else
              ev.groupby(list(KEY), sort=False).size().to_dict())
    out["has_events"] = [int(counts.get((s, f), 0)) for s, f in zip(out["subject"], out["file"])]
    return out.sort_values(["subject", "file"], kind="stable").reset_index(drop=True)


def data_text(row: "Mapping[str, Any]") -> str:
    """'seg · whole · time · 3 events' for an animal of animals_frame()."""
    parts = [name for name, k in (("seg", "has_segments"), ("whole", "has_whole"),
                                  ("time", "has_time")) if row.get(k)]
    n = int(row.get("has_events") or 0)
    if n:
        parts.append(f"{n} event{'s' if n != 1 else ''}")
    return " · ".join(parts) or "no data"


# ── writing ──────────────────────────────────────────────────────────────

def _ordered(df: "pd.DataFrame") -> "pd.DataFrame":
    """Subject, sex, group first (when present), then file, then the rest."""
    first = [c for c in ("subject", "sex", "group", "use", "file") if c in df.columns]
    return df[first + [c for c in df.columns if c not in first]]


def write_sheets(path: str, sheets: "Mapping[str, pd.DataFrame]") -> None:
    """Write *sheets* (every sheet of the table) to *path*: the known ones in
    SHEETS order, then the user's own sheets as they were. Refused, with
    nothing written, when a sheet is past Excel's row limit (a TimeCourse of
    many day-long recordings)."""
    for name, df in sheets.items():
        if df is not None and len(df) >= EXCEL_MAX_ROWS:
            raise ValueError(
                f"The {name} sheet would have {len(df):,} rows, past Excel's limit "
                f"({EXCEL_MAX_ROWS - 1:,}): split the experiment into several group tables.")
    with replacing(path, backup=True) as tmp, \
            pd.ExcelWriter(tmp, engine="openpyxl") as xw:
        for name in SHEETS:
            df = sheets.get(name)
            if df is None:
                continue
            df = _ordered(df)
            sort = [c for c in ("subject", "file", "anchor_s", "t_s", "onset_s", "t_start")
                    if c in df.columns]
            df = df.sort_values(sort, kind="stable").reset_index(drop=True)
            df.to_excel(xw, sheet_name=name, index=False)
        for name, df in sheets.items():
            if name not in SHEETS and df is not None:
                df.to_excel(xw, sheet_name=name, index=False)


def replace_recording(sheets: "dict[str, pd.DataFrame]", subject: str, file: str,
                      new: "Mapping[str, Optional[pd.DataFrame]]") -> bool:
    """Put the lines of *new* ({sheet: frame of this recording}) in *sheets*
    in place of that recording's lines (subject and file); a sheet given as
    None or an empty frame only loses them (the recording has no such data
    any more). Sheets not in *new* are kept. Returns whether the recording
    was in the table already."""
    subject, file = subject_text(subject), str(file)
    was = False
    for name, old in sheets.items():
        if name in SHEETS and old is not None and len(old):
            was = was or bool(((old["subject"].astype(str) == subject)
                               & (old["file"].astype(str) == file)).any())
    for name, add in new.items():
        old = sheets.get(name)
        if old is not None and len(old):
            old = old.loc[~((old["subject"].astype(str) == subject)
                            & (old["file"].astype(str) == file))]
        if add is not None and len(add):
            add = add.copy()
            add["subject"] = subject
            add["file"] = file
            old = add if old is None or not len(old) else pd.concat([old, add], ignore_index=True)
        if old is not None:
            sheets[name] = old
    return was


def _add_to_sheets(sheets: "dict[str, pd.DataFrame]", subject: str, sex: str, file: str,
                   new: "Mapping[str, Optional[pd.DataFrame]]") -> bool:
    """add_recording on *sheets* in memory; returns whether it was there."""
    subject = subject_text(subject)
    file = os.path.basename(file or "")
    an = sheets.get(SHEET_ANIMALS)
    group, use = "", True
    sex = normalise_sex(sex)
    if an is not None and len(an):
        same = (an["subject"] == subject) & (an["file"] == file)
        if same.any():
            r = an.loc[same].iloc[0]
            group = subject_text(r.get("group", ""))
            use = _bool(r.get("use", True))
            # A sex set in the Group tab stays when none is given this time.
            sex = sex or sex_text(r.get("sex", ""))
    seg = new.get(SHEET_SEGMENTS)
    old_seg = sheets.get(SHEET_SEGMENTS)
    if seg is not None and len(seg) and old_seg is not None and len(old_seg):
        # "pre stim 5min" goes under the table's "Pre Stim 5min" column.
        new = dict(new)
        new[SHEET_SEGMENTS] = pd.DataFrame([match_existing_tags(seg.iloc[0].to_dict(),
                                                                list(old_seg.columns))])
    # An older table: the animals of its other sheets enter the Animals sheet.
    animals = animals_frame(sheets) if sheets else pd.DataFrame(columns=list(KEY))
    replaced = replace_recording(sheets, subject, file, new)
    row = pd.DataFrame([{"subject": subject, "sex": sex, "group": group,
                         "use": use, "file": file}])
    base = (animals[["subject", "sex", "group", "use", "file"]] if len(animals)
            else pd.DataFrame(columns=["subject", "sex", "group", "use", "file"]))
    base = base.loc[~((base["subject"] == subject) & (base["file"] == file))]
    sheets[SHEET_ANIMALS] = pd.concat([base, row], ignore_index=True) if len(base) else row
    return replaced


def add_to_sheets(sheets: "dict[str, pd.DataFrame]", subject: str, sex: str, file: str,
                  new: "Mapping[str, Optional[pd.DataFrame]]") -> bool:
    """add_recording on a table held in memory (*sheets*, changed in place),
    for a group made from saved sessions before it is saved as Excel.
    Returns whether the recording was there already."""
    return _add_to_sheets(sheets, subject, sex, file, new)


def remove_recording(sheets: "dict[str, pd.DataFrame]", subject: str, file: str) -> bool:
    """Take a recording (subject and file) out of every sheet of the table
    (the Animals list included), in place. Returns whether it was there."""
    subject, file = subject_text(subject), str(file)
    was = False
    for name in SHEETS:
        df = sheets.get(name)
        if df is None or not len(df):
            continue
        same = (df["subject"].astype(str) == subject) & (df["file"].astype(str) == file)
        if same.any():
            was = True
            sheets[name] = df.loc[~same].reset_index(drop=True)
    return was


def fit_time_course(sheets: "dict[str, pd.DataFrame]") -> "list[tuple[str, str]]":
    """Keep the TimeCourse sheet of *sheets* under Excel's row limit by
    dropping the annotation grids (anchor_s off the recording-start grid) of
    the recordings that have the most of them, until it fits; those
    recordings are then aligned on the recording-start grid (within
    TC_BIN_S / 2, the Group tab says so) rather than the whole add being
    refused. Returns the (subject, file) whose grids were dropped."""
    tc = sheets.get(SHEET_TIME)
    if tc is None or len(tc) < EXCEL_MAX_ROWS or "anchor_s" not in tc.columns:
        return []
    a = pd.to_numeric(tc["anchor_s"], errors="coerce").fillna(0.0).to_numpy(float)
    k = a / TC_BIN_S
    extra = np.abs(k - np.round(k)) * TC_BIN_S >= ANCHOR_TOL_S     # not _same_grid(a, 0)
    key = tc["subject"].astype(str) + "\x00" + tc["file"].astype(str)
    counts = key[extra].value_counts()
    drop = np.zeros(len(tc), dtype=bool)
    rows, dropped = len(tc), []
    for k, n in counts.items():
        if rows < EXCEL_MAX_ROWS:
            break
        drop |= extra & (key == k).to_numpy()
        rows -= int(n)
        dropped.append(tuple(k.split("\x00", 1)))
    sheets[SHEET_TIME] = tc.loc[~drop].reset_index(drop=True)
    return dropped


def add_recording(path: str, subject: str, sex: str, file: str,
                  new: "Mapping[str, Optional[pd.DataFrame]]") -> "tuple[int, bool]":
    """Add one recording to the group table at *path* (created if it does not
    exist): its lines in each sheet of *new* replace the older ones (a sheet
    given as None loses them), and the Animals sheet gets it (its group and
    use kept when it was there, and its sex when *sex* is empty). Returns
    (animals in the table, whether the recording was replaced)."""
    sheets = read_sheets(path) if os.path.exists(path) else {}
    replaced = _add_to_sheets(sheets, subject, sex, file, new)
    fit_time_course(sheets)
    write_sheets(path, sheets)
    return len(sheets[SHEET_ANIMALS]), replaced


def add_recordings(path: str, items: "Iterable[tuple[str, str, str, Mapping]]") -> "tuple[int, int]":
    """add_recording for each (subject, sex, file, lines) of *items*, the
    workbook read and written once. Returns (animals in the table, how many
    were there already)."""
    sheets = read_sheets(path) if os.path.exists(path) else {}
    replaced = sum(_add_to_sheets(sheets, s, x, f, new) for s, x, f, new in items)
    if SHEET_ANIMALS in sheets:
        fit_time_course(sheets)
        write_sheets(path, sheets)
    return len(sheets.get(SHEET_ANIMALS, ())), int(replaced)


def save_animals(path: str, animals: "pd.DataFrame") -> None:
    """Write the sex, group and use of *animals* (subject, file, sex, group,
    use) to the Animals sheet of *path*; every other sheet is kept."""
    sheets = read_sheets(path)
    cols = ["subject", "sex", "group", "use", "file"]
    sheets[SHEET_ANIMALS] = animals[cols].copy()
    write_sheets(path, sheets)


# ── building a recording's lines ─────────────────────────────────────────

def whole_frame(metric_rows: "Iterable[Any]") -> "pd.DataFrame":
    """One line: a column per metric ("label (unit)": value) of the rows of
    a core.metric_table.MetricTable (whole recording or analysed window)."""
    row: "dict[str, Any]" = {}
    for r in metric_rows:
        v = getattr(r, "value", None)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        label = str(getattr(r, "label", "") or getattr(r, "key", ""))
        unit = str(getattr(r, "unit", "") or "")
        name = f"{label} ({unit})" if unit else label
        if name not in row:
            row[name] = v
    return pd.DataFrame([row]) if row else pd.DataFrame()


def _same_grid(a: float, b: float, bin_s: float = TC_BIN_S) -> bool:
    """Whether the bin grids anchored on *a* and *b* (s) coincide: their offset
    is a multiple of *bin_s* to within ANCHOR_TOL_S."""
    k = (a - b) / bin_s
    return abs(k - round(k)) * bin_s < ANCHOR_TOL_S


def _anchor_grids(anchors: "Iterable[Any]", bin_s: float = TC_BIN_S
                  ) -> "list[tuple[float, float, float]]":
    """(grid anchor, from, to) s of each distinct bin grid of *anchors*: the
    recording start (0, everything) first, then each finite anchor whose grid
    no earlier one has, over TC_ANCHOR_SPAN_S around every anchor that shares
    it."""
    lo_s, hi_s = TC_ANCHOR_SPAN_S
    out = [[0.0, -math.inf, math.inf]]
    for a in (() if anchors is None else anchors):
        try:
            a = float(a)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(a):
            continue
        g = next((g for g in out if _same_grid(a, g[0], bin_s)), None)
        if g is None:
            out.append([a, a + lo_s, a + hi_s])
        else:
            g[1], g[2] = min(g[1], a + lo_s), max(g[2], a + hi_s)
    return [tuple(g) for g in out]


def tc_anchors(anchors: "Iterable[Any]", bin_s: float = TC_BIN_S) -> "list[float]":
    """The distinct bin grids of *anchors* (s): 0 (the recording start) first,
    then each finite anchor whose grid no earlier one has."""
    return [g[0] for g in _anchor_grids(anchors, bin_s)]


def _time_frame_one(t: np.ndarray, rr: np.ndarray, ok: np.ndarray, bin_s: float,
                    anchor: float, span: "tuple[float, float]" = (-math.inf, math.inf)
                    ) -> "pd.DataFrame":
    b = np.floor((t - anchor) / bin_s).astype(np.int64)
    lo = int(b[ok].min())
    nb = int(b[ok].max()) - lo + 1
    idx = b - lo
    hr = np.where(ok, 60_000.0 / np.where(rr > 0, rr, np.nan), 0.0)
    w = ok.astype(float)
    rrv = np.where(ok, rr, 0.0)

    def _sum(x: np.ndarray, m: np.ndarray) -> np.ndarray:
        return np.bincount(idx[m], weights=x[m], minlength=nb)[:nb]

    out = {"anchor_s": np.full(nb, anchor),
           "t_s": anchor + (np.arange(nb) + lo) * bin_s,
           "n_rr": _sum(w, ok), "sum_rr": _sum(rrv, ok), "sumsq_rr": _sum(rrv ** 2, ok),
           "sum_hr": _sum(hr, ok), "sumsq_hr": _sum(hr ** 2, ok)}
    pair = ok[1:] & ok[:-1]
    d = np.where(pair, rr[1:] - rr[:-1], 0.0)
    pidx = idx[1:]
    sel = pair & (pidx >= 0) & (pidx < nb)
    out["n_diff"] = np.bincount(pidx[sel], minlength=nb)[:nb].astype(float)
    out["sumsq_diff"] = np.bincount(pidx[sel], weights=(d ** 2)[sel], minlength=nb)[:nb]
    df = pd.DataFrame(out)
    # The span is cut on whole bins, after the sums, so its edge bins keep
    # the successive differences that reach back past it.
    return df.loc[(df["n_rr"] > 0) & (df["t_s"] >= span[0] - ANCHOR_TOL_S)
                  & (df["t_s"] + bin_s <= span[1] + ANCHOR_TOL_S)]


def time_frame(t_s: "Sequence[float]", rr_ms: "Sequence[float]", valid: "Sequence[bool]",
               bin_s: float = TC_BIN_S, anchors: "Iterable[Any]" = ()) -> "pd.DataFrame":
    """The TimeCourse lines of one recording: per *bin_s* bin (``t_s``: its
    start, s from the recording start) the count, sum and sum of squares of
    the valid RR (ms) and of their HR (bpm), and of the successive
    differences between two valid intervals (``n_diff``, ``sumsq_diff``),
    each counted in the bin of the interval's end time. The bins are laid
    out from the recording start (``anchor_s`` 0) and again, over
    TC_ANCHOR_SPAN_S around it, from each of *anchors* (the annotation
    starts, s) with a grid of its own (tc_anchors), so a time course aligned
    on any of them has exact bin edges."""
    t = np.asarray(t_s, dtype=float)
    rr = np.asarray(rr_ms, dtype=float)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(rr) & np.isfinite(t) & (rr > 0)
    if not ok.any():
        return pd.DataFrame(columns=list(TC_COLUMNS))
    df = pd.concat([_time_frame_one(t, rr, ok, bin_s, a, (lo, hi))
                    for a, lo, hi in _anchor_grids(anchors, bin_s)], ignore_index=True)
    df["n_rr"] = df["n_rr"].astype(int)
    df["n_diff"] = df["n_diff"].astype(int)
    return df


def annotations_frame(annotations: "Iterable[Mapping[str, Any]]") -> "pd.DataFrame":
    """label, t_start, t_end (s) of each annotation with a start time."""
    rows = []
    for a in annotations or []:
        try:
            t0 = float(a.get("t_start"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(t0):
            continue
        try:
            t1 = float(a.get("t_end"))
        except (TypeError, ValueError):
            t1 = math.nan
        rows.append({"label": str(a.get("label", "") or "").strip(),
                     "t_start": t0, "t_end": t1})
    return pd.DataFrame(rows, columns=["label", "t_start", "t_end"])


def events_frame(event_rows: "Iterable[Mapping[str, Any]]") -> "pd.DataFrame":
    """One line per event of ``ecg.io.db.list_event_rows`` (one recording):
    label, onset_s, pre_s, post_s, method, window_s, valid (the event's, set
    by the CSI family), then per metric ``<metric> pre``, ``<metric> post``,
    ``<metric> Δ`` and ``<metric> valid`` (that series' own validity: an
    event can be unusable for the CSI and fine for HR)."""
    rows = []
    for e in event_rows or []:
        row: "dict[str, Any]" = {
            "label": str(e.get("label", "") or ""), "onset_s": e.get("onset_s"),
            "pre_s": e.get("pre_s"), "post_s": e.get("post_s"),
            "method": str(e.get("method", "") or ""), "window_s": e.get("window_s"),
            "valid": bool(e.get("valid", True))}
        for k, m in (e.get("metrics") or {}).items():
            if not isinstance(m, Mapping):
                continue
            for part, suffix in (("pre", "pre"), ("post", "post"), ("delta", "Δ")):
                v = m.get(part)
                row[f"{k} {suffix}"] = float(v) if isinstance(v, (int, float)) and \
                    math.isfinite(float(v)) else math.nan
            row[f"{k} valid"] = bool(m.get("valid", row["valid"]))
        rows.append(row)
    return pd.DataFrame(rows)


# ── time course: coarse bins ─────────────────────────────────────────────

def stim_onsets(annot: "Optional[pd.DataFrame]") -> "dict[tuple[str, str], list[tuple[str, float]]]":
    """{(subject, file): [(label, t_start), ...]} of the Annotations sheet."""
    out: "dict[tuple[str, str], list[tuple[str, float]]]" = {}
    if annot is None or annot.empty:
        return out
    for _, r in annot.iterrows():
        try:
            t = float(r["t_start"])
        except (TypeError, ValueError):
            continue
        if math.isfinite(t):
            lab = "" if (isinstance(r["label"], float) and not math.isfinite(r["label"])) \
                else str(r["label"]).strip()
            out.setdefault((r["subject"], r["file"]), []).append((lab, t))
    for v in out.values():
        v.sort(key=lambda x: x[1])
    return out


def annotation_labels(annot: "Optional[pd.DataFrame]") -> "list[str]":
    """The annotation labels of the table, most frequent first."""
    if annot is None or annot.empty:
        return []
    labs = [("" if (isinstance(v, float) and not math.isfinite(v)) else " ".join(str(v).split()))
            for v in annot["label"]]
    counts: "dict[str, int]" = {}
    shown: "dict[str, str]" = {}          # the first spelling of each label
    for lab in labs:
        if lab:
            k = lab.casefold()
            counts[k] = counts.get(k, 0) + 1
            shown.setdefault(k, lab)
    return [shown[k] for k in sorted(counts, key=lambda k: (-counts[k], k))]


def onsets_named(onsets: "Sequence[tuple[str, float]]", label: str) -> "list[float]":
    """The starts, in order, of the annotations named *label* (case and
    spacing insensitive)."""
    norm = " ".join(label.split()).casefold()
    return [t for lab, t in onsets if " ".join(lab.split()).casefold() == norm]


def onset_of(onsets: "Sequence[tuple[str, float]]", label: str, which: int = 0
             ) -> "Optional[float]":
    """The start of the *which*-th (0: first) annotation named *label* (case
    and spacing insensitive), None when the recording has none."""
    hits = onsets_named(onsets, label)
    return hits[which] if 0 <= which < len(hits) else None


def _metric_from_sums(metric: str, n: float, s_rr: float, ss_rr: float, s_hr: float,
                      n_d: float, ss_d: float) -> float:
    if metric == "HR":
        # 60000 / mean RR, as the rest of the app (the Whole sheet's mean HR),
        # not the mean of the beat-by-beat HR (which sits a little higher).
        return 60000.0 * n / s_rr if n > 0 and s_rr > 0 else math.nan
    if metric == "RR":
        return s_rr / n if n > 0 else math.nan
    if metric == "SDNN":
        if n < 2:
            return math.nan
        var = (ss_rr - s_rr * s_rr / n) / (n - 1)
        return math.sqrt(max(var, 0.0))
    if metric == "RMSSD":
        return math.sqrt(ss_d / n_d) if n_d > 0 else math.nan
    raise ValueError(f"unknown time-course metric {metric!r}")


# How anchor_lines met the time 0: on a grid of its own (exact), on the
# recording-start grid because the table has none for it, or because the
# time course goes past the span its own grid covers.
ALIGN_EXACT, ALIGN_NO_GRID, ALIGN_PAST_SPAN = "exact", "no grid", "past span"


def anchor_lines(tc: "pd.DataFrame", t0_s: float,
                 span: "Optional[tuple[float, float]]" = None
                 ) -> "tuple[pd.DataFrame, str]":
    """(lines, how): the TimeCourse lines of one recording laid out on the
    grid of its time 0 *t0_s* (s), and ALIGN_EXACT. When it has no such grid
    (a table made before anchor_s, or an annotation added after the time
    course: ALIGN_NO_GRID), or that grid stops short of the part of *span*
    (from, to s relative to the time 0) the recording covers
    (ALIGN_PAST_SPAN), the recording-start lines -- their bins then meet the
    time 0 within TC_BIN_S / 2."""
    if "anchor_s" in tc.columns:
        a = pd.to_numeric(tc["anchor_s"], errors="coerce").fillna(0.0).to_numpy(float)
    else:
        a = np.zeros(len(tc))
    grids = list(dict.fromkeys(a.tolist()))
    start = next((g for g in grids if _same_grid(0.0, g)), grids[0] if grids else 0.0)
    whole = tc.loc[a == start]
    g = next((g for g in grids if _same_grid(t0_s, g)), None)
    if g is None:
        return whole, ALIGN_NO_GRID
    own = tc.loc[a == g]
    if g == start or span is None or not len(whole) or not len(own):
        return own, ALIGN_EXACT
    # The part of the span with beats (from the whole-recording grid) must
    # lie in the own grid's lines; a fine bin of slack, the grids being up
    # to one apart.
    t_all = whole["t_s"].to_numpy(float)
    need0 = max(t0_s + span[0], float(t_all.min()))
    need1 = min(t0_s + span[1], float(t_all.max()) + TC_BIN_S)
    if need1 <= need0:
        return own, ALIGN_EXACT
    t_own = own["t_s"].to_numpy(float)
    if need0 < float(t_own.min()) - TC_BIN_S or need1 > float(t_own.max()) + 2 * TC_BIN_S:
        return whole, ALIGN_PAST_SPAN
    return own, ALIGN_EXACT


def window_value(tc: "pd.DataFrame", metric: str, t0: float, t1: float,
                 min_coverage: float = 0.5) -> float:
    """*metric* over the fine bins of *tc* (the lines of one recording on one
    grid, see anchor_lines, times relative to its time 0) whose centre lies
    in [t0, t1) s; NaN when the valid RR cover less than *min_coverage* of
    the window."""
    c = tc["t_s"].to_numpy(float) + TC_BIN_S / 2
    m = (c >= t0) & (c < t1)
    if not m.any():
        return math.nan
    s = tc.loc[m]
    n = float(s["n_rr"].sum())
    covered = float(s["sum_rr"].sum()) / 1000.0
    if covered < min_coverage * (t1 - t0):
        return math.nan
    return _metric_from_sums(metric, n, float(s["sum_rr"].sum()), float(s["sumsq_rr"].sum()),
                             float(s["sum_hr"].sum()), float(s["n_diff"].sum()),
                             float(s["sumsq_diff"].sum()))


def bin_edges(start_s: float, end_s: float, bin_s: float) -> np.ndarray:
    """The edges of *bin_s* bins anchored on the time 0 (every k * bin_s)
    covering [start_s, end_s] s; the first and last bins are cut at
    *start_s* and *end_s* when these are not on the grid."""
    if not (bin_s > 0) or not (end_s > start_s):
        raise ValueError("bins: the size must be > 0 and the end after the start")
    # The small slack keeps -600 / 60 from becoming -10.000000001.
    k0 = math.floor(start_s / bin_s + 1e-9)
    k1 = math.ceil(end_s / bin_s - 1e-9)
    edges = np.arange(k0, k1 + 1, dtype=float) * bin_s
    edges[0], edges[-1] = start_s, end_s
    return edges


def binned_values(rel: "pd.DataFrame", metric: str, edges: "Sequence[float]",
                  min_coverage: float = 0.5) -> np.ndarray:
    """*metric* in each bin [edges[i], edges[i + 1]) of *rel* (one grid's
    lines, times relative to the time 0)."""
    e = np.asarray(edges, dtype=float)
    return np.array([window_value(rel, metric, a, b, min_coverage)
                     for a, b in zip(e[:-1], e[1:])])


def binned_series(tc: "pd.DataFrame", metric: str, t0_s: float, start_s: float, end_s: float,
                  bin_s: float, min_coverage: float = 0.5) -> "tuple[np.ndarray, np.ndarray]":
    """(bin starts in s relative to the time 0, values) of *metric* for one
    recording's TimeCourse lines *tc*, its time 0 at *t0_s* s, bins of
    *bin_s* anchored on the time 0 (bin_edges) from *start_s* to *end_s*.
    A bin covered less than *min_coverage* by valid RR is NaN."""
    edges = bin_edges(start_s, end_s, bin_s)
    lines, _how = anchor_lines(tc, t0_s, (start_s, end_s))
    rel = lines.assign(t_s=lines["t_s"].to_numpy(float) - t0_s)
    return edges[:-1], binned_values(rel, metric, edges, min_coverage)
