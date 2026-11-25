# -*- coding: utf-8 -*-
"""
ecg.app.group_helpers
---------------------
The group table without its widgets: the lines one recording writes to it
(the recording open, or a saved session), the group from a project's
saved sessions, and the Group tab's number fields. Moved unchanged from
the Tk tab_group (its functions, and TabGroupMixin's
_group_lines_of_open_recording / _add_open_recording_to_group_table as
functions of the app).
"""
from __future__ import annotations

import logging
import math
import os
import re
from typing import Any, Optional

import pandas as pd

from ecg.io import group_data as gd
from ecg.io.subject_name import normalise_sex

log = logging.getLogger("ecg")


ALL_EVENTS = "All events"


# The last group table opened, remembered per user (read at call time, so a
# test can point it elsewhere).
FIG_SIZE = (12.0, 4.6)


def _file_tag(analysis: str) -> str:
    """A file-name-safe tag of an analysis name ('Pre/Post segments' ->
    'pre_post'): a '/' in the suggested name made the save fail."""
    return re.sub(r"[^a-z0-9]+", "_", analysis.split()[0].lower()).strip("_") or "group"


def recording_lines(results: "Optional[dict]", metric_rows: "Any", annotations: "Any",
                    event_rows: "Any", comparison: "Any" = None, sex: str = ""
                    ) -> "dict[str, Optional[pd.DataFrame]]":
    """The group-table lines of the recording open ({sheet: frame}): its
    whole-recording metrics and time course (after Analyze), annotations,
    event-locked results (the event registry) and, with *comparison* (a
    Compare segments result), its segments. A kind without data this time
    is left out, so the recording's lines from an earlier add stay (Add to
    group table from Compare segments without Analyze must not drop its
    whole-recording and time-course lines)."""
    from ecg.core.analysis import rr_timeseries
    from ecg.io.group_table import group_row
    out: "dict[str, Optional[pd.DataFrame]]" = {}
    whole = gd.whole_frame(metric_rows or [])
    if len(whole):
        out[gd.SHEET_WHOLE] = whole
    an = gd.annotations_frame(annotations or [])
    ts = rr_timeseries(results) if results else None
    # Fine bins laid out on each named annotation too (an unnamed one cannot
    # be chosen as the Time 0): a time course aligned on any of them has
    # exact bin edges.
    named = an.loc[an["label"].astype(str).str.strip() != "", "t_start"].tolist()
    tc = (gd.time_frame(ts["Time_s"], ts["RR_ms"], ts["valid"], anchors=named)
          if ts is not None and len(ts) else None)
    if tc is not None and len(tc):
        out[gd.SHEET_TIME] = tc
    # With a new time course its annotations are written too, even none: the
    # old ones (deleted since) must not stay as a Time 0 its grids miss.
    if len(an) or gd.SHEET_TIME in out:
        out[gd.SHEET_ANNOT] = an
    ev = gd.events_frame(event_rows or [])
    if len(ev):
        out[gd.SHEET_EVENTS] = ev
    if comparison is not None:
        row = group_row(comparison.subject, comparison.filepath, comparison.segs,
                        comparison.metrics, template=comparison.template,
                        flags=comparison.flags, sex=sex)
        for k in ("subject", "file"):
            row.pop(k, None)
        out[gd.SHEET_SEGMENTS] = pd.DataFrame([row])
    return out


def session_lines(filepath: str) -> "tuple[str, str, dict] | str":
    """(subject, sex, {sheet: lines}) of a recording's saved session: its
    whole-recording metrics and time course (its results), annotations and
    event-locked results (the registry), without reopening the file. A
    reason (text) when the session cannot be used. Segments are not in a
    session (Compare segments → Add to group table for them)."""
    from ecg.core.metric_table import build_metric_table
    from ecg.io import db
    from ecg.io.loaders import _deserialise_results
    from ecg.io.session import load_session
    if not os.path.exists(filepath):
        return "file not found"
    state = load_session(filepath)
    if not state:
        return "no saved session (or the file changed since)"
    raw = state.get("results")
    results = _deserialise_results(raw) if raw else None
    if not results:
        return "session saved before Analyze"
    window = state.get("analysed_window")
    try:
        rows = list(build_metric_table(
            results, artifact_report=state.get("artifact_report"),
            window=tuple(window) if window else None).rows)
    except Exception as exc:              # a module of an older session: the rest still goes
        log.warning("group import: metric table of %s: %s", filepath, exc)
        rows = []
    events = db.list_event_rows(filepath=filepath, valid_only=False)
    fields = state.get("recording_fields") or {}
    subject = str(fields.get("subject") or "").strip() or \
        os.path.splitext(os.path.basename(filepath))[0]
    lines = recording_lines(results, rows, state.get("annotations"), events)
    return subject, normalise_sex(fields.get("sex") or ""), lines


def project_items(project: str) -> "tuple[list, list[str], list[str]]":
    """The group-table lines of every saved recording of *project*: (items
    (subject, sex, file, lines), the subjects found, the recordings left out
    with why)."""
    from ecg.io import db
    items: "list" = []
    added: "list[str]" = []
    skipped: "list[str]" = []
    for rec in db.project_recordings(project):
        fp = str(rec.get("filepath") or "")
        got = session_lines(fp)
        name = os.path.basename(fp)
        if isinstance(got, str):
            skipped.append(f"{name}: {got}")
            continue
        subject, sex, lines = got
        items.append((subject, sex, fp, lines))
        added.append(subject)
    return items, added, skipped


def project_sheets(project: str) -> "tuple[dict, list[str], list[str]]":
    """A group table held in memory, made from the saved sessions of
    *project* (no Excel file): (sheets, subjects, left out)."""
    items, added, skipped = project_items(project)
    sheets: "dict" = {}
    for subject, sex, fp, lines in items:
        gd.add_to_sheets(sheets, subject, sex, fp, lines)
    return sheets, added, skipped


def import_project(path: str, project: str) -> "tuple[int, list[str], list[str]]":
    """Add every saved recording of *project* to the group table at *path*,
    written once. Returns (animals in the table, the recordings added, the
    ones left out with why)."""
    items, added, skipped = project_items(project)
    if not items:
        return 0, added, skipped
    n, _replaced = gd.add_recordings(path, items)
    return n, added, skipped


def _float(entry: "Any", default: float) -> float:
    try:
        v = float(str(entry.get()).replace(",", ".").replace("−", "-").strip())
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _bin_size(entry: "Any", default: float = 60.0) -> float:
    """The bin size (s) typed in *entry* (*default* when empty); a text that
    is not a number, or a size below the table's TC_BIN_S, is refused with a
    message (gc.analyse_time rounds the rest to a multiple of TC_BIN_S)."""
    text = str(entry.get()).replace(",", ".").replace("−", "-").strip()
    if not text:
        return default
    try:
        v = float(text)
    except ValueError:
        raise ValueError(f"Bins (s): “{text}” is not a number. Type a bin size in seconds, "
                         f"a multiple of {gd.TC_BIN_S:g} (e.g. 60).") from None
    if not math.isfinite(v) or v < gd.TC_BIN_S:
        raise ValueError(f"Bins (s): {text} s is not a bin size. It must be at least "
                         f"{gd.TC_BIN_S:g} s, a multiple of {gd.TC_BIN_S:g} (e.g. 60).")
    return v


def group_lines_of_open_recording(app: Any, comparison: "Any" = None
                                   ) -> "tuple[str, str, str, dict]":
    """(subject, sex, file, {sheet: lines}) of the recording open."""
    from ecg.io import db
    sig = getattr(app, "signal", None)
    filepath = str(getattr(sig, "filepath", "") or "")
    ent = getattr(app, "ent_subject", None)
    subject = ent.get().strip() if ent is not None else ""
    ent_sex = getattr(app, "ent_sex", None)
    sex = normalise_sex(ent_sex.get()) if ent_sex is not None else ""
    analysis = getattr(app, "analysis", None)
    results = getattr(analysis, "results", None)
    rows = []
    if results:
        try:
            rows = list(app.export_ctrl.metric_table().rows)
        except Exception as exc:          # a stale module must not block the rest
            log.warning("group table: metric table: %s", exc)
    events = db.list_event_rows(filepath=filepath, valid_only=False) if filepath else []
    if comparison is None:
        cur = getattr(app, "_current_comparison", None)
        comparison = cur() if callable(cur) else None
    lines = recording_lines(results, rows, getattr(analysis, "annotations", None),
                            events, comparison, sex)
    # The subject the comparison was run under keys the row, as the
    # Compare segments group row always did (else the same file could
    # come in twice, under two subjects).
    cmp_subject = str(getattr(comparison, "subject", "") or "").strip()
    subject = cmp_subject or subject
    return subject or os.path.splitext(os.path.basename(filepath))[0], sex, filepath, lines


def add_open_recording_to_group_table(app: Any, path: str,
                                       comparison: "Any" = None,
                                       sheets: "Optional[dict]" = None
                                       ) -> "tuple[int, bool, str]":
    """Write the recording open to the group table at *path* (or, with
    *sheets*, to that table held in memory). Returns (animals in the
    table, replaced, what was written)."""
    subject, sex, filepath, lines = group_lines_of_open_recording(app, comparison)
    if not filepath:
        raise ValueError("Open a recording first.")
    # An empty frame only clears older lines (no annotations any more).
    got = [k for k, v in lines.items() if v is not None and len(v)]
    if not any(k in got for k in (gd.SHEET_SEGMENTS, gd.SHEET_WHOLE, gd.SHEET_TIME,
                                  gd.SHEET_EVENTS)):
        raise ValueError("Nothing to add yet: run Analyze (whole recording and time "
                         "course), Compare segments or the event-locked analysis first.")
    if sheets is not None:
        replaced = gd.add_to_sheets(sheets, subject, sex, filepath, lines)
        n = len(sheets[gd.SHEET_ANIMALS])
    else:
        n, replaced = gd.add_recording(path, subject, sex, filepath, lines)
    names = {gd.SHEET_SEGMENTS: "segments", gd.SHEET_WHOLE: "whole recording",
             gd.SHEET_TIME: "time course", gd.SHEET_EVENTS: "events",
             gd.SHEET_ANNOT: "annotations"}
    what = ", ".join(names[k] for k in names if k in got)
    return n, replaced, f"{subject}: {what}"

