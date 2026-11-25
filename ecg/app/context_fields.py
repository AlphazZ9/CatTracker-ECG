# -*- coding: utf-8 -*-
"""
ecg.app.context_fields
----------------------
What the context of the results says (analysis window, exclusions,
artifacts, computed / edited times) and each tab's state from the
freshness of its modules. Moved unchanged from the Tk context strip.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from ecg.app.freshness import COMPUTING, NOT_COMPUTED, UP_TO_DATE, combine_states
from ecg.app.tab_keys import TAB_COMPARE, TAB_DETECTION, TAB_MODULES, TAB_REPORT
from ecg.core.analysis import QTC_FORMULA_LABELS, normalize_qtc_formula
from ecg.core.metric_table import MISSING, MetricTable
from ecg.core.models import QT_REPORTED


TONE_HEAD = "head"


TONE_NORMAL = "normal"


TONE_WARN = "warn"


TONE_MUTED = "muted"


@dataclass(frozen=True)
class StripField:
    key: str
    text: str
    tone: str = TONE_NORMAL


def _number(v: float) -> str:
    return f"{float(v):.1f}".rstrip("0").rstrip(".")


def window_text(window: Any, signal_end_s: "Optional[float]") -> str:
    """'0–300 s': *window* as stored, 0 meaning signal start / end."""
    t0, t1 = window if window is not None else (0.0, 0.0)
    t0 = float(t0 or 0.0)
    t1 = float(t1 or 0.0) or signal_end_s
    if t1 is None:
        return MISSING
    return f"{_number(t0)}–{_number(t1)} s"


def exclusion_field(t: "Optional[MetricTable]", noisy: bool = False) -> StripField:
    """How many RR intervals HRV left out and why, from the table's rr.* rows."""
    if t is None or not t.has("rr.n_excluded"):
        return StripField("excluded", f"RR excluded: {MISSING}", TONE_MUTED)
    n = int(t.value("rr.n_excluded"))
    if n <= 0:
        return StripField("excluded", "0 RR excluded")
    parts = [f"{int(t.value(key, 0))} {name}" for key, name in (
        ("rr.n_too_short", "too short"), ("rr.n_too_long", "too long"),
        ("rr.n_spanning_removed",
         "in a noisy region or across a removed beat" if noisy
         else "spanning a removed beat")) if t.value(key, 0)]
    parts.append(f"{float(t.value('rr.excluded_pct', 0.0)):.2f} %")
    return StripField("excluded", f"{n} RR excluded ({' · '.join(parts)})", TONE_WARN)


def artifact_field(t: MetricTable) -> StripField:
    if not t.has("artifact.removed"):
        return StripField("artifact", "Artifact correction: not applied")
    n = int(t.value("artifact.removed"))
    return StripField("artifact", "Artifact correction: "
                      + (f"{n} beat{'s' if n != 1 else ''} removed" if n else "no beat removed"))


def times_text(computed_at: "Optional[str]", edited_at: "Optional[str]",
               now: "Optional[dt.datetime]" = None) -> str:
    """'Computed 14:32 · peaks edited 14:35'; the date is added when not today."""
    now = now or dt.datetime.now()

    def fmt(iso: "Optional[str]") -> str:
        try:
            when = dt.datetime.fromisoformat(str(iso))
        except ValueError:
            return ""
        return when.strftime("%H:%M" if when.date() == now.date() else "%Y-%m-%d %H:%M")

    parts = []
    if computed_at and fmt(computed_at):
        parts.append(f"Computed {fmt(computed_at)}")
    if edited_at and fmt(edited_at):
        parts.append(f"peaks edited {fmt(edited_at)}")
    return " · ".join(parts)


def context_fields(results: "Optional[dict]", table: "Optional[MetricTable]",
                   artifact_table: MetricTable, has_signal: bool, n_peaks: "Optional[int]",
                   window: Any, signal_end_s: "Optional[float]",
                   current_rr: "tuple[float, float]", qtc_formula: str,
                   noisy: bool = False) -> "list[StripField]":
    """The strip's fields. With core results they say what those were computed
    on; without, what the next Analyze would use."""
    if not has_signal:
        head = "No recording"
    else:
        head = "Analysed" if results is not None else "Not analysed"
    if results is not None and table is not None and table.has("hr.n_beats"):
        n_peaks = int(table.value("hr.n_beats"))
    lo, hi = current_rr
    if table is not None and table.has("rr.rr_min_ms"):
        lo, hi = table.value("rr.rr_min_ms"), table.value("rr.rr_max_ms")
    return [
        StripField("head", head, TONE_HEAD),
        StripField("window", f"Window {window_text(window, signal_end_s) if has_signal else MISSING}"),
        StripField("beats", f"{n_peaks if n_peaks is not None else MISSING} peaks"),
        StripField("rr", f"RR bounds {float(lo):.1f}–{float(hi):.1f} ms"),
        exclusion_field(table, noisy),
        artifact_field(artifact_table),
    ] + ([StripField("qtc", "QTc: " + QTC_FORMULA_LABELS[normalize_qtc_formula(qtc_formula)])]
         if QT_REPORTED else [])          # no QTc while QT is not reported


def pending_window_field(results: "Optional[dict]", analysed: Any, applied: Any,
                         signal_end_s: "Optional[float]") -> "Optional[StripField]":
    """With core results, a window applied since they were computed: the
    window entries already show it, so the strip says what the numbers on
    screen still rest on. None otherwise."""
    if results is None:
        return None
    done, new = window_text(analysed, signal_end_s), window_text(applied, signal_end_s)
    if done == new:
        return None
    return StripField("window_pending", f"Results on {done}: Analyze to use {new}", TONE_WARN)


def unapplied_window_field(typed: "tuple[str, str]", applied: Any) -> "Optional[StripField]":
    """Window entries that do not say the applied window (typed, not yet
    applied): the entries stand where the strip states the window, so the
    strip says the next Analyze would not use them. None when they agree."""
    try:
        t0, t1 = (float(v.strip()) if v.strip() else 0.0 for v in typed)
    except ValueError:
        t0 = t1 = float("nan")
    a0, a1 = applied if applied is not None else (0.0, 0.0)
    if abs(t0 - float(a0 or 0.0)) < 1e-9 and abs(t1 - float(a1 or 0.0)) < 1e-9:
        return None
    return StripField("window_unapplied", "Window typed, not applied: Enter or Apply", TONE_WARN)


def tab_states(states: "dict[str, str]", has_peaks: bool,
               running: "Iterable[str]" = ()) -> "dict[str, str]":
    """Tab -> state. Signal & Detection judges the peaks themselves: they are
    current as soon as they exist, whatever was computed from them. A tab with
    a module in *running* is computing, whatever its parts say; Report holds
    them all, so it is computing as soon as one of them is."""
    running = set(running)
    out = {TAB_DETECTION: UP_TO_DATE if has_peaks else NOT_COMPUTED}
    # Compare segments computes on demand from the peaks: their state.
    out[TAB_COMPARE] = out[TAB_DETECTION]
    for tab, modules in TAB_MODULES.items():
        out[tab] = (COMPUTING if running.intersection(modules)
                    else combine_states(states[m] for m in modules))
    if running:
        out[TAB_REPORT] = COMPUTING
    return out

