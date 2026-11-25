# -*- coding: utf-8 -*-
"""
ecg.app.morphology_helpers
--------------------------
Morphology & Intervals without its widgets: the three steps (mean beat,
check the landmarks, measure every beat) and the interval table (the
measured headline, then each interval's median, IQR and beats measured).
Moved unchanged from the Tk tab_morphology.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ecg.app.freshness import UP_TO_DATE, ModuleFreshness
from ecg.core.analysis import INTERVAL_DEFINITIONS, ensemble_short_text
from ecg.core.metric_table import (
    INTERVAL_COLUMNS, MISSING, MetricTable, interval_rows, intervals_measured,
)
from ecg.core.metric_table import MEASURED_NAMES


_DEFINITIONS = dict(INTERVAL_DEFINITIONS)


@dataclass(frozen=True)
class MorphStep:
    number: int
    text: str
    state: str
    reasons: "tuple[str, ...]" = ()
    current: bool = False     # the step the user is on: highlighted


@dataclass(frozen=True)
class IntervalLine:
    name: str
    definition: str
    median: str
    iqr: str
    measured: str


def morphology_steps(core: ModuleFreshness, intervals: ModuleFreshness,
                     n_beats: "Optional[int]") -> "list[MorphStep]":
    """The three steps. The first two rest on the core results (the mean beat and
    the template drawn on it), the third on the intervals: a template applied
    since the last measure makes step 3 out of date, through its own freshness. The
    current step is the first with something left to do: nothing records a check of the
    landmarks, so step 2 lasts until the beats are measured with them."""
    beats = f" · {n_beats} beats extracted" if n_beats else ""
    on_core = core.state != UP_TO_DATE
    on_landmarks = not on_core and intervals.state != UP_TO_DATE
    return [
        MorphStep(1, f"Mean beat{beats}", core.state, core.reasons, on_core),
        MorphStep(2, "Check landmarks on the template", core.state, core.reasons, on_landmarks),
        MorphStep(3, "Measure every beat", intervals.state, intervals.reasons),
    ]


def step_tip(step: MorphStep) -> str:
    why = "; ".join(step.reasons)
    tip = f"Step {step.number}: {step.state}" + (f" — {why}" if why else "")
    if step.number == 2 and step.current:
        tip += " — check the landmarks on the mean beat, then Measure intervals"
    return tip


def measured_headline(ivl: Any) -> "tuple[str, str]":
    """('16/1748 beats measured (1%) on PR, QRS and QT', tier of that share): what
    the medians next to it rest on, in the words and grades of the Intervals status
    line; each interval's own count is in its line. ('', '') when not delineated."""
    if not MetricTable(interval_rows(ivl)).has("intervals.beats_measured"):
        return "", ""
    n_ok, total = intervals_measured(ivl)
    pct = 100.0 * n_ok / total if total else 0.0
    tier = "good" if pct >= 90 else "partial" if pct >= 50 else "low"
    smoothed = ensemble_short_text(ivl)
    tail = f" · {smoothed}" if smoothed else ""
    return f"{n_ok}/{total} beats measured ({pct:.0f}%) on {MEASURED_NAMES}{tail}", tier


def interval_lines(ivl: Any) -> "list[IntervalLine]":
    """One line per interval, read from the shared table; dashes when there is
    nothing measured. A frame from an older version keeps its old definitions,
    which the ones listed here do not describe."""
    t = MetricTable(interval_rows(ivl))
    lines = []
    for col, name in INTERVAL_COLUMNS:
        row = t.row(f"intervals.{col}")
        definition = _DEFINITIONS.get(name, "")
        if row is None:
            lines.append(IntervalLine(name, definition, MISSING, MISSING, MISSING))
            continue
        if row.note:
            definition = "legacy definition (older session)"
        q1, q3 = t.text(f"intervals.{col}.q1"), t.text(f"intervals.{col}.q3")
        total = row.basis_total or 0
        pct = f"{100.0 * (row.basis_n or 0) / total:.0f}%" if total else MISSING
        lines.append(IntervalLine(
            name, definition, t.text(f"intervals.{col}"),
            f"{q1}–{q3}" if q1 != MISSING else MISSING,
            f"{pct} · {row.basis_n}/{total}"))
    return lines

