# -*- coding: utf-8 -*-
"""
ecg.services.events_service
---------------------------
Abnormal-event classification (Events tab) without the UI: the bounds of
its three settings, the inputs it is given (baseline, the PR of the
AV-delay check, the removed beats put back), the classification and the
premature-beat summary.

Moved unchanged from ecg.ui.analysis_controller (run_arrhythmia_analysis
and AnalysisController.av_delay_input). The controller calls the three
steps at the same moments as before: prepare_events and the settings on
the main thread, classify_events in the worker, summarise_events when the
result lands. The reference outputs of tests/regression check it is the
same computation.

Peaks are sample indices; times in seconds; PR in ms.
Nothing here may import from ecg.ui.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from ecg.core.analysis import align_interval_column_to_peaks, ensemble_beats_used
from ecg.core.detection import (arrhythmia_baseline_info, classify_arrhythmias,
                                premature_beat_summary, removed_beats_for_classification)


def event_settings(baseline: Any, brady_pct: Any, min_beats: Any) -> "tuple[float, float, int]":
    """(baseline s, bradycardia threshold %, minimum beats) from the entered
    values (text, numbers, or None for a missing entry), bounded as the
    Events tab always did: baseline >= 5 s (default 30), threshold within
    5-60 % (default 20), at least 3 beats (default 10). An unreadable value
    gives the default."""
    try:
        baseline_s = max(5.0, float(baseline))
    except (ValueError, TypeError):
        baseline_s = 30.0
    try:
        brady = max(5.0, min(60.0, float(brady_pct)))
    except (ValueError, TypeError):
        brady = 20.0
    try:
        beats = max(3, int(min_beats))
    except (ValueError, TypeError):
        beats = 10
    return baseline_s, brady, beats


def av_delay_input(ivl: "Any", rpeaks: "np.ndarray",
                   fs: float) -> "tuple[np.ndarray | None, str]":
    """(PR per peak or None, the note saying why) for the AV-delay check.

    AV conduction-delay detection needs per-beat PR data from Interval
    delineation, which is a separate, optional analysis step. Its
    DataFrame is not one row per peak (edge beats have no row), so PR
    (P onset -> QRS onset) is matched to these windowed rpeaks by sample
    index, NaN where a peak has no row (e.g. peaks edited since). The
    check only runs when at least half the peaks got a PR -- Intervals
    run on another analysis window leaves most peaks unmatched, and a
    sparse PR series would just fragment the runs."""
    pr_ms = None
    av_note = "AV-delay check skipped — measure intervals on the current window"
    n_ens = ensemble_beats_used(ivl)
    if ivl is not None and not ivl.empty:
        pr_aligned = align_interval_column_to_peaks(ivl, rpeaks, fs, "PR_ms")
        if np.isfinite(pr_aligned).sum() >= 0.5 * len(rpeaks):
            pr_ms = pr_aligned
            av_note = "AV-delay check on"
            if n_ens > 1:
                av_note += (f" (PR is a rolling median of {n_ens} beats: only a run of "
                            f"at least {n_ens} rows counts, a shorter episode is missed)")
        elif (np.isfinite(align_interval_column_to_peaks(ivl, rpeaks, fs, "R_peak_s")).sum()
              >= 0.5 * len(rpeaks)):
            # Rows do match this window -- PR itself was mostly out of
            # bounds, so re-running Intervals as-is would not help.
            av_note = "AV-delay check skipped — PR measured on < 50% of beats"
    return pr_ms, av_note


def prepare_events(rpeaks: np.ndarray, fs: float, baseline_s: float, intervals: Any,
                   artifact_report: "dict | None", accepted_peaks: "np.ndarray | None",
                   tol: int) -> dict:
    """The inputs of the classification of *rpeaks* (the analysis window's
    peaks): the rhythm baseline ("baseline", arrhythmia_baseline_info), the
    PR of the AV-delay check ("pr_ms", "av_note", from *intervals*), the
    beats removed by artifact correction that count again ("removed",
    "removal_info", matched to *accepted_peaks* within *tol* samples) and
    the minimum length of an AV-delay run ("min_av_beats": the rolling
    median of the intervals)."""
    bl_info = arrhythmia_baseline_info(rpeaks, fs, baseline_s)
    pr_ms, av_note = av_delay_input(intervals, rpeaks, fs)
    removed, removal_info = removed_beats_for_classification(artifact_report, accepted_peaks, tol)
    return {"baseline": bl_info, "pr_ms": pr_ms, "av_note": av_note, "removed": removed,
            "removal_info": removal_info, "min_av_beats": ensemble_beats_used(intervals)}


def classify_events(rpeaks: np.ndarray, fs: float, prep: dict, baseline_s: float,
                    brady_pct: float, min_beats: int,
                    progress_cb: "Optional[Callable[[int, str], None]]" = None) -> list:
    """The abnormal events of *rpeaks* (classify_arrhythmias) with the inputs
    of prepare_events."""
    removed = prep["removed"]
    return classify_arrhythmias(
        rpeaks, fs,
        baseline_s=baseline_s,
        brady_pct=brady_pct,
        min_brady_beats=min_beats,
        progress_cb=progress_cb,
        pr_ms=prep["pr_ms"],
        removed_beats=removed if len(removed) else None,
        min_av_beats=prep["min_av_beats"],
    )


def summarise_events(events: list, rpeaks: np.ndarray, fs: float, prep: dict) -> dict:
    """The premature-beat summary of *events* over the span of *rpeaks*."""
    return premature_beat_summary(
        events, prep["baseline"]["span_s"], prep["removal_info"],
        t_start=float(rpeaks[0]) / fs, t_end=float(rpeaks[-1]) / fs)
