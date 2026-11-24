# -*- coding: utf-8 -*-
"""
ecg.services.session_service
----------------------------
The recording's side of a session (.ecgsession): which parts of the
recording state are saved and how (model_session_fields), and how a saved
session is turned back into a recording (restore_bundle: reload the signal
from the .mat file, prepare it again with the saved filter parameters, read
back the peaks edits, results, annotations...).

Moved from ecg.ui.session_controller, unchanged (collect_session_state's
model fields, the body of restore_session_worker, wave_template_from_session
and pacing_as_annotations). The interface adds its own fields (current tab,
view position, ticked metrics, widget values) and maps the saved tab names
to its own. The file format does not change (ecg.io.session, SESSION_VERSION).

Signal arrays are NOT saved: restore_bundle re-prepares them from the .mat
file, which keeps a session small whatever the recording length.

Nothing here may import from ecg.ui.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

from ecg.core.analysis import migrate_legacy_results, signal_crop_info
from ecg.core.models import FilterParams, normalize_qtc_formula
from ecg.core.pipeline import prepare_signal
from ecg.core.recording import AnalysisState, DetectionState, SignalState
from ecg.core.wave_template import WaveTemplate
from ecg.io.loaders import (
    _deserialise_results, _serialise_results, load_mat_signal, read_spike2_markers,
)

log = logging.getLogger("ecg")


def wave_template_from_session(state: dict) -> "WaveTemplate | None":
    """The recording's wave template saved in a session *state* (None for
    sessions saved before templates were kept per recording -- the saved
    default template is then used at the next Intervals run)."""
    raw = state.get("wave_template")
    return WaveTemplate.from_dict(raw) if isinstance(raw, dict) else None


PACING_COLOR = "#14B8A6"


def pacing_as_annotations(periods: "Any") -> "list[dict]":
    """The pacing periods of a session saved before 6.2 ({t_start, t_end,
    note}) as annotations, so they keep driving nothing but stay visible and
    can now align time courses and event-locked analyses."""
    out: "list[dict]" = []
    for p in periods or []:
        try:
            t0, t1 = float(p["t_start"]), float(p["t_end"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({"t_start": t0, "t_end": t1,
                    "label": str(p.get("note") or "").strip() or "Pacing",
                    "color": PACING_COLOR})
    return out


def model_session_fields(signal: SignalState, detection: DetectionState,
                         analysis: AnalysisState,
                         artifact_report: "dict | None") -> dict:
    """The recording's fields of a session, keyed as in the file. Optional
    ones (prepare_params, results, analysed_window, artifact_report,
    artifact_candidates, epoch_df, rolling_hrv_df, windowed_view,
    annotations, noise_regions, wave_template) are present only when there
    is something to save. *artifact_report*: only the removals still in
    effect on the current peaks (AnalysisController.current_artifact_report)
    -- restore re-detects the peaks and re-applies these, so it must not
    remove beats that have come back since."""
    out: dict = {
        "fs":              int(signal.fs),
        "threshold":       float(detection.thresh_amp),
        "manual_excluded": list(detection.manual_excluded),
        "manual_added":    list(detection.manual_added),
        "no_filter_mode":  signal.no_filter_mode,
        "signal_inverted": signal.inverted,
        # Beat count stored for the restore-dialog display only
        "n_beats":         len(detection.rpeaks_ok) if detection.rpeaks_ok is not None else 0,
        "analysis_t_start": analysis.t_start,
        "analysis_t_end":   analysis.t_end,
    }
    # What the signal / peaks were prepared with (filters, crop) -- the
    # restore re-prepares from it, so the results keep their provenance.
    prep = getattr(signal, "prepare_params", None)
    if prep is not None:
        out["prepare_params"] = dict(prep)
    if analysis.results is not None:
        out["results"] = _serialise_results(analysis.results)
        # The window the results were computed on (the entries can have
        # been applied since without re-running Analyze).
        aw = analysis.analysed_window
        if aw is not None:
            out["analysed_window"] = [float(aw[0] or 0.0), float(aw[1] or 0.0)]
    if artifact_report is not None:
        out["artifact_report"] = artifact_report
    if analysis.artifact_candidates:
        out["artifact_candidates"] = analysis.artifact_candidates
    if analysis.epoch_df is not None and not analysis.epoch_df.empty:
        out["epoch_df"] = analysis.epoch_df.to_dict(orient="list")
    if analysis.rolling_hrv_df is not None and not analysis.rolling_hrv_df.empty:
        out["rolling_hrv_df"] = analysis.rolling_hrv_df.to_dict(orient="list")
    if analysis.windowed_view:
        out["windowed_view"] = analysis.windowed_view
    if analysis.annotations:
        out["annotations"] = analysis.annotations
    if analysis.noise_regions:
        out["noise_regions"] = analysis.noise_regions
    # The recording's wave template: restoring the session restores the
    # windows (and provenance) its delineation used / will use.
    if analysis.wave_template is not None:
        out["wave_template"] = analysis.wave_template.to_dict()
    return out


def restore_bundle(state: dict, path: str,
                   prepare: "Callable[[np.ndarray, Any, dict], dict]" = prepare_signal) -> dict:
    """Everything a saved session *state* restores, for the recording file
    *path*: reloads the raw signal (cropped as saved), prepares it again
    with *prepare* (default ecg.core.pipeline.prepare_signal) and the saved
    filter parameters, and reads back the rest. Plain data, no UI: the
    interface writes it to its state and widgets on the main thread.

    "current_tab" and "time_course_mode" are returned as saved; the
    interface maps former names to its own.
    """
    fs = int(state["fs"])

    # ── Parse saved filter params via FilterParams.from_dict ────────
    # from_dict tolerates missing keys with safe defaults, so old session
    # files (saved before any new field was added) restore without error.
    # fp = the widget values at save time (restored onto the widgets);
    # prep = the params the saved signal / peaks / results were prepared
    # with (SignalState.prepare_params). They differ when the cut-offs,
    # notch or crop were edited after detection without re-detecting:
    # the signal is rebuilt from prep, so the restored results and their
    # Filters / Signal crop provenance rows match. Sessions saved before
    # prepare_params was recorded fall back to the widget values.
    fp = FilterParams.from_dict(state.get("filter_params", {}))
    prep_raw = state.get("prepare_params")
    prep = FilterParams.from_dict(prep_raw) if isinstance(prep_raw, dict) else fp

    try:
        sig_raw, detected_ch, _, detected_fs = load_mat_signal(path, prep.channel)
    except Exception as exc:
        raise RuntimeError(f"Could not reload signal from {path}: {exc}") from exc

    if detected_fs is not None:
        fs = int(detected_fs)

    # Apply the same time-crop the saved signal was prepared with
    i0 = int(prep.t_start * fs) if prep.t_start > 0 else 0
    i1 = int(prep.t_end   * fs) if prep.t_end   > 0 else len(sig_raw)
    signal_crop = signal_crop_info(prep.t_start, prep.t_end, len(sig_raw) / fs)
    # Spike2 markers re-read from the file (not saved in the session).
    markers = read_spike2_markers(path, prep.t_start, prep.t_end)
    sig_raw = sig_raw[i0:i1]
    n_samples = len(sig_raw)

    # ── Re-run signal processing with saved filter params (pure/static) ──
    # prep.to_dict() produces the "notch" key (not "notch_filter") that
    # _compute_preview_bundle expects.
    prepare_params = prep.to_dict()
    # Not a FilterParams field (see compute_preview_bundle): carried over
    # so the SG + Derivative candidates use the saved window.
    if isinstance(prep_raw, dict) and prep_raw.get("sg_window_ms") is not None:
        prepare_params["sg_window_ms"] = prep_raw["sg_window_ms"]
    signal_bundle = prepare(sig_raw, fs, prepare_params)
    # The analysis window the saved results were computed on -- not the
    # window entries, which may have been applied since without
    # re-running Analyze. Older sessions did not save it: the entries
    # are then the best remaining guess. None without results.
    aw = state.get("analysed_window")
    if not state.get("results"):
        analysed_window = None
    elif isinstance(aw, (list, tuple)) and len(aw) == 2:
        analysed_window = (float(aw[0] or 0.0), float(aw[1] or 0.0))
    else:
        analysed_window = (float(state.get("analysis_t_start", 0.0)),
                           float(state.get("analysis_t_end",   0.0)))

    results_raw  = state.get("results")
    results      = _deserialise_results(results_raw) if results_raw else None
    # Results saved by an older version: recompute the HR stats under
    # the current definitions and note a legacy (arbitrary-unit) spectrum.
    legacy_notes = migrate_legacy_results(results)
    for _note in legacy_notes:
        log.info("Session restore: %s", _note)
    epoch_raw    = state.get("epoch_df")
    rolling_raw  = state.get("rolling_hrv_df")

    return {
        "fs":               fs,
        "n_samples":        n_samples,
        "detected_ch":      detected_ch,
        "signal_raw":       sig_raw,
        "signal_bundle":    signal_bundle,
        "signal_crop":      signal_crop,
        "markers":          markers,
        "filter_params":    fp,
        "thresh_amp":       float(state.get("threshold", float(state.get("thresh_amp", 0.5)))),
        # The slider FRACTION (0.01-2.0, apply_threshold()'s thresh_frac
        # param) -- distinct from thresh_amp above, which is the
        # DERIVED absolute amplitude (thresh_frac * local median
        # prominence, can be e.g. 5.7). Sessions saved before this field
        # existed only have thresh_amp under the "threshold" key, so
        # fall back to the slider's own default (0.5) rather than
        # re-running detection with an amplitude value in the fraction
        # slot -- apply_threshold() would then accept almost no peaks
        # (any prominence >= amplitude * median_prominence).
        "thresh_frac":      float(state.get("threshold_frac", 0.5)),
        "manual_excluded":  set(int(x) for x in state.get("manual_excluded", [])),
        "manual_added":     set(int(x) for x in state.get("manual_added",    [])),
        "signal_inverted":  bool(state.get("signal_inverted", False)),
        "no_filter_mode":   bool(state.get("no_filter_mode",  fp.no_filter)),
        "verified_for_training": bool(state.get("verified_for_training", False)),
        "results":          results,
        "legacy_notes":     legacy_notes,
        "artifact_report":  state.get("artifact_report"),
        "artifact_candidates": state.get("artifact_candidates", []),
        "epoch_df":         pd.DataFrame(epoch_raw)   if epoch_raw   is not None else None,
        "rolling_hrv_df":   pd.DataFrame(rolling_raw) if rolling_raw is not None else None,
        # Pacing periods of older sessions (drawn only, no effect on any
        # result) come back as annotations.
        "annotations":      list(state.get("annotations", []))
                            + pacing_as_annotations(state.get("pacing_periods")),
        "noise_regions":    list(state.get("noise_regions", [])),
        "analysis_t_start": float(state.get("analysis_t_start", 0.0)),
        "analysis_t_end":   float(state.get("analysis_t_end",   0.0)),
        "analysed_window":  analysed_window,
        # What each module was computed on (absent in older sessions).
        "freshness_basis":  state.get("freshness_basis"),
        # The context strip's times (absent in older sessions).
        "computed_at":      state.get("computed_at"),
        "peaks_edited_at":  state.get("peaks_edited_at"),
        "report_include":   state.get("report_include"),
        "nav_pos":          float(state.get("nav_pos", 0.0)),
        "edit_mode":        bool(state.get("edit_mode", False)),
        # The tab and Time course mode as saved; the interface maps
        # former names to its own (ecg.ui.session_controller).
        "current_tab":      state.get("current_tab", "Detection"),
        "qtc_formula":      normalize_qtc_formula(state.get("qtc_formula", "mitchell")),
        # None in a session saved before the rolling median.
        "ensemble_beats":   state.get("ensemble_beats"),
        # None when the session names no mode: the current one stays. An
        # older session may hold the two values this replaced.
        "time_course_mode": state.get("time_course_mode"),
        "windowed_view":    state.get("windowed_view") or {},
        # The ticks of Epochs and Sliding (absent in older sessions).
        "epoch_metrics":    state.get("epoch_metrics"),
        "roll_metrics":     state.get("roll_metrics"),
        # Raw: absent in sessions saved before the Autonomic view (then
        # None -- the current parameters stay); normalised on apply,
        # where an invalid key keeps the current value.
        "autonomic_params": state.get("autonomic_params"),
        # Same for the Event-locked panel (label filter, pre / post, z).
        "event_locked_params": state.get("event_locked_params"),
        # Subject ID, SG window, raw / filtered view (absent in older sessions).
        "recording_fields": state.get("recording_fields"),
        # The Compare segments rows / reference / view and whether the
        # comparison was run (absent in older sessions: None, nothing
        # restored). Its statistics are not saved: re-run on restore.
        "compare_segments": state.get("compare_segments"),
        # Which filter_params keys the session holds: only those are
        # written back to widgets the restore did not always set.
        "filter_keys":      sorted(state.get("filter_params", {}) or {}),
        # The recording's own wave template (with its provenance).
        "wave_template":    wave_template_from_session(state),
    }
