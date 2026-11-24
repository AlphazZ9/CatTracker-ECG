"""
tests.regression.golden
-----------------------
Reference ("golden master") outputs of the whole analysis pipeline, and the
comparison that tells whether a later version still computes them.

Why
    The unit tests check each algorithm on synthetic signals. Nothing checked
    that a real recording gives the same numbers from one version to the
    next. Before the CustomTkinter -> PySide6 migration moves code out of the
    UI controllers, this module records what version 6.2.0 computes, end to
    end, so every later step can be compared against it.

How
    ``run_pipeline`` drives the app's own code without a window: the
    SignalController / DetectionController / AnalysisController methods the
    buttons call (Detect Peaks, Analyze, Epochs, Sliding...) run on a stub app
    that holds the state dataclasses of ecg.ui.state and no widget. Where a
    button only assembles parameters and calls one ecg.core function, that
    function is called here with the same parameters (the defaults of
    FilterParams and of the widgets' initial values). The scientific code
    that runs is therefore the app's, unchanged.

    ``flatten`` turns the outputs (arrays, DataFrames, dicts, events) into a
    flat {name: value} map; ``save`` writes it as ``<stem>.npz`` (arrays) +
    ``<stem>.json`` (scalars and metadata); ``compare`` lists every
    difference between two maps.

Tolerance
    Integers, booleans and strings (peak indices, counts, flags, labels) must
    be equal. Floats must satisfy |a - b| <= ATOL + RTOL * |b|, NaN equal to
    NaN; values in units of the z-scored signal use the absolute tolerance
    SIGNAL_ATOL instead of ATOL (see SIGNAL_UNIT_NAMES). The tolerance absorbs floating-point summation-order effects of a
    different BLAS or CPU, not a change of method: a moved algorithm gives
    the same floats to the last bit on the same machine.

Signals
    A per-sample array (the filtered signal: 7.2 M samples per hour at
    2 kHz) is stored on a regular stride of at most SIGNAL_POINTS samples,
    plus its length, mean, standard deviation, minimum and maximum.

Nothing here imports Tk widgets or opens a window. The user's own settings
(~/.ecg_wave_template.json, the ML model) are not read: the built-in default
wave template is used, and the ML detector is not part of the reference.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

FORMAT_VERSION = 1
SIGNAL_POINTS = 10_000          # per-sample arrays: at most this many stored points
SIGNAL_MIN_LEN = 200_000        # arrays at least this long count as per-sample
RTOL = 1e-9
ATOL = 1e-12
# Values in units of the z-scored ECG (standard deviation 1): the prepared
# signal, prominences, amplitudes and beat templates. Their error is an
# absolute error on the signal scale -- a sample near a zero crossing has a
# meaningless relative error -- so they get this absolute tolerance, 1e-7 of
# the signal's standard deviation. Observed between Linux and Windows runners
# with the same library versions: 1e-11 on one, 1.3e-9 on another -- the
# low high-pass cut-offs (ecg_clean, band-pass) are ill-conditioned filters
# that amplify the CPU's rounding. A real change of method is far larger
# (normalising with ddof=1, ~2e-6, still changes 75 outputs).
SIGNAL_ATOL = 1e-7
SIGNAL_UNIT_NAMES = ("signal_flt", "signal_raw_norm", "all_proms", "thresh_amp",
                     "beat_template", "beat_matrix", "beat_sd", "peak_amps")
# Libraries whose version is part of the reference: a different version may
# legitimately change a result (tests/regression/test_golden_master.py).
SCIENTIFIC_PACKAGES = ("numpy", "scipy", "pandas", "neurokit2", "scikit-learn", "PyWavelets")

# Detection scenarios: name -> FilterParams overrides. "auto_raw" is the
# FilterParams defaults (no filtering, Auto detector, threshold 0.5) -- the
# analyses below run on it. (The Tk Method box itself starts on SG +
# Derivative: that start is the "sg_derivative" scenario.)
DETECTION_SCENARIOS: "dict[str, dict]" = {
    "auto_raw":          {},
    "auto_filtered":     {"no_filter": False},
    "auto_filtered_notch": {"no_filter": False, "notch_filter": True},
    "sg_derivative":     {"detection_method": "sg_derivative"},
    "wavelet":           {"detection_method": "wavelet"},
    "envelope_max":      {"detection_method": "envelope_max"},
}

# Widget defaults (ecg/ui/tab_time_course.py, tab_events.py): Sliding window
# and step, and the metrics ticked at start-up.
SLIDING_WIN_S = 30.0
SLIDING_STEP_S = 5.0
SLIDING_DEFAULT_METRICS = ("HR", "SDNN", "RMSSD")
SLIDING_ALL_METRICS = ("HR", "SDNN", "RMSSD", "pNN6", "SD1", "SD2", "CCD",
                       "SD1_LW", "SD2_LW", "LF_nu", "HF_nu", "LF_HF")
EPOCH_ALL_METRICS = ("HR", "SDNN", "RMSSD", "pNN6", "SD1", "SD2", "LF_nu", "HF_nu", "LF_HF")
ANALYSIS_WINDOW_S = (20.0, 120.0)
# Noisy regions marked by hand ({t_start, t_end} in s): the synthetic
# recording's EMG burst (95-96.5 s) and a stretch of clean signal.
NOISE_REGIONS_S = ({"t_start": 40.0, "t_end": 45.5}, {"t_start": 95.0, "t_end": 96.5})


# ════════════════════════════════════════════════════════════
#  Stub app: the controllers' view of ECGApp, without widgets
# ════════════════════════════════════════════════════════════

class _Null:
    """Any widget or UI callback the controllers touch: every attribute is
    this object again and every call returns None, so label updates, plot
    slots and buttons do nothing."""

    def __getattr__(self, name: str) -> "_Null":
        if name.startswith("__"):
            raise AttributeError(name)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None

    def __getitem__(self, key: Any) -> "_Null":
        return self

    def __iter__(self):
        # Iterating a widget (its children, its items) yields nothing: with
        # __getitem__ alone, Python's fallback iteration would never stop.
        return iter(())


class _Entry:
    """A text entry or check box holding *value* (its .get())."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def get(self) -> Any:
        return self.value


class _StubApp(SimpleNamespace):
    """ECGApp as the controllers see it: state dataclasses and entries with
    their start-up values; anything else is a _Null widget."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return _NULL


_NULL = _Null()


def _safe_float(widget: Any, default: float) -> float:
    try:
        return float(widget.get())
    except (TypeError, ValueError, AttributeError):
        return default


def _new_app(path: "str | Path") -> _StubApp:
    from ecg.core.models import MouseECG
    from ecg.core.wave_template import WaveTemplate
    from ecg.ui.analysis_controller import AnalysisController
    from ecg.ui.detection_controller import DetectionController
    from ecg.ui.signal_controller import SignalController
    from ecg.ui.state import AnalysisState, DetectionState, SignalState, UIState

    app = _StubApp(
        signal=SignalState(filepath=str(path)),
        detection=DetectionState(),
        analysis=AnalysisState(wave_template=WaveTemplate()),
        ui=UIState(),
        _generation=0,
        after=lambda ms, fn=None: None,
        after_idle=lambda fn=None: None,
        _set_status=lambda text, color=None: None,
        _set_progress=lambda pct, msg: None,
        _safe_float=_safe_float,
        ent_minrr=_Entry(MouseECG.RR_MIN_MS),
        ent_maxrr=_Entry(MouseECG.RR_MAX_MS),
        ent_entropy_cap=_Entry(""),
        cb_freq_band=None,
        sw_permissive=None,
        ent_epoch=_Entry(MouseECG.EPOCH_DEFAULT_S),
        ent_overlap=_Entry(0),
        ent_roll_win=_Entry(SLIDING_WIN_S),
        ent_roll_step=_Entry(SLIDING_STEP_S),
        ent_arr_baseline=_Entry(30),
        ent_arr_brady_pct=_Entry(20),
        ent_arr_min_beats=_Entry(10),
        _roll_metrics={m: _Entry(1) for m in SLIDING_DEFAULT_METRICS},
        _epoch_metrics=None,
        _start_async_result=lambda btn, label, worker, done: done(worker()),
        tabs=SimpleNamespace(set=lambda name: None, get=lambda: ""),
    )
    app.signal_ctrl = SignalController(app)
    app.detection_ctrl = DetectionController(app)
    app.analysis_ctrl = AnalysisController(app)
    app._windowed_peaks = app.signal_ctrl.windowed_peaks
    return app


def _params(**overrides: Any) -> dict:
    """The parameter dict Detect Peaks hands its worker (snapshot_params)
    with the FilterParams defaults, plus *overrides* (FilterParams fields)."""
    from ecg.core.models import FilterParams
    fp = FilterParams()
    for key, value in overrides.items():
        setattr(fp, key, value)
    params = fp.to_dict()
    params["analysis_t_start"] = 0.0
    params["analysis_t_end"] = 0.0
    params["sg_window_ms"] = 20.0
    return params


def _land_preview(app: _StubApp, bundle: dict) -> None:
    """The state writes of SignalController.on_preview_done that the
    analysis reads (the rest updates widgets)."""
    import numpy as np
    fs = bundle["fs"]
    app.signal.fs = fs
    app.signal.raw = bundle["signal_raw"]
    app.signal.raw_norm = bundle["signal_raw_norm"]
    app.signal.filtered = bundle["signal_flt"]
    app.signal.time = np.arange(len(bundle["signal_raw"])) / fs
    app.signal.no_filter_mode = bundle["no_filter_mode"]
    app.signal.filter_text = bundle.get("filter_text")
    app.signal.inverted = bundle.get("inverted", False)
    app.detection.all_candidates = bundle["all_cands"]
    app.detection.all_prominences = bundle["all_proms"]
    app.detection.rpeaks_ok = bundle["rpeaks_ok"]
    app.detection.rpeaks_rej = bundle["rpeaks_rej"]
    app.detection.thresh_amp = bundle["thresh_amp"]


def _analysis_params(app: _StubApp, **overrides: Any) -> dict:
    """What AnalysisController.run_analysis passes analysis_worker."""
    params = _params(**{k: v for k, v in overrides.items()
                        if k not in ("analysis_t_start", "analysis_t_end")})
    params["analysis_t_start"] = overrides.get("analysis_t_start", app.analysis.t_start)
    params["analysis_t_end"] = overrides.get("analysis_t_end", app.analysis.t_end)
    params["hrv_rr_bounds"] = app.analysis_ctrl.hrv_rr_bounds()
    params["break_tol_samples"] = app.analysis_ctrl.break_tol_samples()
    params["noise_regions"] = [dict(r) for r in app.analysis.noise_regions or []]
    return params


def _analyze(app: _StubApp, **overrides: Any) -> dict:
    """Analyze: analysis_worker, then the state writes of on_analysis_done."""
    params = _analysis_params(app, **overrides)
    bundle = app.analysis_ctrl.analysis_worker(params, app.analysis.artifact_report)
    app.analysis.results = bundle["results"]
    app.detection.rpeaks_ok = bundle["rpeaks_ok"]
    app.analysis.artifact_report = bundle["artifact_report"]
    app.analysis.analysed_window = bundle.get("analysis_window")
    return bundle


def _freq(app: _StubApp) -> pd.DataFrame:
    """run_freq's worker, with its parameters."""
    from ecg.core.analysis import analyse_hrv_freq
    ctl = app.analysis_ctrl
    lo, hi = ctl.results_rr_bounds()
    df = analyse_hrv_freq(app._windowed_peaks(), app.signal.fs, rr_min_ms=lo, rr_max_ms=hi,
                          breaks=ctl.hrv_breaks(), bands=ctl.get_freq_bands())
    app.analysis.results["hrv_freq"] = df
    return df


def _nonlinear(app: _StubApp) -> pd.DataFrame:
    """run_nonlinear's worker, with its parameters."""
    from ecg.core.analysis import analyse_hrv_nonlinear
    ctl = app.analysis_ctrl
    lo, hi = ctl.results_rr_bounds()
    df = analyse_hrv_nonlinear(app._windowed_peaks(), app.signal.fs,
                               max_beats=ctl.entropy_cap(), rr_min_ms=lo, rr_max_ms=hi,
                               breaks=ctl.hrv_breaks())
    app.analysis.results["hrv_nonlin"] = df
    return df


def _intervals(app: _StubApp, ensemble_beats: int) -> pd.DataFrame:
    """run_intervals's worker, with its parameters (strict bounds, Mitchell QTc)."""
    from ecg.core.analysis import analyse_intervals
    ctl = app.analysis_ctrl
    rp = app._windowed_peaks()
    fs = app.signal.fs
    lo, hi = ctl.results_rr_bounds()
    return analyse_intervals(app.signal.filtered, rp, fs, np.diff(rp).astype(float) / fs * 1000,
                             wave_template=ctl.recording_wave_template().copy(),
                             permissive_bounds=False, qtc_formula=ctl.qtc_formula(),
                             rr_min_ms=lo, rr_max_ms=hi, ensemble_beats=ensemble_beats)


def _epochs(app: _StubApp, metrics: "tuple[str, ...] | None" = None) -> "pd.DataFrame | None":
    app._epoch_metrics = ({m: _Entry(1) for m in metrics} if metrics else None)
    app.analysis.epoch_df = None
    app.analysis_ctrl.compute_epochs()
    return app.analysis.epoch_df


def _sliding(app: _StubApp, metrics: "tuple[str, ...]") -> "pd.DataFrame | None":
    app._roll_metrics = {m: _Entry(1) for m in metrics}
    app.analysis.rolling_hrv_df = None
    app.analysis_ctrl.compute_rolling_hrv()
    return app.analysis.rolling_hrv_df


def _arrhythmias(app: _StubApp) -> dict:
    """run_arrhythmia_analysis's worker and summary, with its parameters."""
    from ecg.core.analysis import ensemble_beats_used
    from ecg.core.detection import (arrhythmia_baseline_info, classify_arrhythmias,
                                    premature_beat_summary)
    ctl = app.analysis_ctrl
    rp = app._windowed_peaks()
    fs = float(app.signal.fs)
    ivl = (app.analysis.results or {}).get("intervals")
    bl = arrhythmia_baseline_info(rp, fs, 30.0)
    pr_ms, av_note = ctl.av_delay_input(ivl, rp, fs)
    removed, removal_info = ctl.arrhythmia_removed_beats(fs)
    events = classify_arrhythmias(rp, fs, baseline_s=30.0, brady_pct=20.0, min_brady_beats=10,
                                  pr_ms=pr_ms, removed_beats=removed if len(removed) else None,
                                  min_av_beats=ensemble_beats_used(ivl))
    summary = premature_beat_summary(events, bl["span_s"], removal_info,
                                     t_start=float(rp[0]) / fs, t_end=float(rp[-1]) / fs)
    return {"baseline": bl, "av_note": av_note, "pr_ms": pr_ms, "events": events,
            "summary": summary, "removal_info": removal_info}


def _autonomic(app: _StubApp, lf_hf: bool) -> dict:
    """compute_autonomic's worker with the default Autonomic parameters."""
    from ecg.core.analysis import rr_interval_series
    from ecg.core.autonomic import beat_dataframe, compute_autonomic_indices, to_dataframe
    from ecg.core.models import MouseECG
    from ecg.core.spwvd import spwvd_lf_hf
    from ecg.ui.autonomic_view import normalise_autonomic_params
    ctl = app.analysis_ctrl
    p = normalise_autonomic_params(None)
    rp = app._windowed_peaks()
    fs = app.signal.fs
    lo, hi = ctl.hrv_rr_bounds()
    breaks = ctl.hrv_breaks()
    res = compute_autonomic_indices(rp, fs, window_s=p["window_s"], method=p["method"],
                                    ks=p["ks"], kp=p["kp"], rr_min_ms=lo, rr_max_ms=hi,
                                    breaks=breaks, min_pairs=p["min_pairs"])
    out: dict = {"params": p, "result": res, "frame": to_dataframe(res),
                 "beats": beat_dataframe(res)}
    if lf_hf:
        ser = rr_interval_series(rp, fs, lo, hi, breaks=breaks)
        try:
            out["spwvd"] = spwvd_lf_hf(ser["rr_ms"], ser["t_s"], ser["valid"],
                                       lf_band=tuple(MouseECG.LF), hf_band=tuple(MouseECG.HF))
        except ValueError as exc:
            out["spwvd"] = {"error": str(exc)}
    return out


def _peak_edits(app: _StubApp) -> dict:
    """Edit mode: exclude three accepted peaks and add one by hand, then
    run_detection (threshold 0.5) rebuilds the peaks as a click does."""
    rp = np.asarray(app.detection.rpeaks_ok)
    excluded = {int(rp[len(rp) // 10]), int(rp[len(rp) // 2]), int(rp[9 * len(rp) // 10])}
    gap = int(np.argmax(np.diff(rp)))
    added = int((rp[gap] + rp[gap + 1]) // 2)
    app.detection.manual_excluded = set(excluded)
    app.detection.manual_added = {added}
    n = app.detection_ctrl.run_detection(thresh=0.5)
    return {"excluded": sorted(excluded), "added": added, "n": n,
            "rpeaks_ok": app.detection.rpeaks_ok,
            "rpeaks_manual_excl": app.detection.rpeaks_manual_excl,
            "rpeaks_manual_added": app.detection.rpeaks_manual_added,
            "sig_quality": app.detection.sig_quality}


def _bundle_outputs(bundle: dict, raw_norm: bool = False) -> dict:
    """What Detect Peaks produces. The normalised raw signal does not depend
    on the detection settings: it is kept for one scenario only."""
    keep = ("fs", "dur_s", "detected_ch", "detected_fs", "fs_from_file", "thresh",
            *(("signal_raw_norm",) if raw_norm else ()), "signal_flt", "no_filter_mode", "all_cands", "all_proms",
            "inverted", "filter_text", "rpeaks_ok", "rpeaks_rej", "thresh_amp",
            "detector_warning", "signal_crop")
    return {k: bundle.get(k) for k in keep}


# ════════════════════════════════════════════════════════════
#  The pipeline
# ════════════════════════════════════════════════════════════

def run_pipeline(path: "str | Path", channel: str = "ECG",
                 log: "Optional[Callable[[str], None]]" = None) -> "tuple[dict, dict]":
    """Every output of the reference pipeline for the recording at *path*.

    Returns (outputs, timings_s). *outputs* is a nested dict of the raw
    results (arrays, DataFrames, dicts, events); pass it to ``flatten``.
    """
    from ecg.core.detection import detect_rr_artifacts
    from ecg.core.models import MouseECG
    from ecg.io.loaders import load_mat_signal

    say = log or (lambda msg: None)
    out: dict = {}
    timings: "dict[str, float]" = {}

    def step(name: str, fn: Callable[[], Any]) -> Any:
        say(f"  {name}…")
        t0 = time.perf_counter()
        value = fn()
        timings[name] = round(time.perf_counter() - t0, 3)
        out[name] = value
        return value

    def load() -> dict:
        sig, ch, keys, fs = load_mat_signal(str(path), channel)
        return {"signal": sig, "channel": ch, "keys": keys, "fs": fs}
    step("load", load)

    for name, overrides in DETECTION_SCENARIOS.items():
        app = _new_app(path)
        step(f"detect/{name}",
             lambda a=app, o=overrides, n=name: _bundle_outputs(
                 a.signal_ctrl.preview_worker(_params(**o)), raw_norm=(n == "auto_raw")))

    # ── The main recording: FilterParams defaults (Auto detector) ──
    app = _new_app(path)
    _land_preview(app, app.signal_ctrl.preview_worker(_params()))

    def quality() -> "int | None":
        app.detection_ctrl.update_signal_quality(app.detection.rpeaks_ok)
        return app.detection.sig_quality
    step("quality/before_analysis", quality)
    step("artifact_review/candidates", lambda: detect_rr_artifacts(
        app.detection.rpeaks_ok, app.signal.fs, rr_min_ms=MouseECG.RR_MIN_MS,
        rr_max_ms=MouseECG.RR_MAX_MS, window_beats=11, dev_threshold=0.20,
        signal=app.signal.filtered))

    step("analysis/core", lambda: _analyze(app))
    step("quality/after_analysis", quality)
    step("analysis/frequency", lambda: _freq(app))
    step("analysis/nonlinear", lambda: _nonlinear(app))
    step("intervals/n1", lambda: _intervals(app, 1))
    ivl = step("intervals/n32", lambda: _intervals(app, 32))
    app.analysis.results["intervals"] = ivl
    step("epochs/default", lambda: _epochs(app))
    step("epochs/all_metrics", lambda: _epochs(app, EPOCH_ALL_METRICS))
    step("sliding/default", lambda: _sliding(app, SLIDING_DEFAULT_METRICS))
    step("sliding/all_metrics", lambda: _sliding(app, SLIDING_ALL_METRICS))
    step("events", lambda: _arrhythmias(app))
    step("autonomic", lambda: _autonomic(app, lf_hf=True))
    step("peak_edits/manual", lambda: _peak_edits(app))

    # ── Auto artifact correction (Parameters > Artifact correction) ──
    app = _new_app(path)
    _land_preview(app, app.signal_ctrl.preview_worker(_params()))
    step("corrected/core", lambda: _analyze(app, artifact_correction=True))
    step("corrected/frequency", lambda: _freq(app))
    step("corrected/nonlinear", lambda: _nonlinear(app))
    step("corrected/epochs", lambda: _epochs(app))
    step("corrected/sliding", lambda: _sliding(app, SLIDING_ALL_METRICS))
    step("corrected/events", lambda: _arrhythmias(app))
    step("corrected/peak_edits", lambda: _peak_edits(app))

    # ── Analysis window (context strip) ─────────────────────────
    app = _new_app(path)
    _land_preview(app, app.signal_ctrl.preview_worker(_params()))
    app.analysis.t_start, app.analysis.t_end = ANALYSIS_WINDOW_S
    step("window/core", lambda: _analyze(app))
    step("window/frequency", lambda: _freq(app))
    step("window/epochs", lambda: _epochs(app))
    step("window/sliding", lambda: _sliding(app, SLIDING_DEFAULT_METRICS))

    # ── Noisy regions marked by hand (their RR left out of HR / HRV) ──
    app = _new_app(path)
    _land_preview(app, app.signal_ctrl.preview_worker(_params()))
    app.analysis.noise_regions = [dict(r) for r in NOISE_REGIONS_S]
    step("noise/core", lambda: _analyze(app))
    step("noise/frequency", lambda: _freq(app))
    step("noise/nonlinear", lambda: _nonlinear(app))
    step("noise/epochs", lambda: _epochs(app, EPOCH_ALL_METRICS))
    step("noise/sliding", lambda: _sliding(app, SLIDING_ALL_METRICS))
    step("noise/events", lambda: _arrhythmias(app))
    step("noise/autonomic", lambda: _autonomic(app, lf_hf=False))

    return out, timings


# ════════════════════════════════════════════════════════════
#  Flattening
# ════════════════════════════════════════════════════════════

def _scalar(v: Any) -> Any:
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return float("nan")
    return v


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float, str, np.generic))


def _store_array(flat: dict, name: str, arr: np.ndarray) -> None:
    if arr.dtype == object:
        arr = np.array(["None" if x is None else str(x) for x in arr.ravel()]).reshape(arr.shape)
    if arr.ndim == 1 and len(arr) >= SIGNAL_MIN_LEN and arr.dtype.kind in "fiub":
        stride = int(math.ceil(len(arr) / SIGNAL_POINTS))
        a = arr.astype(np.float64)
        flat[f"{name}/len"] = int(len(arr))
        flat[f"{name}/stride"] = stride
        flat[f"{name}/samples"] = arr[::stride].copy()
        flat[f"{name}/mean"] = float(np.nanmean(a))
        flat[f"{name}/std"] = float(np.nanstd(a))
        flat[f"{name}/min"] = float(np.nanmin(a))
        flat[f"{name}/max"] = float(np.nanmax(a))
        return
    flat[name] = arr


def flatten(obj: Any, prefix: str = "", flat: "dict | None" = None) -> dict:
    """{name: array or scalar} of every value in *obj*, names joined by '/'."""
    flat = {} if flat is None else flat
    p = prefix
    if isinstance(obj, pd.DataFrame):
        flat[f"{p}/__columns__"] = json.dumps([str(c) for c in obj.columns])
        flat[f"{p}/__index__"] = np.asarray(obj.index.to_numpy())
        for col in obj.columns:
            _store_array(flat, f"{p}/{col}", np.asarray(obj[col].to_numpy()))
        if obj.attrs:
            flatten(dict(obj.attrs), f"{p}/__attrs__", flat)
    elif isinstance(obj, pd.Series):
        flatten(obj.to_frame(), p, flat)
    elif isinstance(obj, np.ndarray):
        _store_array(flat, p, obj)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        flat[f"{p}/__type__"] = type(obj).__name__
        flatten({f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}, p, flat)
    elif isinstance(obj, dict):
        if not obj:
            flat[f"{p}/__empty__"] = "dict"
        for k in sorted(obj, key=str):
            flatten(obj[k], f"{p}/{k}" if p else str(k), flat)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        items = sorted(obj) if isinstance(obj, (set, frozenset)) else list(obj)
        if all(_is_scalar(x) for x in items) and not any(isinstance(x, str) for x in items):
            flat[p] = np.asarray([np.nan if x is None else _scalar(x) for x in items])
        elif all(isinstance(x, str) for x in items):
            flat[p] = np.asarray(items, dtype=str)
        else:
            flat[f"{p}/__len__"] = len(items)
            for i, x in enumerate(items):
                flatten(x, f"{p}/{i}", flat)
    elif _is_scalar(obj):
        flat[p] = _scalar(obj)
    elif hasattr(obj, "to_dict"):
        flatten(obj.to_dict(), p, flat)
    else:
        flat[f"{p}/__type__"] = type(obj).__name__
    return flat


# ════════════════════════════════════════════════════════════
#  Save / load
# ════════════════════════════════════════════════════════════

def file_sha256(path: "str | Path") -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def environment() -> dict:
    import importlib.metadata as md
    import ecg
    versions = {}
    for pkg in ("numpy", "scipy", "pandas", "neurokit2", "scikit-learn", "PyWavelets",
                "h5py", "matplotlib", "customtkinter"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, cwd=Path(__file__).resolve().parent,
                                timeout=10).stdout.strip() or None
    except Exception:
        commit = None
    return {"app_version": ecg.__version__, "git_commit": commit,
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": versions}


def _json_value(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return {"__float__": repr(v)}
    return v


def _from_json(v: Any) -> Any:
    if isinstance(v, dict) and "__float__" in v:
        return float(v["__float__"])
    return v


def save(flat: dict, out_dir: "str | Path", stem: str, meta: dict) -> "tuple[Path, Path]":
    """Write *flat* as <stem>.npz (arrays) and <stem>.json (scalars, metadata)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays, names, scalars = {}, {}, {}
    for i, (name, value) in enumerate(sorted(flat.items())):
        if isinstance(value, np.ndarray):
            key = f"a{i:05d}"
            arrays[key] = value
            names[name] = key
        else:
            scalars[name] = _json_value(value)
    npz = out_dir / f"{stem}.npz"
    np.savez_compressed(npz, **arrays)
    manifest = {"format_version": FORMAT_VERSION, "tolerance": {"rtol": RTOL, "atol": ATOL},
                **meta, "arrays": names, "scalars": scalars}
    js = out_dir / f"{stem}.json"
    js.write_text(json.dumps(manifest, indent=1, ensure_ascii=False, sort_keys=False) + "\n",
                  encoding="utf-8")
    return npz, js


def load(out_dir: "str | Path", stem: str) -> "tuple[dict, dict]":
    """(flat outputs, manifest) written by ``save``."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / f"{stem}.json").read_text(encoding="utf-8"))
    flat: dict = {k: _from_json(v) for k, v in manifest["scalars"].items()}
    with np.load(out_dir / f"{stem}.npz", allow_pickle=False) as z:
        for name, key in manifest["arrays"].items():
            flat[name] = z[key]
    return flat, manifest


# ════════════════════════════════════════════════════════════
#  Comparison
# ════════════════════════════════════════════════════════════

@dataclass
class Difference:
    name: str
    kind: str        # missing | extra | type | shape | value
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.name}: {self.detail}"


def _float_close(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> np.ndarray:
    both_nan = np.isnan(a) & np.isnan(b)
    same_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    with np.errstate(invalid="ignore"):
        close = np.abs(a - b) <= atol + rtol * np.abs(b)
    return both_nan | same_inf | np.nan_to_num(close, nan=False)


def _compare_value(name: str, ref: Any, new: Any, rtol: float, atol: float) -> "Optional[Difference]":
    if isinstance(ref, np.ndarray) or isinstance(new, np.ndarray):
        if not (isinstance(ref, np.ndarray) and isinstance(new, np.ndarray)):
            return Difference(name, "type", f"{type(ref).__name__} -> {type(new).__name__}")
        if ref.shape != new.shape:
            return Difference(name, "shape", f"{ref.shape} -> {new.shape}")
        rk, nk = ref.dtype.kind, new.dtype.kind
        if rk in "US" or nk in "US":
            if rk not in "US" or nk not in "US":
                return Difference(name, "type", f"{ref.dtype} -> {new.dtype}")
            bad = ref != new
            if bad.any():
                i = int(np.flatnonzero(bad.ravel())[0])
                return Difference(name, "value", f"{int(bad.sum())} differ; first at "
                                  f"{i}: {ref.ravel()[i]!r} -> {new.ravel()[i]!r}")
            return None
        if (rk in "iub") != (nk in "iub"):
            return Difference(name, "type", f"{ref.dtype} -> {new.dtype}")
        if rk in "iub":
            bad = ref.astype(np.int64) != new.astype(np.int64)
        else:
            bad = ~_float_close(ref.astype(np.float64), new.astype(np.float64), rtol, atol)
        if bad.any():
            i = int(np.flatnonzero(bad.ravel())[0])
            a, b = ref.ravel()[i], new.ravel()[i]
            return Difference(name, "value", f"{int(bad.sum())}/{bad.size} differ; first at "
                              f"index {i}: {a!r} -> {b!r}")
        return None
    if isinstance(ref, bool) or isinstance(new, bool) or isinstance(ref, str) or isinstance(new, str) \
            or ref is None or new is None:
        return None if ref == new else Difference(name, "value", f"{ref!r} -> {new!r}")
    if isinstance(ref, (int, float)) and isinstance(new, (int, float)):
        if isinstance(ref, int) and isinstance(new, int):
            return None if ref == new else Difference(name, "value", f"{ref!r} -> {new!r}")
        ok = _float_close(np.array([float(ref)]), np.array([float(new)]), rtol, atol)[0]
        return None if ok else Difference(name, "value", f"{ref!r} -> {new!r}")
    return None if ref == new else Difference(name, "value", f"{ref!r} -> {new!r}")


def is_signal_unit(name: str) -> bool:
    """True for a value in units of the z-scored signal (SIGNAL_UNIT_NAMES)."""
    return any(part in SIGNAL_UNIT_NAMES for part in name.split("/"))


def package_changes(manifest: dict) -> "dict[str, tuple]":
    """{package: (reference version, installed version)} for the
    SCIENTIFIC_PACKAGES whose installed version is not the reference's."""
    ref = (manifest.get("environment") or {}).get("packages") or {}
    now = environment()["packages"]
    return {p: (ref.get(p), now.get(p)) for p in SCIENTIFIC_PACKAGES if ref.get(p) != now.get(p)}


def compare(ref: dict, new: dict, rtol: float = RTOL, atol: float = ATOL) -> "list[Difference]":
    """Every difference between the reference outputs *ref* and *new*."""
    diffs: "list[Difference]" = []
    for name in sorted(set(ref) - set(new)):
        diffs.append(Difference(name, "missing", "in the reference, not computed any more"))
    for name in sorted(set(new) - set(ref)):
        diffs.append(Difference(name, "extra", "computed now, not in the reference"))
    for name in sorted(set(ref) & set(new)):
        a = max(atol, SIGNAL_ATOL) if is_signal_unit(name) else atol
        d = _compare_value(name, ref[name], new[name], rtol, a)
        if d is not None:
            diffs.append(d)
    return diffs


def reference_stem(recording: "str | Path") -> str:
    return Path(recording).stem


def record(recording: "str | Path", out_dir: "str | Path", channel: str = "ECG",
           log: "Optional[Callable[[str], None]]" = None) -> "tuple[Path, Path]":
    """Run the pipeline on *recording* and save its reference outputs."""
    outputs, timings = run_pipeline(recording, channel=channel, log=log)
    flat = flatten(outputs)
    meta = {"recording": {"name": Path(recording).name, "sha256": file_sha256(recording),
                          "size": Path(recording).stat().st_size, "channel": channel},
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": environment(), "timings_s": timings}
    return save(flat, out_dir, reference_stem(recording), meta)


def check(recording: "str | Path", ref_dir: "str | Path", channel: "str | None" = None,
          log: "Optional[Callable[[str], None]]" = None) -> "tuple[list[Difference], dict]":
    """Re-run the pipeline on *recording* and compare it with its reference.

    Returns (differences, manifest of the reference). Raises ValueError when
    the recording is not the file the reference was made from."""
    ref, manifest = load(ref_dir, reference_stem(recording))
    sha = file_sha256(recording)
    if sha != manifest["recording"]["sha256"]:
        raise ValueError(f"{Path(recording).name}: not the file the reference was made from "
                         f"(SHA-256 {sha[:12]}… vs {manifest['recording']['sha256'][:12]}…)")
    outputs, _ = run_pipeline(recording, channel=channel or manifest["recording"]["channel"],
                              log=log)
    tol = manifest.get("tolerance", {})
    return compare(ref, flatten(outputs), tol.get("rtol", RTOL), tol.get("atol", ATOL)), manifest
