# -*- coding: utf-8 -*-
"""
ecg.core.recording
------------------
The state of one ECG recording: the signal (SignalState), the R-peak
detection and its edits (DetectionState), and the analysis results
(AnalysisState), grouped by ``Recording``.

Moved unchanged from ecg.ui.state (which re-exports the three classes), so
the services and the next (PySide6) interface use the same model as the
current one. Two changes, without effect on any result:

* ``SignalState.raw`` is stored read-only: the recorded signal can never be
  overwritten (filtering works on copies and writes ``filtered``);
* ``Recording`` groups the three states and states the units.

Units and conventions
    signal          arbitrary units as recorded (raw) / z-scored (raw_norm,
                    filtered: zero mean, unit standard deviation)
    fs              Hz (samples per second)
    time            s, sample i at i / fs from the start of the (cropped)
                    signal
    R-peaks         sample indices (np.ndarray of int), never times: a time
                    is peaks / fs
    RR intervals    ms; heart rate in bpm; analysis window and annotation
                    times in s

Some fields still serve the Tk interface only (hover position, debounce
handles, selected event, view positions); they will move to the UI state of
the PySide6 interface in phase 2 of the migration.

Nothing here may import from ecg.ui.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

import numpy as np
import pandas as pd

from ecg.core.analysis import ENSEMBLE_BEATS_DEFAULT
from ecg.core.models import MouseECG
from ecg.core.wave_template import WaveTemplate


@dataclass
class SignalState:
    """The loaded recording and its filtered/derived arrays."""
    filepath: Optional[str] = None
    raw: Optional[np.ndarray] = None            # raw float64 (after time-crop)
    raw_norm: Optional[np.ndarray] = None       # zero-mean/unit-var raw (display)
    filtered: Optional[np.ndarray] = None       # filtered + normalised
    time: Optional[np.ndarray] = None
    fs: int = MouseECG.FS_DEFAULT
    peak_distance_ms: float = MouseECG.PEAK_DISTANCE_MS
    # Raw-only load state -- file just opened, no filtering/detection run yet.
    # True right after _load_raw_only(); cleared once Preview Detection runs.
    raw_only_loaded: bool = False
    no_filter_mode: bool = True                 # default: detect on raw signal
    # Filters the current filtered signal / peaks were computed with
    # (filtering.filter_description); None = not recorded (raw-only load).
    filter_text: Optional[str] = None
    # FilterParams.to_dict() snapshot the filtered signal / peaks were
    # prepared with (SignalController.compute_preview_bundle) -- saved in
    # sessions so a restore re-prepares the signal the results came from,
    # not widget values edited afterwards. None = not prepared / unknown.
    prepare_params: Optional[dict] = None
    # Load-time crop in file time (analysis.signal_crop_info: t_start,
    # t_end, file_duration_s); all times are relative to its start.
    # None = not recorded (session saved before it was).
    crop: Optional[dict] = None
    inverted: bool = False                      # True if polarity was auto-flipped
    # Channel the loaded signal was read from (load_mat_signal's detected
    # channel). A channel can be another animal: re-detecting on a
    # different channel drops the recording's wave template
    # (SignalController.on_preview_done). None = nothing loaded yet.
    channel: Optional[str] = None
    # Marker / event channels of a Spike2 file ({channel: [times s]}, in the
    # loaded signal's time base: loaders.read_spike2_markers), read at load
    # time; {} for MATLAB files and before a load. The Compare segments
    # "Stim marker" picker lists them.
    markers: "dict[str, list[float]]" = field(default_factory=dict)

    def __setattr__(self, name: str, value: Any) -> None:
        # The recorded signal is never overwritten in place: it is kept as a
        # read-only view (the caller's own array is not changed). Filtering
        # works on copies (ecg.core.pipeline) and writes ``filtered``.
        if name == "raw" and isinstance(value, np.ndarray) and value.flags.writeable:
            value = value.view()
            value.flags.writeable = False
        object.__setattr__(self, name, value)


@dataclass
class DetectionState:
    """R-peak detection results and manual edit/undo state."""
    EDIT_UNDO_LIMIT: ClassVar[int] = 50

    rpeaks_ok: Optional[np.ndarray] = None      # accepted peaks
    rpeaks_rej: Optional[np.ndarray] = None     # rejected candidates
    all_candidates: Optional[np.ndarray] = None  # all candidates (post-polarity fix)
    all_prominences: Optional[np.ndarray] = None  # corresponding prominences
    thresh_amp: float = 0.0
    sig_quality: Optional[int] = None

    # Live per-beat correlation cache for the Detection tab's Quality
    # mini-plot (plot_controller.py's _ensure_beat_corr_cache()) -- computed
    # once per distinct rpeaks_ok via compute_beat_correlation(), not on
    # every redraw. _beat_corr_for holds the exact rpeaks_ok array reference
    # the cache was built from; since rpeaks_ok is always reassigned (never
    # mutated in place) whenever peaks change, an `is` check against it is a
    # correct, self-invalidating cache-validity test.
    beat_corr: Optional[np.ndarray] = None
    beat_corr_peaks: Optional[np.ndarray] = None
    _beat_corr_for: Optional[np.ndarray] = None
    corr_debounce_id: "str | None" = None       # beat-correlation recompute debounce handle

    # Same identity-cache pattern for draw_detail()'s per-peak confidence
    # alpha (plot_controller.py's _ensure_alpha_cache()) -- this used to be
    # rebuilt from all_candidates/all_prominences (whole-recording arrays)
    # with pure-Python dict/list work on every single redraw, including
    # every 30ms mouse-hover tick in Edit mode.
    rp_ok_alpha: Optional[np.ndarray] = None
    _alpha_cache_for: Optional[np.ndarray] = None

    # Manual peak exclusion
    manual_excluded: set[int] = field(default_factory=set)   # sample indices excluded by user
    rpeaks_manual_excl: Optional[np.ndarray] = None           # view-ready array of excl. peaks
    manual_added: set[int] = field(default_factory=set)      # sample indices added by user
    rpeaks_manual_added: Optional[np.ndarray] = None          # view-ready array of added peaks
    edit_mode: bool = False                     # click-to-exclude active
    edit_free_placement: bool = False           # bypass proximity constraint
    edit_noise_mode: bool = False               # drag marks a noisy region
    # Undo/Redo stacks -- each entry: (frozenset excluded, frozenset added)
    edit_undo: list[tuple[frozenset, frozenset]] = field(default_factory=list)
    edit_redo: list[tuple[frozenset, frozenset]] = field(default_factory=list)

    # Hover preview state (edit mode -- shows snapped peak position on mouse move)
    hover_samp: Optional[int] = None            # snapped sample under cursor
    hover_samp_near: bool = False                # True = replaces nearby peak

    # Cross-method agreement (detection_controller.py: check_method_agreement())
    agreement_disagree_samples: Optional[np.ndarray] = None  # accepted peaks 0 other methods confirmed
    agreement_summary: str = ""                  # last run's human-readable summary line
    agreement_pct: Optional[float] = None        # last run's % of beats confirmed by >=1 other method
    agreement_peaks: Optional[np.ndarray] = None  # the peaks that run was on; other peaks = no value


@dataclass
class AnalysisState:
    """HRV/frequency/nonlinear/interval/arrhythmia analysis results."""
    results: Optional[dict] = None
    epoch_df: Optional[pd.DataFrame] = None
    rolling_hrv_df: Optional[pd.DataFrame] = None
    # What Epochs and Sliding were last computed with, to draw those frames
    # again after a UI rebuild or a session restore: {"epochs": {...}, "rolling": {...}}.
    windowed_view: dict = field(default_factory=dict)
    # Time course > Autonomic (core.autonomic, Candia-Rivera et al. 2025): the
    # time-resolved CSI / CPI of the accepted peaks (an AutonomicResult)
    # and, when requested, the SPWVD LF / HF (core.spwvd.spwvd_lf_hf()'s
    # dict, or {"error": reason} when it could not be computed). Derived
    # from the peaks, so invalidated with every other peak-derived result
    # (AnalysisController.invalidate_autonomic); recomputed, not saved.
    autonomic_result: Optional[Any] = None
    autonomic_spwvd: Optional[dict] = None
    # The Autonomic view's parameters (window_s, method, ks, kp, lf_hf,
    # min_pairs) -- app state rather than only widget values, so a UI
    # rebuild (theme toggle) shows them again; saved with the session.
    # Empty = defaults (ecg.ui.autonomic_view.normalise_autonomic_params).
    autonomic_params: dict = field(default_factory=dict)
    # Why the Autonomic / event-locked result was dropped ("Peaks changed —
    # ..."): the status labels show it until the next compute, and a UI
    # rebuild shows it again instead of the idle hint. None = nothing stale.
    autonomic_stale_note: Optional[str] = None
    event_locked_stale: Optional[str] = None
    # Per-event pre/post table derived from autonomic_result
    # (core.event_locked.event_window_means); dropped with it.
    event_locked_df: Optional[pd.DataFrame] = None
    # Time course > Event-locked: the full result behind that table
    # (ecg.ui.event_locked_view.EventLockedResult -- table, traces, the
    # events and configuration it was built from); dropped with
    # autonomic_result and when the annotations it was locked to change.
    event_locked: Optional[Any] = None
    # The Event-locked panel's parameters (label filter, pre_s, post_s,
    # zscore) -- app state like autonomic_params, so a UI rebuild shows
    # them again; saved with the session. Empty = defaults
    # (ecg.ui.event_locked_view.normalise_event_params).
    event_locked_params: dict = field(default_factory=dict)
    arrhythmia_events: list = field(default_factory=list)
    arrhythmia_tsv: str = ""
    # detection.premature_beat_summary() of arrhythmia_events (counts and
    # per-hour rates of the RR-based premature beats); None before a run.
    premature_summary: Optional[dict] = None
    # What arrhythmia_events were classified from: {"peaks": the windowed
    # peaks, "removed": the removed beats put back} -- compared with the
    # current ones (AnalysisController.arrhythmia_is_current) so exports
    # never write counts from an earlier peak set or analysis window.
    arrhythmia_basis: Optional[dict] = None
    # How they were classified, to draw them again and say what they rest on:
    # {"baseline": arrhythmia_baseline_info(), "brady_pct", "av_note", "av_on",
    # "peaks", "fs"}; None before a run.
    arrhythmia_run: Optional[dict] = None
    arr_selected_idx: int = -1                  # index into arrhythmia_events
    arr_nav_pos: float = 0.0                    # current t_start in arrhythmia ECG view
    arr_win: float = 3.0                        # seconds shown in arrhythmia ECG view
    arr_edit_mode: bool = False                 # edit mode specific to arrhythmia tab
    last_seg_a: Optional[dict] = None           # last Compare A result
    last_seg_b: Optional[dict] = None           # last Compare B result
    # The last Compare segments run (ecg.io.segment_export.ComparisonResult)
    # and the recording state it rests on (CompareSegmentsMixin.
    # _compare_signature): the main exports include it only while current.
    last_comparison: Any = None
    last_comparison_basis: Optional[tuple] = None
    # The windows and block length of that run ({"rows": [[label, lo, hi]...],
    # "block_s"}): a theme rebuild re-shows the kept result instead of
    # recomputing it when they match (compare_segments._apply_setup).
    last_comparison_run: Optional[dict] = None
    # Cumulative artifact report (merge_artifact_reports): "removed_samples"
    # / "removed_types" are every removal still in effect (re-applied after
    # a session restore re-detects the peaks), "break_samples" the subset
    # that removed a real beat -- the HRV breaks every HRV path passes on
    # (AnalysisController.hrv_breaks). Persisted with the session.
    artifact_report: Optional[dict] = None
    # Full edited candidate list from the last Artifact Review pass (schema:
    # {index, sample, type, rr_prev_ms, rr_next_ms, rr_ref_ms, deviation,
    # decision}) -- kept so draw_detail() can render former peak positions
    # that were removed. Overwritten (not appended) on every review, exactly
    # like artifact_report above; only the latest pass's markers are shown.
    artifact_candidates: list[dict] = field(default_factory=list)

    # Analysis window (independent from signal crop) -- restricts HRV analysis
    # to a sub-range of the already-detected signal. 0.0 = "use full signal".
    t_start: float = 0.0
    t_end: float = 0.0
    # (t_start, t_end) the current results were computed on -- t_start/t_end
    # above can be edited afterwards; the result labels describe these.
    analysed_window: Optional[tuple] = None

    # Time annotations: list of dicts {t_start, t_end, label, color}
    annotations: list[dict] = field(default_factory=list)
    # Manually-marked noisy regions: {t_start, t_end, note}.
    # Deliberately a separate list from `annotations` above (per product
    # decision) -- fixed render color, no per-item color choice.
    noise_regions: list[dict] = field(default_factory=list)
    wave_template: Optional[WaveTemplate] = None
    # QTc formula key (QTC_FORMULA_LABELS) the stored intervals' QTc_ms
    # column is computed with -- the Intervals-tab selector's value, kept
    # here rather than only on the widget so it survives a UI rebuild
    # (theme toggle), which recreates the combobox at its default. Read by
    # the exports and session save; set by the selector, Intervals and
    # session restore.
    qtc_formula: str = "mitchell"
    # N of the rolling-median delineation (analyse_intervals, ensemble_beats):
    # the Morphology & Intervals control's value, kept here for the same
    # reason. 1 = per beat. What a frame was measured with is in its attrs.
    ensemble_beats: int = ENSEMBLE_BEATS_DEFAULT
    # What each module's result was computed from (ecg.ui.freshness):
    # module -> {peaks, removed, window, rr}, captured at job start. Derived
    # again on a session restore, not saved. Empty = nothing computed yet.
    basis: dict = field(default_factory=dict)


@dataclass
class Recording:
    """One recording: its signal, detection and analysis states."""
    signal: SignalState = field(default_factory=SignalState)
    detection: DetectionState = field(default_factory=DetectionState)
    analysis: AnalysisState = field(default_factory=AnalysisState)

    @property
    def n_samples(self) -> int:
        """Samples of the (cropped) signal, 0 before loading."""
        return 0 if self.signal.raw is None else len(self.signal.raw)

    @property
    def duration_s(self) -> float:
        """Duration of the (cropped) signal, s."""
        return self.n_samples / float(self.signal.fs) if self.signal.fs else 0.0

    def time_axis(self) -> np.ndarray:
        """Time of every sample, s (i / fs)."""
        return np.arange(self.n_samples) / float(self.signal.fs)

    def peak_times_s(self, peaks: "Optional[np.ndarray]" = None) -> np.ndarray:
        """Times (s) of *peaks* (sample indices; default: the accepted peaks)."""
        p = self.detection.rpeaks_ok if peaks is None else peaks
        if p is None:
            return np.array([], dtype=float)
        return np.asarray(p) / float(self.signal.fs)

    def rr_ms(self, peaks: "Optional[np.ndarray]" = None) -> np.ndarray:
        """Every RR interval of *peaks* (default: the accepted peaks), ms --
        all of them, valid or not (rr_interval_series sorts them)."""
        p = self.detection.rpeaks_ok if peaks is None else peaks
        if p is None or len(p) < 2:
            return np.array([], dtype=float)
        return np.diff(np.asarray(p)).astype(float) / float(self.signal.fs) * 1000.0
