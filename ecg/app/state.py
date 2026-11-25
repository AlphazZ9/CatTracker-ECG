# -*- coding: utf-8 -*-
"""
state.py
--------
Typed state containers for ECGApp, grouped by responsibility. SignalState,
DetectionState and AnalysisState are defined in ecg.core.recording (UI-free);
UIState and SessionState, the interface's own, stay here.

These dataclasses hold every piece of non-widget instance state that used to
live as scattered ``self._xxx`` attributes directly on ``ECGApp``. Widgets
(CTk entries, buttons, labels, canvases, ...) stay directly on ``ECGApp`` --
they are the GUI, not application state.

ECGApp keeps backward-compatible ``@property`` shims (see app.py) for every
renamed attribute so existing method bodies (and the small set of sibling
modules that reach into an ECGApp instance -- dialogs.py, wave_editor.py,
models.py) keep working unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# The recording's state (signal, detection, analysis) lives in ecg.core.recording;
# re-exported here for the existing imports.
from ecg.core.recording import AnalysisState, DetectionState, SignalState  # noqa: F401

# Narrowest time window (s) the ECG views can be zoomed to, by scroll wheel
# or the window entry -- shared by the Detection detail view and the
# Abnormal Events editor so both allow the same zoom depth.
MIN_VIEW_WINDOW_S: float = 0.1


@dataclass
class UIState:
    """Transient view/render state: navigation position, caches, mpl event ids."""
    nav_pos: float = 0.0
    show_raw: bool = True                       # raw/filtered toggle
    rrhr_strip_visible: bool = True              # RR/HR/Quality sub-plot strip toggle
    show_spike_markers: bool = True              # RR/HR tachogram spike-marker toggle
    # (No sidebar / right-panel collapse flags: both panels are gone. A UI
    # snapshot taken before still holds them; restore_ui_state ignores them.)
    # Filter preview (before/after overlay) -- computed on-demand for the
    # currently visible detail window only, never touches filtered signal or
    # any detection state.
    filter_preview_on: bool = False
    thr_debounce_id: "str | None" = None        # slider debounce handle
    hover_motion_cid: Optional[int] = None       # mpl event connection id
    hover_after_id: "str | None" = None          # motion debounce handle
    ov_drag_after_id: "str | None" = None        # minimap drag-to-scrub debounce handle
    scroll_sync_after_id: "str | None" = None    # detail scroll-zoom resync debounce handle
    nav_key_after_id: "str | None" = None        # arrow-key repeat redraw debounce handle
    arr_scroll_after_id: "str | None" = None     # Events scroll-zoom resync debounce handle
    beat_nav_cid: Optional[int] = None           # mpl event ID for beat navigator
    rr_click_cid: Optional[int] = None           # mpl event connection id for RR click-to-navigate
    rr_spike_times: Optional[np.ndarray] = None  # cached spike times from the last plot_rr(), for the Detection-tab stepper
    # Time course tab: the mode shown, kept across a UI rebuild and saved with
    # the session. Replaces hrv_current_view and autonomic_subview, which a
    # session may still hold (ecg.app.tab_keys.time_course_mode).
    time_course_mode: str = "Epochs"
    # Morphology & Intervals: distribution plots open; a UI rebuild keeps it.
    morph_dist_open: bool = False
    # Signal quality panel: the ML DETECTOR section open (collapsed by default).
    ml_section_open: bool = False
    # When each module's result last landed (ISO strings; the out-of-date banner).
    module_computed_at: dict = field(default_factory=dict)
    # Report: Include ticks that differ from the default, as {line: [ticked, the
    # line's state when ticked]}: a tick holds until that state changes.
    report_include: dict = field(default_factory=dict)
    # Report: the text report open under the table; a UI rebuild keeps it.
    report_text_open: bool = False
    # Rhythm & HRV: human and raw NeuroKit metrics shown, kept across a rebuild.
    hrv_show_all: bool = False
    # The modules a job is running or about to run on (the computing state of
    # the tab bar); ticks without a job, after which a chain that stopped
    # (refused, failed) no longer counts as running.
    running_modules: set = field(default_factory=set)
    running_idle_ticks: int = 0
    # Shown by the context strip (ISO strings, saved with the session): when
    # core results last landed, when the peaks were last edited by hand.
    computed_at: Optional[str] = None
    peaks_edited_at: Optional[str] = None
    ctx_keys: list = field(default_factory=list)
    operation_start_time: Optional[float] = None
    figure_cache: dict = field(default_factory=dict)
    # TSV clipboard store: keyed by widget id, value = TSV string
    tsv_store: "dict[int, str]" = field(default_factory=dict)


@dataclass
class SessionState:
    """Session persistence (.ecgsession) and recent-file bookkeeping."""
    recent_files: list[str] = field(default_factory=list)
    dirty: bool = False
    recording_notes: str = ""                    # freetext notes for this recording
    verified_for_training: bool = False           # the recording is an ML training example (cached corrected R-peaks)
