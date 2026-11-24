# -*- coding: utf-8 -*-
"""
ecg.qt.tabs.detection_tab
-------------------------
Signal & Detection: choose the detector and the filtering, Detect Peaks,
adjust the threshold, see the trace and its peaks.

The computations are the app's own: ecg.services.detection_service.
load_and_detect (in a background job) and ecg.core.peak_editing.
apply_peak_edits (threshold changes, on the main thread, debounced as in
the Tk interface). The tab writes the results into the window's Recording
(ecg.core.recording); the plot only displays them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtWidgets

from ecg.core.models import FilterParams, MouseECG
from ecg.core.peak_editing import (apply_click_edit, apply_peak_edits, edit_at_click,
                                   push_undo, signal_quality_score, undo_redo)
from ecg.qt.widgets.ecg_plot import EcgPlot
from ecg.services.detection_service import load_and_detect

if TYPE_CHECKING:
    from ecg.qt.main_window import MainWindow

# Method box: label -> FilterParams.detection_method. The Tk interface
# starts on SG + Derivative; so does this one.
METHODS = (("Auto (NeuroKit2)", "auto"), ("SG + Derivative (10 kHz)", "sg_derivative"),
           ("Wavelet (CWT)", "wavelet"), ("Envelope Max", "envelope_max"),
           ("ML Detector", "ml"))
DEFAULT_METHOD = "sg_derivative"
THRESH_MIN, THRESH_MAX, THRESH_DEFAULT = 0.01, 2.0, 0.50   # Tk slider range and start
THRESH_DEBOUNCE_MS = 150
BREAK_REDETECT_TOL_MS = 5.0     # AnalysisController.BREAK_REDETECT_TOL_MS


class DetectionTab(QtWidgets.QWidget):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.win = window
        lay = QtWidgets.QVBoxLayout(self)

        bar = QtWidgets.QFrame(objectName="Card")
        row = QtWidgets.QHBoxLayout(bar)
        row.addWidget(QtWidgets.QLabel("Method"))
        self.cb_method = QtWidgets.QComboBox()
        for label, key in METHODS:
            self.cb_method.addItem(label, key)
        self.cb_method.setCurrentIndex([k for _l, k in METHODS].index(DEFAULT_METHOD))
        row.addWidget(self.cb_method)
        self.chk_filter = QtWidgets.QCheckBox("Filtering")
        self.chk_filter.setToolTip("Band-pass, optional notch and NeuroKit2 ecg_clean before "
                                   "detection (off: detection on the raw signal)")
        row.addWidget(self.chk_filter)
        self.btn_detect = QtWidgets.QPushButton("Detect Peaks")
        self.btn_detect.setProperty("primary", True)
        self.btn_detect.setEnabled(False)
        self.btn_detect.clicked.connect(self.detect)
        row.addWidget(self.btn_detect)
        row.addSpacing(16)
        row.addWidget(QtWidgets.QLabel("Threshold"))
        self.sl_thr = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_thr.setRange(int(THRESH_MIN * 100), int(THRESH_MAX * 100))
        self.sl_thr.setValue(int(THRESH_DEFAULT * 100))
        self.sl_thr.setMinimumWidth(180)
        row.addWidget(self.sl_thr, 1)
        self.sp_thr = QtWidgets.QDoubleSpinBox()
        self.sp_thr.setRange(THRESH_MIN, THRESH_MAX)
        self.sp_thr.setSingleStep(0.01)
        self.sp_thr.setDecimals(2)
        self.sp_thr.setValue(THRESH_DEFAULT)
        row.addWidget(self.sp_thr)
        self.lbl_thr = QtWidgets.QLabel("of the median R height")
        self.lbl_thr.setProperty("muted", True)
        row.addWidget(self.lbl_thr)
        row.addSpacing(16)
        self.lbl_peaks = QtWidgets.QLabel("Peaks: —")
        row.addWidget(self.lbl_peaks)
        self.lbl_quality = QtWidgets.QLabel("Quality: —")
        row.addWidget(self.lbl_quality)
        lay.addWidget(bar)

        edit_bar = QtWidgets.QHBoxLayout()
        self.btn_edit = QtWidgets.QPushButton("Edit Peaks")
        self.btn_edit.setCheckable(True)
        self.btn_edit.setEnabled(False)
        self.btn_edit.setToolTip("Left click near a peak: exclude / include it. Right click: add a "
                                 "peak at the local maximum (replacing a peak too close), or remove "
                                 "a peak added by hand.")
        self.btn_edit.toggled.connect(self._edit_toggled)
        edit_bar.addWidget(self.btn_edit)
        self.chk_free = QtWidgets.QCheckBox("Free placement")
        self.chk_free.setToolTip("Right click adds the peak exactly where clicked (no snapping, "
                                 "no replacement)")
        edit_bar.addWidget(self.chk_free)
        self.btn_undo = QtWidgets.QPushButton("Undo")
        self.btn_redo = QtWidgets.QPushButton("Redo")
        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo.clicked.connect(self.redo)
        edit_bar.addWidget(self.btn_undo)
        edit_bar.addWidget(self.btn_redo)
        self.lbl_edit = QtWidgets.QLabel("")
        self.lbl_edit.setProperty("muted", True)
        edit_bar.addWidget(self.lbl_edit, 1)
        self.btn_mark_noise = QtWidgets.QPushButton("Mark Noise")
        self.btn_mark_noise.setCheckable(True)
        self.btn_mark_noise.setEnabled(False)
        self.btn_mark_noise.setToolTip("Drag over a noisy stretch to leave its RR intervals out "
                                       "of HR / HRV; right click a noisy region to remove it")
        self.btn_mark_noise.toggled.connect(self._mark_noise_toggled)
        edit_bar.addWidget(self.btn_mark_noise)
        self.btn_annotations = QtWidgets.QPushButton("Annotations")
        self.btn_annotations.setToolTip("Labelled periods of the recording (drug, stimulus, "
                                        "sleep...): drawn on the traces and the time course")
        self.btn_annotations.clicked.connect(lambda: self.open_periods("annotation"))
        edit_bar.addWidget(self.btn_annotations)
        self.btn_noise = QtWidgets.QPushButton("Noisy regions")
        self.btn_noise.setToolTip("Stretches too noisy to trust: their RR intervals are left "
                                  "out of HR / HRV at the next Analyze")
        self.btn_noise.clicked.connect(lambda: self.open_periods("noise"))
        edit_bar.addWidget(self.btn_noise)
        lay.addLayout(edit_bar)

        self.plot = EcgPlot()
        self.plot.time_clicked.connect(self._clicked)
        self.plot.span_drawn.connect(self._noise_drawn)
        lay.addWidget(self.plot, 1)
        self._update_undo_buttons()

        self._thr_timer = QtCore.QTimer(self, singleShot=True, interval=THRESH_DEBOUNCE_MS)
        self._thr_timer.timeout.connect(self.apply_threshold)
        self.sl_thr.valueChanged.connect(lambda v: self._set_threshold(v / 100.0, from_slider=True))
        self.sp_thr.valueChanged.connect(lambda v: self._set_threshold(v, from_slider=False))

    # ── parameters ──────────────────────────────────────────
    def threshold(self) -> float:
        return float(self.sp_thr.value())

    def params(self) -> dict:
        """The Detect Peaks snapshot -- SessionController.snapshot_params:
        FilterParams.from_widgets on the Tk widget names the window's
        FigureHost publishes (this tab and the Settings dialog)."""
        host = self.win.figures
        params = FilterParams.from_widgets(host).to_dict()
        params["analysis_t_start"] = self.win.recording.analysis.t_start
        params["analysis_t_end"] = self.win.recording.analysis.t_end
        params["sg_window_ms"] = host._safe_float(getattr(host, "ent_sg_window_ms", None), 20.0)
        return params

    def _set_threshold(self, value: float, from_slider: bool) -> None:
        if from_slider:
            self.sp_thr.blockSignals(True)
            self.sp_thr.setValue(value)
            self.sp_thr.blockSignals(False)
        else:
            self.sl_thr.blockSignals(True)
            self.sl_thr.setValue(int(round(value * 100)))
            self.sl_thr.blockSignals(False)
        self._thr_timer.start()

    # ── Detect Peaks ────────────────────────────────────────
    def file_opened(self) -> None:
        self.btn_detect.setEnabled(True)
        self.btn_edit.setChecked(False)
        self.btn_edit.setEnabled(False)
        self.btn_mark_noise.setChecked(False)
        self.btn_mark_noise.setEnabled(False)
        self._update_undo_buttons()
        self.plot.clear()
        self.lbl_peaks.setText("Peaks: —")
        self.lbl_quality.setText("Quality: —")
        self.update_period_counts()

    def detect(self) -> None:
        path = self.win.recording.signal.filepath
        if not path:
            return
        self.btn_detect.setEnabled(False)
        self.win.run_job("Detect Peaks", load_and_detect, path, self.params(),
                         on_done=self._detected, on_end=lambda: self.btn_detect.setEnabled(True))

    def _detected(self, bundle: dict) -> None:
        """SignalController.on_preview_done's state writes, on the main thread."""
        rec = self.win.recording
        s, d = rec.signal, rec.detection
        fs = bundle["fs"]
        s.fs = fs
        s.raw = bundle["signal_raw"]
        s.raw_norm = bundle["signal_raw_norm"]
        s.filtered = bundle["signal_flt"]
        s.time = np.arange(len(bundle["signal_raw"])) / fs
        s.no_filter_mode = bundle["no_filter_mode"]
        s.filter_text = bundle.get("filter_text")
        s.prepare_params = bundle.get("prepare_params")
        s.crop = bundle.get("signal_crop")
        s.inverted = bundle.get("inverted", False)
        s.channel = bundle.get("detected_ch")
        s.markers = bundle.get("markers") or {}
        d.all_candidates = bundle["all_cands"]
        d.all_prominences = bundle["all_proms"]
        d.rpeaks_ok = bundle["rpeaks_ok"]
        d.rpeaks_rej = bundle["rpeaks_rej"]
        d.thresh_amp = bundle["thresh_amp"]
        d.manual_excluded.clear()
        d.manual_added.clear()
        d.edit_undo.clear()
        d.edit_redo.clear()
        rec.analysis.results = None
        self.btn_edit.setEnabled(True)
        self.btn_mark_noise.setEnabled(True)
        self._update_undo_buttons()
        self.plot.set_signal(s.filtered, fs, s.time)
        self.plot.set_noise_regions(rec.analysis.noise_regions)
        self.plot.set_annotations(rec.analysis.annotations)
        self._show_peaks()
        self.win.peaks_changed()
        warning = bundle.get("detector_warning")
        self.win.set_status(warning or f"{len(d.rpeaks_ok):,} peaks detected · "
                                       f"{s.filter_text or ''}", warn=bool(warning))

    # ── threshold ───────────────────────────────────────────
    def apply_threshold(self) -> None:
        """DetectionController.run_detection: the peaks again from every
        candidate, with the hand edits and artifact removals still in effect."""
        rec = self.win.recording
        s, d = rec.signal, rec.detection
        if s.filtered is None or d.all_candidates is None or d.all_prominences is None:
            return
        arep = rec.analysis.artifact_report
        removed = arep.get("removed_samples") if isinstance(arep, dict) else None
        tol = max(1, int(round(BREAK_REDETECT_TOL_MS / 1000.0 * float(s.fs))))
        edits = apply_peak_edits(s.filtered, d.all_candidates, d.all_prominences,
                                 self.threshold(), s.fs, d.manual_excluded, d.manual_added,
                                 removed_samples=removed, previous_peaks=d.rpeaks_ok, tol=tol)
        d.rpeaks_manual_excl = edits.manual_excl
        d.rpeaks_manual_added = edits.manual_added
        if d.manual_added:
            d.manual_excluded -= d.manual_added
        d.rpeaks_ok, d.rpeaks_rej, d.thresh_amp = edits.accepted, edits.rejected, edits.thresh_amp
        self._show_peaks()
        self.win.peaks_changed()

    # ── editing by hand ─────────────────────────────────────
    def _edit_toggled(self, on: bool) -> None:
        self.btn_edit.setText("Edit Mode ON" if on else "Edit Peaks")
        self.lbl_edit.setText("Left click: exclude / include · right click: add / remove · "
                              "Ctrl+Z / Ctrl+Y" if on else "")

    # ── Mark Noise (DetectionController.toggle_noise_mode / _noise_press /
    #    on_detail_release) ──────────────────────────────────
    NOISE_MIN_S = 0.05          # a shorter drag is a click, not a region

    def _mark_noise_toggled(self, on: bool) -> None:
        self.plot.span_mode = on
        self.btn_mark_noise.setText("Mark Noise ON" if on else "Mark Noise")
        if on:
            if not self.btn_edit.isChecked():
                self.btn_edit.setChecked(True)
            self.win.set_status("Mark Noise ON — drag over a noisy stretch to leave its RR "
                                "intervals out of HR / HRV; right-click a noisy region to "
                                "remove it.", warn=True)
        else:
            self.win.set_status("Mark Noise OFF — clicks edit peaks again.")

    def _noise_drawn(self, a: float, b: float) -> None:
        s = self.win.recording.signal
        if s.fs and s.filtered is not None:
            b = min(b, len(s.filtered) / float(s.fs))
        a = max(a, 0.0)
        if b - a < self.NOISE_MIN_S:
            return
        regions = self.win.recording.analysis.noise_regions
        regions.append({"t_start": round(a, 3), "t_end": round(b, 3), "note": ""})
        regions.sort(key=lambda r: float(r["t_start"]))
        self.win.session_dirty = True
        self.noise_changed()
        self.win.set_status(f"Noisy region {a:.2f}–{b:.2f} s marked: its RR intervals are left "
                            "out of HR / HRV at the next Analyze.")

    def _noise_click(self, t: float, button: str) -> None:
        if button != "right":
            return
        regions = self.win.recording.analysis.noise_regions
        hit = [i for i, r in enumerate(regions) if float(r["t_start"]) <= t <= float(r["t_end"])]
        if hit:
            for i in reversed(hit):
                del regions[i]
            self.win.session_dirty = True
            self.noise_changed()
            self.win.set_status(f"Noisy region at {t:.2f} s removed.")

    def _clicked(self, t: float, button: str) -> None:
        """A click on the trace in edit mode (DetectionController.on_detail_click)."""
        if self.btn_mark_noise.isChecked():
            self._noise_click(t, button)
            return
        if not self.btn_edit.isChecked():
            return
        rec = self.win.recording
        s, d = rec.signal, rec.detection
        if s.filtered is None:
            return
        t0, t1 = self.plot.window()
        edit = edit_at_click(s.filtered, s.fs, t, button, t1 - t0, d.rpeaks_ok,
                             d.rpeaks_manual_excl, d.rpeaks_manual_added,
                             d.manual_excluded, d.manual_added,
                             free_placement=self.chk_free.isChecked())
        if not edit.changes_peaks:
            return                      # not near a peak: nothing changes, results kept
        self._drop_results()
        push_undo(d.edit_undo, d.edit_redo, d.manual_excluded, d.manual_added)
        apply_click_edit(edit, d.manual_excluded, d.manual_added)
        self.apply_threshold()
        self._update_undo_buttons()
        fs, n_ok = s.fs, len(d.rpeaks_ok)
        text = {
            "add_free": f"[Free] Added peak at {(edit.peak or 0) / fs:.3f} s",
            "remove_added": f"Removed manually added peak at {t:.3f} s",
            "replace": f"Replaced peak {(edit.old_peak or 0) / fs:.3f} s → "
                       f"{(edit.peak or 0) / fs:.3f} s ({(edit.distance or 0) / fs * 1000:.1f} ms apart)",
            "add": f"Added peak at {(edit.peak or 0) / fs:.3f} s (snapped to local max)",
        }.get(edit.action, f"Manual exclusions: {len(d.manual_excluded)}")
        self.win.set_status(f"{text}  |  Accepted: {n_ok}  — click Analyze to refresh HRV.",
                            warn=True)

    def undo(self) -> None:
        self._undo_redo(redo=False)

    def redo(self) -> None:
        self._undo_redo(redo=True)

    def _undo_redo(self, redo: bool) -> None:
        d = self.win.recording.detection
        src, dst = (d.edit_redo, d.edit_undo) if redo else (d.edit_undo, d.edit_redo)
        sets = undo_redo(src, dst, d.manual_excluded, d.manual_added)
        if sets is None:
            self.win.set_status("Nothing to redo." if redo else "Nothing to undo.")
            return
        d.manual_excluded, d.manual_added = sets
        self._drop_results()
        self.apply_threshold()
        self._update_undo_buttons()
        self.win.set_status(f"Redone ({len(d.edit_redo)} remaining)" if redo else
                            f"Undone ({len(d.edit_undo) or 'none'} left) — Ctrl+Y to redo",
                            warn=True)

    def _update_undo_buttons(self) -> None:
        d = self.win.recording.detection
        self.btn_undo.setEnabled(bool(d.edit_undo))
        self.btn_redo.setEnabled(bool(d.edit_redo))

    def _drop_results(self) -> None:
        """A peak change makes the analysis results stale: they are dropped
        (DetectionController._invalidate_stale_results)."""
        a = self.win.recording.analysis
        a.results = None
        a.autonomic_result = None

    # ── annotations and noisy regions ───────────────────────
    def update_period_counts(self) -> None:
        a = self.win.recording.analysis
        n_ann, n_noise = len(a.annotations or []), len(a.noise_regions or [])
        self.btn_annotations.setText(f"Annotations ({n_ann})" if n_ann else "Annotations")
        self.btn_noise.setText(f"Noisy regions ({n_noise})" if n_noise else "Noisy regions")

    def open_periods(self, kind: str) -> None:
        from ecg.qt.dialogs.periods_dialog import PeriodsManager
        changed = self.annotations_changed if kind == "annotation" else self.noise_changed
        PeriodsManager(self.win, kind, changed).exec()

    def annotations_changed(self) -> None:
        """ECGApp._update_ann_count: drawn here and on the Autonomic, Epochs
        and Sliding plots (no-op without a result)."""
        self.plot.set_annotations(self.win.recording.analysis.annotations)
        self.update_period_counts()
        ctl = self.win.figures.analysis_ctrl
        ctl.redraw_autonomic_plot()
        ctl.redraw_windowed_plots()
        self.win.figures.event_ctrl.on_annotations_changed()

    def noise_changed(self) -> None:
        """ECGApp._noise_regions_changed: drawn and counted; HR / HRV leave
        their RR intervals out from the next Analyze."""
        self.plot.set_noise_regions(self.win.recording.analysis.noise_regions)
        self.update_period_counts()
        if self.win.recording.analysis.results is not None:
            self.win.set_status("Noisy regions changed — click Analyze to update the results.",
                                warn=True)

    def refresh(self) -> None:
        """Redraw the peaks after the model changed (e.g. Analyze's artifact
        correction)."""
        self._show_peaks()

    def _show_peaks(self) -> None:
        rec = self.win.recording
        s, d = rec.signal, rec.detection
        self.plot.set_peaks(d.rpeaks_ok, d.rpeaks_rej, d.rpeaks_manual_excl, d.rpeaks_manual_added)
        n = 0 if d.rpeaks_ok is None else len(d.rpeaks_ok)
        self.lbl_peaks.setText(f"Peaks: {n:,}")
        beat_corr = (rec.analysis.results or {}).get("beat_corr")
        q = (signal_quality_score(d.rpeaks_ok, s.fs, float(s.time[-1]), beat_corr)
             if d.rpeaks_ok is not None and s.time is not None else None)
        if q is not None:
            d.sig_quality = q
        self.lbl_quality.setText(f"Quality: {q}" if q is not None else "Quality: —")
        self.lbl_thr.setText(f"→ ≈{d.thresh_amp:.2f} norm. of the median R height")

    def apply_palette(self, p) -> None:
        self.plot.apply_palette(p)
        self._show_peaks() if self.win.recording.detection.rpeaks_ok is not None else None


__all__ = ["DetectionTab", "METHODS", "DEFAULT_METHOD", "MouseECG"]
