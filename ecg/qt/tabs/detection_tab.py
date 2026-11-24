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
        lay.addLayout(edit_bar)

        self.plot = EcgPlot()
        self.plot.time_clicked.connect(self._clicked)
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
        """The Detect Peaks snapshot (as SessionController.snapshot_params)."""
        fp = FilterParams(detection_method=self.cb_method.currentData(),
                          no_filter=not self.chk_filter.isChecked(),
                          thresh=self.threshold())
        params = fp.to_dict()
        params["analysis_t_start"] = self.win.recording.analysis.t_start
        params["analysis_t_end"] = self.win.recording.analysis.t_end
        params["sg_window_ms"] = 20.0
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
        self._update_undo_buttons()
        self.plot.clear()
        self.lbl_peaks.setText("Peaks: —")
        self.lbl_quality.setText("Quality: —")

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
        self._update_undo_buttons()
        self.plot.set_signal(s.filtered, fs, s.time)
        self.plot.set_noise_regions(rec.analysis.noise_regions)
        self._show_peaks()
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

    # ── editing by hand ─────────────────────────────────────
    def _edit_toggled(self, on: bool) -> None:
        self.btn_edit.setText("Edit Mode ON" if on else "Edit Peaks")
        self.lbl_edit.setText("Left click: exclude / include · right click: add / remove · "
                              "Ctrl+Z / Ctrl+Y" if on else "")

    def _clicked(self, t: float, button: str) -> None:
        """A click on the trace in edit mode (DetectionController.on_detail_click)."""
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
