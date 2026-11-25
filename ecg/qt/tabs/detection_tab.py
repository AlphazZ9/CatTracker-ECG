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
WINDOW_PRESETS_S = (1.0, 2.0, 5.0, 10.0)   # the view's lengths offered as one click
NARROW_PX = 1000                            # below this page width, a more compact bar


class DetectionTab(QtWidgets.QWidget):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.win = window
        from ecg.qt.widgets.shell import Segmented
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(24, 16, 24, 10)
        lay.setSpacing(12)

        # ── title: the page, what it shows, and the view's length ──
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(12)
        titles = QtWidgets.QVBoxLayout()
        titles.setSpacing(0)
        title = QtWidgets.QLabel("Signal")
        title.setProperty("role", "title")
        titles.addWidget(title)
        self.lbl_summary = QtWidgets.QLabel("Open a recording, then detect its peaks.")
        self.lbl_summary.setProperty("muted", True)
        titles.addWidget(self.lbl_summary)
        top.addLayout(titles, 1)
        self.seg_window = Segmented([f"{w:g} s" for w in WINDOW_PRESETS_S], exclusive=False)
        self.seg_window.setToolTip("Length of the view")
        for b, w in zip(self.seg_window.buttons, WINDOW_PRESETS_S):
            b.clicked.connect(lambda _c=False, w=w: self.set_view_width(w))
        self.seg_trace = Segmented(["Filtered", "Raw"])
        self.seg_trace.setToolTip("The trace shown: the filtered signal the peaks were found "
                                  "on, or the raw one (display only)")
        self.seg_trace.buttons[0].setChecked(True)
        for b in self.seg_trace.buttons:
            b.clicked.connect(lambda _c=False: self._show_trace())
        top.addWidget(self.seg_trace, 0, QtCore.Qt.AlignmentFlag.AlignBottom)
        top.addWidget(self.seg_window, 0, QtCore.Qt.AlignmentFlag.AlignBottom)
        # The detection settings in a sheet under their button: the trace
        # keeps the page's width. Detect peaks is also the header's button.
        self.btn_detection = QtWidgets.QToolButton()
        self.btn_detection.setText("Detection…")
        self.btn_detection.setToolTip("Method, filtering and threshold of the R-peak detection")
        self.btn_detection.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        det_menu = QtWidgets.QMenu(self.btn_detection)
        self.btn_detection.setMenu(det_menu)
        top.addWidget(self.btn_detection, 0, QtCore.Qt.AlignmentFlag.AlignBottom)
        lay.addLayout(top)

        # ── detection: method, filtering, threshold, Detect peaks ──
        bar = QtWidgets.QFrame()
        bar.setMinimumWidth(640)
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(14, 10, 14, 10)
        row.setSpacing(8)
        self.lbl_method = QtWidgets.QLabel("Method")
        row.addWidget(self.lbl_method)
        self.cb_method = QtWidgets.QComboBox()
        self.cb_method.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cb_method.setMinimumContentsLength(14)
        for label, key in METHODS:
            self.cb_method.addItem(label, key)
        self.cb_method.setCurrentIndex([k for _l, k in METHODS].index(DEFAULT_METHOD))
        row.addWidget(self.cb_method)
        self.chk_filter = QtWidgets.QCheckBox("Filtering")
        self.chk_filter.setToolTip("Band-pass, optional notch and NeuroKit2 ecg_clean before "
                                   "detection (off: detection on the raw signal)")
        row.addWidget(self.chk_filter)
        row.addSpacing(10)
        lbl = QtWidgets.QLabel("Threshold")
        lbl.setToolTip("Fraction of the median R-peak height a candidate must reach")
        row.addWidget(lbl)
        self.sl_thr = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_thr.setRange(int(THRESH_MIN * 100), int(THRESH_MAX * 100))
        self.sl_thr.setValue(int(THRESH_DEFAULT * 100))
        self.sl_thr.setMinimumWidth(140)
        row.addWidget(self.sl_thr, 1)
        self.sp_thr = QtWidgets.QDoubleSpinBox()
        self.sp_thr.setRange(THRESH_MIN, THRESH_MAX)
        self.sp_thr.setSingleStep(0.01)
        self.sp_thr.setDecimals(2)
        self.sp_thr.setValue(THRESH_DEFAULT)
        row.addWidget(self.sp_thr)
        self.lbl_thr = QtWidgets.QLabel("× median R")
        self.lbl_thr.setProperty("muted", True)
        row.addWidget(self.lbl_thr)
        row.addSpacing(10)
        self.btn_detect = QtWidgets.QPushButton("Detect peaks")
        self.btn_detect.setProperty("action", True)
        self.btn_detect.setToolTip("Find the R peaks with the method and filtering above (Ctrl+D)")
        self.btn_detect.setEnabled(False)
        self.btn_detect.clicked.connect(self.detect)
        row.addWidget(self.btn_detect)
        self.btn_agreement = QtWidgets.QPushButton("Check agreement")
        self.btn_agreement.setEnabled(False)
        self.btn_agreement.setToolTip("Run the other detectors on the same filtered signal and "
                                      "circle the accepted beats none of them found")
        self.btn_agreement.clicked.connect(self.check_agreement)
        row.addWidget(self.btn_agreement)
        wa = QtWidgets.QWidgetAction(det_menu)
        wa.setDefaultWidget(bar)
        det_menu.addAction(wa)
        # Read in the summary line and the status row rather than on the bar.
        self.lbl_peaks = QtWidgets.QLabel("Peaks: —")
        self.lbl_quality = QtWidgets.QLabel("Quality: —")
        self.lbl_agreement = QtWidgets.QLabel("")

        # ── the ECG, with its tools ──
        ecg_card = QtWidgets.QFrame(objectName="Card")
        ecg_lay = QtWidgets.QVBoxLayout(ecg_card)
        ecg_lay.setContentsMargins(14, 10, 14, 6)
        ecg_lay.setSpacing(6)
        tools = QtWidgets.QHBoxLayout()
        tools.setSpacing(8)
        head = QtWidgets.QLabel("ECG")
        head.setProperty("role", "heading")
        tools.addWidget(head)
        self.lbl_view = QtWidgets.QLabel("")
        self.lbl_view.setProperty("muted", True)
        tools.addWidget(self.lbl_view)
        tools.addSpacing(10)
        self.legend = QtWidgets.QHBoxLayout()
        self.legend.setSpacing(12)
        self._legend_icons = []
        for kind, text in (("peak", "R peak"), ("excluded", "Excluded"), ("check", "To check"),
                           ("noise", "Noisy")):
            item = QtWidgets.QHBoxLayout()
            item.setSpacing(5)
            icon = QtWidgets.QLabel()
            self._legend_icons.append((kind, icon))
            item.addWidget(icon)
            lbl = QtWidgets.QLabel(text)
            lbl.setProperty("muted", True)
            item.addWidget(lbl)
            self.legend.addLayout(item)
        tools.addLayout(self.legend)
        tools.addStretch(1)
        # Move / Edit peaks / Mark noise: one mode at a time.
        self.seg_tools = Segmented([], exclusive=False)
        self.btn_move = QtWidgets.QPushButton("Move")
        self.btn_move.setCheckable(True)
        self.btn_move.setChecked(True)
        self.btn_move.setToolTip("Drag to pan, wheel to zoom the time axis")
        self.btn_edit = QtWidgets.QPushButton("Edit peaks")
        self.btn_edit.setCheckable(True)
        self.btn_edit.setEnabled(False)
        self.btn_edit.setToolTip("Left click near a peak: exclude / include it. Right click: add a "
                                 "peak at the local maximum (replacing a peak too close), or remove "
                                 "a peak added by hand.")
        self.btn_mark_noise = QtWidgets.QPushButton("Mark noise")
        self.btn_mark_noise.setCheckable(True)
        self.btn_mark_noise.setEnabled(False)
        self.btn_mark_noise.setToolTip("Drag over a noisy stretch to leave its RR intervals out "
                                       "of HR / HRV; right click a noisy region to remove it")
        for b in (self.btn_move, self.btn_edit, self.btn_mark_noise):
            self.seg_tools.add_button(b, tool=True)
        self.btn_move.clicked.connect(self._move_mode)
        self.btn_edit.toggled.connect(self._edit_toggled)
        self.btn_mark_noise.toggled.connect(self._mark_noise_toggled)
        tools.addWidget(self.seg_tools)
        self.chk_free = QtWidgets.QCheckBox("Free placement")
        self.chk_free.setToolTip("Right click adds the peak exactly where clicked (no snapping, "
                                 "no replacement)")
        self.chk_free.setVisible(False)
        tools.addWidget(self.chk_free)
        self.btn_undo = QtWidgets.QPushButton("Undo")
        self.btn_redo = QtWidgets.QPushButton("Redo")
        self.btn_undo.setToolTip("Undo the last peak edit (Ctrl+Z)")
        self.btn_redo.setToolTip("Redo (Ctrl+Y)")
        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo.clicked.connect(self.redo)
        tools.addWidget(self.btn_undo)
        tools.addWidget(self.btn_redo)
        self.btn_strip = QtWidgets.QPushButton("RR · HR · Quality")
        self.btn_strip.setCheckable(True)
        self.btn_strip.setToolTip("The RR, heart rate and beat quality (template correlation, "
                                  "0.90 dashed) under the trace")
        tools.addWidget(self.btn_strip)
        # Less frequent: in a menu.
        self.btn_more = QtWidgets.QToolButton()
        self.btn_more.setText("More")
        self.btn_more.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_more.setToolTip("Clear edits, artifacts, annotations, noisy regions, go to a time")
        more = QtWidgets.QMenu(self.btn_more)
        self.btn_more.setMenu(more)
        tools.addWidget(self.btn_more)
        ecg_lay.addLayout(tools)
        self.lbl_edit = QtWidgets.QLabel("")
        self.lbl_edit.setProperty("muted", True)
        self.lbl_edit.setVisible(False)
        ecg_lay.addWidget(self.lbl_edit)
        self.plot = EcgPlot()
        self.plot.time_clicked.connect(self._clicked)
        self.plot.span_drawn.connect(self._noise_drawn)
        self.plot.window_changed.connect(self._view_moved)
        ecg_lay.addWidget(self.plot, 1)
        lay.addWidget(ecg_card, 1)

        # The menu's entries: kept as buttons too (the same commands).
        self.btn_clear_edits = QtWidgets.QPushButton("Clear edits")
        self.btn_clear_edits.setToolTip("Re-include every peak excluded by hand and remove every "
                                        "peak added by hand (Undo brings them back)")
        self.btn_clear_edits.clicked.connect(self.clear_edits)
        self.btn_review = QtWidgets.QPushButton("Review artifacts…")
        self.btn_review.setEnabled(False)
        self.btn_review.setToolTip("Find the RR artifacts (non-physiological, ectopic, "
                                   "duplicate) and decide for each: keep or remove")
        self.btn_review.clicked.connect(self._review)
        self.btn_annotations = QtWidgets.QPushButton("Annotations")
        self.btn_annotations.setToolTip("Labelled periods of the recording (drug, stimulus, "
                                        "sleep...): drawn on the traces and the time course")
        self.btn_annotations.clicked.connect(lambda: self.open_periods("annotation"))
        self.btn_noise = QtWidgets.QPushButton("Noisy regions")
        self.btn_noise.setToolTip("Stretches too noisy to trust: their RR intervals are left "
                                  "out of HR / HRV at the next Analyze")
        self.btn_noise.clicked.connect(lambda: self.open_periods("noise"))
        for b in (self.btn_clear_edits, self.btn_review, self.btn_annotations, self.btn_noise):
            b.setVisible(False)
            b.setParent(self)
        self._more_actions = {}
        for key, btn in (("clear", self.btn_clear_edits), ("review", self.btn_review),
                         ("annotations", self.btn_annotations), ("noise", self.btn_noise)):
            act = more.addAction(btn.text())
            act.setToolTip(btn.toolTip())
            act.triggered.connect(btn.click)
            self._more_actions[key] = act
        more.addSeparator()
        goto = QtWidgets.QWidget()
        gl = QtWidgets.QHBoxLayout(goto)
        gl.setContentsMargins(12, 4, 12, 4)
        gl.addWidget(QtWidgets.QLabel("Go to (s)"))
        self.sp_goto = QtWidgets.QDoubleSpinBox()
        self.sp_goto.setRange(0.0, 1e7)
        self.sp_goto.setDecimals(3)
        self.sp_goto.setToolTip("Start of the view, in seconds (Enter or Go)")
        self.sp_goto.editingFinished.connect(lambda: self.go_to(self.sp_goto.value()))
        gl.addWidget(self.sp_goto)
        b_go = QtWidgets.QPushButton("Go")
        b_go.clicked.connect(lambda: self.go_to(self.sp_goto.value()))
        gl.addWidget(b_go)
        wa = QtWidgets.QWidgetAction(more)
        wa.setDefaultWidget(goto)
        more.addAction(wa)
        more.aboutToShow.connect(self._sync_more_menu)

        # ── heart rate: the number, and the whole recording as navigator ──
        hr_card = QtWidgets.QFrame(objectName="Card")
        hr_lay = QtWidgets.QHBoxLayout(hr_card)
        hr_lay.setContentsMargins(16, 10, 14, 8)
        hr_lay.setSpacing(18)
        figs = QtWidgets.QVBoxLayout()
        figs.setSpacing(2)
        head = QtWidgets.QLabel("Heart rate")
        head.setProperty("role", "heading")
        figs.addWidget(head)
        big = QtWidgets.QHBoxLayout()
        big.setSpacing(6)
        self.lbl_hr = QtWidgets.QLabel("—")
        self.lbl_hr.setProperty("role", "big")
        big.addWidget(self.lbl_hr, 0, QtCore.Qt.AlignmentFlag.AlignBottom)
        unit = QtWidgets.QLabel("bpm")
        unit.setProperty("role", "unit")
        big.addWidget(unit, 0, QtCore.Qt.AlignmentFlag.AlignBottom)
        big.addStretch(1)
        figs.addLayout(big)
        self.lbl_hr_range = QtWidgets.QLabel("in view")
        self.lbl_hr_range.setProperty("muted", True)
        figs.addWidget(self.lbl_hr_range)
        self.lbl_hr_whole = QtWidgets.QLabel("")
        self.lbl_hr_whole.setProperty("muted", True)
        figs.addWidget(self.lbl_hr_whole)
        figs.addStretch(1)
        side = QtWidgets.QWidget()
        side.setFixedWidth(190)
        side.setLayout(figs)
        hr_lay.addWidget(side)
        hr_lay.addWidget(self.plot.over_view, 1)
        lay.addWidget(hr_card)

        # ── status: beats to check, the analysis window ──
        status = QtWidgets.QHBoxLayout()
        status.setSpacing(8)
        self.lbl_quality.setProperty("muted", True)
        status.addWidget(self.lbl_quality)
        status.addSpacing(14)
        self.lbl_agreement.setProperty("muted", True)
        status.addWidget(self.lbl_agreement)
        status.addSpacing(14)
        self.lbl_to_check = QtWidgets.QLabel("Beats to check")
        self.lbl_to_check.setProperty("muted", True)
        status.addWidget(self.lbl_to_check)
        for text, step, tip in (("‹", -1, "The previous beat flagged on the tachogram, centred"),
                                ("›", +1, "The next beat flagged on the tachogram, centred")):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.setFixedWidth(34)
            b.clicked.connect(lambda _c=False, d=step: self.step_spike(d))
            status.addWidget(b)
        status.addSpacing(14)
        self.btn_review_link = QtWidgets.QPushButton("Review artifacts")
        self.btn_review_link.setProperty("link", True)
        self.btn_review_link.setToolTip(self.btn_review.toolTip())
        self.btn_review_link.clicked.connect(self.btn_review.click)
        self.btn_review_link.setEnabled(False)
        status.addWidget(self.btn_review_link)
        status.addStretch(1)
        lbl = QtWidgets.QLabel("Analysis window:")
        lbl.setProperty("muted", True)
        status.addWidget(lbl)
        self.lbl_window = QtWidgets.QLabel("")
        status.addWidget(self.lbl_window)
        self.btn_window = QtWidgets.QToolButton()
        self.btn_window.setText("Change")
        self.btn_window.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_window.setToolTip("The part of the recording Analyze uses")
        wmenu = QtWidgets.QMenu(self.btn_window)
        self.btn_window.setMenu(wmenu)
        wbox = QtWidgets.QWidget()
        wl = QtWidgets.QHBoxLayout(wbox)
        wl.setContentsMargins(12, 6, 12, 6)
        wl.addWidget(QtWidgets.QLabel("From (s)"))
        self.ed_win0, self.ed_win1 = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        for ed, ph in ((self.ed_win0, "start"), (self.ed_win1, "end")):
            ed.setPlaceholderText(ph)
            ed.setMaximumWidth(80)
            ed.setToolTip("The part of the recording Analyze uses (empty = signal start / "
                          "end); applied at the next Analyze")
            ed.returnPressed.connect(self.apply_analysis_window)
            wl.addWidget(ed)
            if ed is self.ed_win0:
                wl.addWidget(QtWidgets.QLabel("to"))
        b = QtWidgets.QPushButton("Apply")
        b.setProperty("action", True)
        b.clicked.connect(lambda: (self.apply_analysis_window(), wmenu.close()))
        wl.addWidget(b)
        b = QtWidgets.QPushButton("Whole signal")
        b.setToolTip("The whole signal")
        b.clicked.connect(lambda: (self.reset_analysis_window(), wmenu.close()))
        wl.addWidget(b)
        wa = QtWidgets.QWidgetAction(wmenu)
        wa.setDefaultWidget(wbox)
        wmenu.addAction(wa)
        status.addWidget(self.btn_window)
        lay.addLayout(status)
        self._update_undo_buttons()
        # The heart rate has its own card now: the strip is there on demand.
        window.figures.ui.rrhr_strip_visible = False
        on = False
        self.btn_strip.setChecked(on)
        self.plot.set_strip_visible(on)
        self.btn_strip.toggled.connect(self._strip_toggled)
        from ecg.qt import theme as _theme
        self._paint_legend(_theme.current())
        # The beat quality is real per-beat work: recomputed 250 ms after the
        # last peak change (PlotController._ensure_beat_corr_cache).
        self._corr_timer = QtCore.QTimer(self, singleShot=True, interval=250)
        self._corr_timer.timeout.connect(self._update_beat_quality)

        self._thr_timer = QtCore.QTimer(self, singleShot=True, interval=THRESH_DEBOUNCE_MS)
        self._thr_timer.timeout.connect(self.apply_threshold)
        self.sl_thr.valueChanged.connect(lambda v: self._set_threshold(v / 100.0, from_slider=True))
        self.sp_thr.valueChanged.connect(lambda v: self._set_threshold(v, from_slider=False))

    def resizeEvent(self, ev) -> None:
        """A narrow window: the bar drops its two quiet labels and shortens
        the slider rather than cutting its buttons."""
        super().resizeEvent(ev)
        narrow = ev.size().width() < NARROW_PX
        self.lbl_thr.setVisible(not narrow)
        self.lbl_method.setVisible(not narrow)
        self.sl_thr.setMinimumWidth(70 if narrow else 140)

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
        self.btn_review.setEnabled(False)
        self.btn_agreement.setEnabled(False)
        self.lbl_agreement.setText("")
        self.ed_win0.clear()
        self.ed_win1.clear()
        self._say_window("")
        self._update_undo_buttons()
        self.plot.clear()
        self.lbl_peaks.setText("Peaks: —")
        self.lbl_quality.setText("Quality —")
        self.btn_review_link.setEnabled(False)
        self.seg_trace.buttons[0].setChecked(True)
        self.seg_trace.buttons[1].setChecked(False)
        self.lbl_summary.setText("Detect the peaks: choose the method and threshold, then "
                                 "Detect peaks.")
        self.lbl_view.setText("")
        self._show_hr()
        self.update_period_counts()

    def detect(self) -> None:
        path = self.win.recording.signal.filepath
        if not path or not self.confirm_redetect():
            return
        self.btn_detect.setEnabled(False)
        self.win.refresh_primary()

        def _end() -> None:
            self.btn_detect.setEnabled(True)
            self.win.refresh_primary()
        self.win.run_job("Detect Peaks", load_and_detect, path, self.params(),
                         on_done=self._detected, on_end=_end)

    def confirm_redetect(self) -> bool:
        """A new detection starts from the detector's peaks: the peaks
        excluded and added by hand, and the Undo history, are dropped. Asked
        first when there are any (True: go on)."""
        d = self.win.recording.detection
        n_excl, n_add = len(d.manual_excluded), len(d.manual_added)
        if not (n_excl or n_add):
            return True
        parts = [f"{n} {what}" for n, what in ((n_excl, "excluded"), (n_add, "added"))
                 if n]
        B = QtWidgets.QMessageBox.StandardButton
        box = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Icon.Question, "Detect the peaks again",
            f"Detecting again replaces the peaks edited by hand ({' and '.join(parts)}) "
            "with the detector's, and clears Undo.", B.Cancel, self)
        box.addButton("Detect again", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        box.setDefaultButton(B.Cancel)
        box.exec()
        clicked = box.clickedButton()
        return clicked is not None and box.buttonRole(clicked) == \
            QtWidgets.QMessageBox.ButtonRole.DestructiveRole

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
        self._clear_agreement()
        rec.analysis.results = None
        self.reset_analysis_window(quiet=True)
        self.btn_edit.setEnabled(True)
        self.btn_agreement.setEnabled(True)
        self.btn_mark_noise.setEnabled(True)
        self.btn_review.setEnabled(True)
        self._update_undo_buttons()
        self.plot.set_signal(s.filtered, fs, s.time)
        self.plot.set_noise_regions(rec.analysis.noise_regions)
        self.plot.set_annotations(rec.analysis.annotations)
        self._show_peaks()
        self.win.peaks_changed()
        self.win.update_file_label()
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
        if on and self.btn_mark_noise.isChecked():
            self.btn_mark_noise.setChecked(False)
        self._sync_tools()

    def _move_mode(self) -> None:
        """Move: neither peak editing nor noise marking (a drag pans)."""
        self.btn_mark_noise.setChecked(False)
        self.btn_edit.setChecked(False)
        self._sync_tools()

    def _sync_tools(self) -> None:
        """One tool at a time (Move, Edit peaks, Mark noise), its hint, and
        Free placement while editing peaks."""
        noise, edit = self.btn_mark_noise.isChecked(), self.btn_edit.isChecked()
        self.btn_move.setChecked(not (noise or edit))
        self.chk_free.setVisible(edit)
        hint = ("Drag over a noisy stretch to mark it · right click a noisy region to remove it"
                if noise else
                "Left click a peak: exclude / include · right click: add / remove · "
                "Ctrl+Z / Ctrl+Y" if edit else "")
        self.lbl_edit.setText(hint)
        self.lbl_edit.setVisible(bool(hint))

    def refresh_to_check(self) -> None:
        """How many beats the tachogram flagged (after Analyze)."""
        spikes = self.win.figures.ui.rr_spike_times
        k = (0 if spikes is None or self.win.recording.analysis.results is None
             else len(spikes))
        self.lbl_to_check.setText(f"Beats to check: {k}" if k else "Beats to check")

    def _show_trace(self) -> None:
        """Filtered or raw on the trace (display only: the peaks stay those
        found on the filtered signal), the view kept."""
        s = self.win.recording.signal
        raw = self.seg_trace.buttons[1].isChecked()
        self.seg_trace.buttons[0].setChecked(not raw)
        sig = s.raw_norm if raw and s.raw_norm is not None else s.filtered
        if sig is None or s.fs is None:
            return
        t0, t1 = self.plot.window()
        self.plot.set_signal(sig, s.fs, s.time)
        self.plot.set_noise_regions(self.win.recording.analysis.noise_regions)
        self.plot.set_annotations(self.win.recording.analysis.annotations)
        self._show_peaks()
        self.plot.set_window(t0, t1)

    def _paint_legend(self, p) -> None:
        """The trace's legend: the marks as drawn."""
        from PySide6 import QtGui
        for kind, label in self._legend_icons:
            pix = QtGui.QPixmap(28, 28)
            pix.setDevicePixelRatio(2)
            pix.fill(QtCore.Qt.GlobalColor.transparent)
            painter = QtGui.QPainter(pix)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            if kind == "peak":
                painter.setBrush(QtGui.QColor(p.plot_rpeak))
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawEllipse(QtCore.QRectF(3, 3, 8, 8))
            elif kind == "excluded":
                painter.setPen(QtGui.QPen(QtGui.QColor(p.TEXT), 1.8))
                painter.drawLine(QtCore.QPointF(3, 3), QtCore.QPointF(11, 11))
                painter.drawLine(QtCore.QPointF(11, 3), QtCore.QPointF(3, 11))
            elif kind == "check":
                pen = QtGui.QPen(QtGui.QColor("#FF9800"), 1.6)
                painter.setPen(pen)
                painter.drawEllipse(QtCore.QRectF(2, 2, 10, 10))
            else:
                c = QtGui.QColor(p.ORANGE)
                c.setAlpha(80)
                painter.setBrush(c)
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawRoundedRect(QtCore.QRectF(0, 3, 14, 8), 2, 2)
            painter.end()
            label.setPixmap(pix)

    def set_view_width(self, width: float) -> None:
        """Show *width* seconds from the start of the current view."""
        if self.win.recording.signal.time is None:
            return
        t0, _t1 = self.plot.window()
        start = max(0.0, min(self._end() - width, t0))
        self.plot.set_window(start, start + width)

    def _view_moved(self, t0: float, t1: float) -> None:
        """The view's time span, its length among the presets, and the heart
        rate of the beats in view."""
        self.lbl_view.setText(f"{t0:.2f} – {t1:.2f} s")
        width = t1 - t0
        for b, w in zip(self.seg_window.buttons, WINDOW_PRESETS_S):
            b.setChecked(abs(width - w) < 0.01 * w)
        self._show_hr()

    def _show_hr(self) -> None:
        stats = self.plot.hr_stats()
        if stats is None:
            self.lbl_hr.setText("—")
            self.lbl_hr_range.setText("in view")
        else:
            med, lo, hi = stats
            self.lbl_hr.setText(f"{med:.0f}")
            self.lbl_hr_range.setText(f"in view · {lo:.0f} – {hi:.0f} bpm")
        hr = (self.win.recording.analysis.results or {}).get("hr") or {}
        mean = hr.get("mean")
        if mean is not None and np.isfinite(mean):
            self.lbl_hr_whole.setText(f"Recording: {mean:.0f} bpm")
            self.lbl_hr_whole.setToolTip("Mean heart rate of the analysis window (Analyze)")
        else:
            m = self.plot.mean_hr()
            self.lbl_hr_whole.setText(f"Recording: ≈ {m:.0f} bpm" if m else "")
            self.lbl_hr_whole.setToolTip("Mean of the beat-to-beat heart rate; Analyze gives "
                                         "the value on the valid intervals")

    def _sync_more_menu(self) -> None:
        """The More menu's entries follow their buttons (enabled or not)."""
        for key, btn in (("clear", self.btn_clear_edits), ("review", self.btn_review),
                         ("annotations", self.btn_annotations), ("noise", self.btn_noise)):
            act = self._more_actions[key]
            act.setText(btn.text())
            act.setEnabled(btn.isEnabled())

    # ── Mark Noise (DetectionController.toggle_noise_mode / _noise_press /
    #    on_detail_release) ──────────────────────────────────
    NOISE_MIN_S = 0.05          # a shorter drag is a click, not a region

    def _mark_noise_toggled(self, on: bool) -> None:
        self.plot.span_mode = on
        if on:
            if self.btn_edit.isChecked():
                self.btn_edit.setChecked(False)
            self.win.set_status("Mark noise: drag over a noisy stretch to leave its RR "
                                "intervals out of HR / HRV; right click a noisy region to "
                                "remove it.")
        else:
            self.win.set_status("Mark noise off.")
        self._sync_tools()

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
        self.mark_peaks_edited()
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

    def mark_peaks_edited(self) -> None:
        """DetectionController.mark_peaks_edited: the time of a hand change."""
        import datetime as dt
        self.win.figures.ui.peaks_edited_at = dt.datetime.now().isoformat(timespec="seconds")

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
        self.mark_peaks_edited()
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

    # ── analysis window ─────────────────────────────────────
    def _say_window(self, text: str, color: "str | None" = None) -> None:
        self.lbl_window.setText(text or "whole signal")
        self.lbl_window.setStyleSheet(f"color: {color};" if color else "")

    def show_window_in_entries(self) -> None:
        """The entries say the applied window (empty = signal start / end)."""
        a = self.win.recording.analysis
        for ed, v in ((self.ed_win0, a.t_start), (self.ed_win1, a.t_end)):
            ed.setText(f"{v:g}" if v and v > 0 else "")

    def apply_analysis_window(self) -> None:
        """SignalController.apply_analysis_window: stored for the next
        Analyze, with the peaks inside it counted (nothing recomputed)."""
        from ecg.qt import theme
        p = theme.current()
        rec = self.win.recording
        s, d, a = rec.signal, rec.detection, rec.analysis
        try:
            t0_raw, t1_raw = self.ed_win0.text().strip(), self.ed_win1.text().strip()
            t0 = float(t0_raw) if t0_raw else 0.0
            t1 = float(t1_raw) if t1_raw else 0.0
        except ValueError:
            self.win.set_status("Invalid window — enter numeric values.", warn=True)
            return
        if t1 > 0 and t0 >= t1:
            self.win.set_status("Start bound must be less than the end bound.", warn=True)
            return
        if s.time is not None and t1 > float(s.time[-1]) + 0.1:
            self.win.set_status(f"End bound exceeds the signal duration ({s.time[-1]:.1f} s).",
                                warn=True)
        a.t_start, a.t_end = t0, t1
        if d.rpeaks_ok is not None and s.fs is not None:
            fs = s.fs
            t_end_eff = float(s.time[-1]) if (s.time is not None and t1 == 0) else t1
            mask = d.rpeaks_ok / fs >= t0
            if t1 > 0:
                mask &= d.rpeaks_ok / fs <= t1
            n = int(mask.sum())
            dur = (t_end_eff - t0) if t1 > 0 else (float(s.time[-1]) - t0
                                                   if s.time is not None else 0)
            text = (f"✓  {n} peaks  ·  {t0:.1f} s → {t_end_eff:.1f} s  ({dur:.1f} s)"
                    if t0 > 0 or t1 > 0 else f"✓  {n} peaks  ·  full signal")
            self._say_window(text, p.GREEN if n >= 5 else p.ORANGE)
        else:
            self._say_window("✓  Window applied — run analysis")
        self.win.session_dirty = True
        self.win.set_status("Analysis window updated — click Analyze to refresh.")
        self.win.figures.analysis_ctrl.refresh_autonomic_status()

    def reset_analysis_window(self, quiet: bool = False) -> None:
        """SignalController.reset_analysis_window: the full signal."""
        rec = self.win.recording
        a, d, s = rec.analysis, rec.detection, rec.signal
        a.t_start = a.t_end = 0.0
        self.show_window_in_entries()
        if d.rpeaks_ok is not None:
            dur = float(s.time[-1]) if s.time is not None else 0.0
            self._say_window(f"Full signal  ·  {len(d.rpeaks_ok)} peaks  ·  {dur:.1f} s")
        else:
            self._say_window("")
        if not quiet:
            self.win.set_status("Analysis window reset — full signal.")
            self.win.figures.analysis_ctrl.refresh_autonomic_status()

    # ── navigation ──────────────────────────────────────────
    def _view_width(self) -> float:
        t0, t1 = self.plot.window()
        return max(t1 - t0, 0.1)

    def _end(self) -> float:
        time = self.win.recording.signal.time
        return float(time[-1]) if time is not None and len(time) else 0.0

    def go_to(self, t: float) -> None:
        """NavigationController.nav_goto: the view starts at *t* (kept inside
        the recording), its width kept."""
        if self.win.recording.signal.time is None:
            return
        width = self._view_width()
        start = max(0.0, min(self._end() - width, float(t)))
        self.plot.set_window(start, start + width)

    def center_on(self, t: float) -> None:
        """The view centred on *t*, its width kept, on this tab."""
        width = self._view_width()
        start = max(0.0, min(float(t) - width / 2, max(0.0, self._end() - width)))
        self.win.tabs.setCurrentWidget(self)
        self.plot.set_window(start, start + width)

    def step_spike(self, direction: int) -> None:
        """NavigationController.step_spike: the next / previous beat flagged
        on the tachogram (ui.rr_spike_times), from the centre of the view."""
        import numpy as np
        spikes = self.win.figures.ui.rr_spike_times
        if spikes is None or len(spikes) == 0 or self.win.recording.signal.time is None:
            self.win.set_status("No suspicious beats to step through — run Analyze first")
            return
        spikes = np.asarray(spikes, dtype=float)
        t0, t1 = self.plot.window()
        cur = (t0 + t1) / 2
        later, earlier = spikes[spikes > cur + 1e-6], spikes[spikes < cur - 1e-6]
        if direction > 0:
            t_nav = float(later.min()) if len(later) else float(spikes.min())
        else:
            t_nav = float(earlier.max()) if len(earlier) else float(spikes.max())
        self.center_on(t_nav)
        self.win.set_status(f"Navigation → spike at {t_nav:.3f} s")

    def clear_edits(self) -> None:
        """DetectionController.clear_manual_exclusions (undoable)."""
        import numpy as np
        d = self.win.recording.detection
        if not d.manual_excluded and not d.manual_added:
            return
        push_undo(d.edit_undo, d.edit_redo, d.manual_excluded, d.manual_added)
        n_excl, n_added = len(d.manual_excluded), len(d.manual_added)
        d.manual_excluded.clear()
        d.manual_added.clear()
        self._drop_results()
        self.mark_peaks_edited()
        d.rpeaks_manual_excl = np.array([], dtype=int)
        d.rpeaks_manual_added = np.array([], dtype=int)
        self.apply_threshold()
        self._update_undo_buttons()
        self.win.set_status(f"Manual edits cleared — {n_excl} exclusion(s), {n_added} "
                            "addition(s) removed.")

    # ── Check agreement ─────────────────────────────────────
    def _clear_agreement(self) -> None:
        """A new detection drops the last check (SignalController's reset)."""
        d = self.win.recording.detection
        d.agreement_disagree_samples = None
        d.agreement_summary = ""
        d.agreement_pct = None
        d.agreement_peaks = None
        self.lbl_agreement.setText("")
        self.plot.set_unconfirmed(None)

    def check_agreement(self) -> None:
        """DetectionController.check_method_agreement: the other detectors
        in a background job (ecg.services.agreement_service), then the
        report and the unconfirmed beats circled on the trace."""
        from ecg.services.agreement_service import method_agreement
        rec = self.win.recording
        s, d = rec.signal, rec.detection
        if s.filtered is None or d.rpeaks_ok is None or len(d.rpeaks_ok) == 0:
            QtWidgets.QMessageBox.warning(self, "Not ready", "Click Detect Peaks first.")
            return
        sig, fs, primary = s.filtered, s.fs, d.rpeaks_ok.copy()
        host = self.win.figures
        min_rr_ms = host._safe_float(host.ent_minrr, MouseECG.MIN_RR_MS)
        active = self.cb_method.currentText().lower()
        self.btn_agreement.setEnabled(False)
        self.lbl_agreement.setProperty("warn", True)
        self.lbl_agreement.setText("Running other detectors…")
        self.win.run_job("Check agreement",
                         lambda progress_cb=None: method_agreement(sig, fs, primary,
                                                                   min_rr_ms, active),
                         on_done=lambda r: self._agreement_done(r, primary, fs),
                         on_end=lambda: self.btn_agreement.setEnabled(True))

    def _agreement_done(self, result: dict, primary, fs: float) -> None:
        from ecg.qt.dialogs.agreement_dialog import AgreementDialog, agreement_color
        from ecg.services.agreement_service import agreement_summary
        d = self.win.recording.detection
        if d.rpeaks_ok is None:
            return
        lbl = self.lbl_agreement
        if not result.get("ok"):
            reason = "; ".join(result.get("methods_skipped", [])) or "no other methods available"
            lbl.setText(f"Agreement check: {reason}")
            self.win.set_status(f"Cross-method agreement: {reason}", warn=True)
            return
        summary, pct = agreement_summary(result)
        d.agreement_disagree_samples = result["disagree_samples"]
        d.agreement_summary = summary
        d.agreement_pct = pct
        d.agreement_peaks = primary
        lbl.setProperty("warn", False)
        lbl.setStyleSheet(f"color: {agreement_color(result['n_disagree'], pct)};")
        lbl.setText(summary)
        skipped = result.get("methods_skipped", [])
        note = ("  ·  skipped: " + ", ".join(skipped)) if skipped else ""
        self.win.set_status(f"Cross-method agreement done — {summary}{note}")
        self.plot.set_unconfirmed(d.agreement_disagree_samples)
        self.agreement_dialog = AgreementDialog(self.win, result, fs)
        self.agreement_dialog.show()

    def _review(self) -> None:
        from ecg.qt.dialogs.artifact_review import review_artifacts
        review_artifacts(self.win)

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

    def _strip_toggled(self, on: bool) -> None:
        self.win.figures.ui.rrhr_strip_visible = bool(on)
        self.plot.set_strip_visible(on)

    def _update_beat_quality(self) -> None:
        """PlotController._flush_beat_corr_recompute: each beat's template
        correlation, cached on the detection state for the peaks it is of."""
        from ecg.core.analysis import compute_beat_correlation
        rec = self.win.recording
        s, d = rec.signal, rec.detection
        if d.rpeaks_ok is not d._beat_corr_for:
            if s.filtered is None or d.rpeaks_ok is None or len(d.rpeaks_ok) < 3:
                d.beat_corr = d.beat_corr_peaks = None
            else:
                lo, hi = self.win.figures.analysis_ctrl.hrv_rr_bounds(quiet=True)
                r = compute_beat_correlation(s.filtered, d.rpeaks_ok, s.fs, lo, hi)
                d.beat_corr, d.beat_corr_peaks = r["beat_corr"], r["valid_rp"]
            d._beat_corr_for = d.rpeaks_ok
        q_t = d.beat_corr_peaks / s.fs if d.beat_corr_peaks is not None else None
        self.plot.set_beats(d.rpeaks_ok, q_t, d.beat_corr)

    def _show_peaks(self) -> None:
        rec = self.win.recording
        s, d = rec.signal, rec.detection
        self.plot.set_peaks(d.rpeaks_ok, d.rpeaks_rej, d.rpeaks_manual_excl, d.rpeaks_manual_added)
        self.plot.set_beats(d.rpeaks_ok,
                            d.beat_corr_peaks / s.fs if (d.beat_corr_peaks is not None
                                                         and d._beat_corr_for is d.rpeaks_ok)
                            else None,
                            d.beat_corr if d._beat_corr_for is d.rpeaks_ok else None)
        self._corr_timer.start()
        n = 0 if d.rpeaks_ok is None else len(d.rpeaks_ok)
        self.lbl_peaks.setText(f"Peaks: {n:,}")
        beat_corr = (rec.analysis.results or {}).get("beat_corr")
        q = (signal_quality_score(d.rpeaks_ok, s.fs, float(s.time[-1]), beat_corr)
             if d.rpeaks_ok is not None and s.time is not None else None)
        if q is not None:
            d.sig_quality = q
        word = ("" if q is None else " · good" if q >= 85 else " · fair" if q >= 60
                else " · poor")
        self.lbl_quality.setText(f"Quality {q}/100{word}" if q is not None else "Quality —")
        self.btn_review_link.setEnabled(self.btn_review.isEnabled())
        self.refresh_to_check()
        self._show_summary(n, q)
        self._show_hr()
        # The amplitude itself is a detail: in the tooltip, not on the bar.
        self.lbl_thr.setText("× median R")
        tip = (f"{self.threshold():.2f} × the median R-peak height = {d.thresh_amp:.2f} in "
               "normalised signal units")
        for w in (self.lbl_thr, self.sp_thr, self.sl_thr):
            w.setToolTip(tip)

    def _show_summary(self, n: int, quality) -> None:
        """Under the title: the peaks, how they were found, the hand edits."""
        d = self.win.recording.detection
        if d.rpeaks_ok is None:
            self.lbl_summary.setText("Detect the peaks: choose the method and threshold, then "
                                     "Detect peaks.")
            return
        parts = [f"{n:,} peaks", self.cb_method.currentText().split(" (")[0],
                 f"threshold {self.threshold():.2f}"]
        if quality is not None:
            parts.append(f"quality {quality}/100")
        edits = [f"{k} {what}" for k, what in ((len(d.manual_excluded), "excluded"),
                                                (len(d.manual_added), "added")) if k]
        if edits:
            parts.append(", ".join(edits) + " by hand")
        self.lbl_summary.setText(" · ".join(parts))

    def apply_palette(self, p) -> None:
        self._paint_legend(p)
        self.plot.apply_palette(p)
        self._show_peaks() if self.win.recording.detection.rpeaks_ok is not None else None


__all__ = ["DetectionTab", "METHODS", "DEFAULT_METHOD", "MouseECG"]
