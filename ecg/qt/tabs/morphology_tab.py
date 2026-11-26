# -*- coding: utf-8 -*-
"""
ecg.qt.tabs.morphology_tab
--------------------------
Morphology & Intervals: the mean beat (after Analyze), Measure intervals
(PR / QRS / QT / QTc per beat, ecg.core.analysis.analyse_intervals in a
background job), the rolling-median window, permissive bounds, and the
interval and beat-quality distributions.

The figures come from the Tk interface's own figure code (FigureHost). The
parameters are those of AnalysisController.run_intervals: the recording's
wave template (the user's saved default the first time, as Tk), strict or
permissive bounds, the QTc formula, the RR bounds of the results and the
rolling-median N (ensemble_beats, default 32).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from PySide6 import QtCore, QtWidgets

from ecg.core.analysis import (ENSEMBLE_BEATS_DEFAULT, analyse_intervals,
                               normalize_ensemble_beats, recompute_qtc_column)
from ecg.core.models import QT_REPORTED, QTC_FORMULA_LABELS, normalize_qtc_formula
from ecg.core.wave_template import WaveTemplate
from ecg.qt.figure_host import StatusLabel

if TYPE_CHECKING:
    from ecg.qt.main_window import MainWindow

NEEDS_ANALYZE = "Click Analyze first"


class MorphologyTab(QtWidgets.QScrollArea):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.win = window
        host = window.figures
        self.setWidgetResizable(True)
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        self.setWidget(page)
        lay.setContentsMargins(24, 16, 24, 16)
        lay.setSpacing(12)
        from ecg.qt.widgets.shell import page_header
        lay.addLayout(page_header('Morphology & intervals'))

        # The Tk arrangement: the three steps on top; the mean beat (left)
        # beside what measures the intervals and their table (right); then
        # the intervals per beat and the beat distributions.
        steps = QtWidgets.QHBoxLayout()
        self.step_labels = []
        for _i in range(3):
            lbl = QtWidgets.QLabel("")
            steps.addWidget(lbl)
            self.step_labels.append(lbl)
        steps.addStretch(1)
        lay.addLayout(steps)
        from ecg.qt.widgets.stale_banner import StaleBanner
        self.banner = StaleBanner(self, on_recompute=lambda: self.measure())
        lay.addWidget(self.banner)

        def card(title: str, widget: "Optional[QtWidgets.QWidget]" = None, min_h: int = 0,
                 extra: "Optional[QtWidgets.QWidget]" = None) -> "tuple[QtWidgets.QFrame, QtWidgets.QVBoxLayout]":
            f = QtWidgets.QFrame(objectName="Card")
            fl = QtWidgets.QVBoxLayout(f)
            fl.setContentsMargins(16, 12, 16, 12)
            head = QtWidgets.QHBoxLayout()
            t = QtWidgets.QLabel(title)
            t.setProperty("role", "heading")
            head.addWidget(t)
            head.addStretch(1)
            if extra is not None:
                head.addWidget(extra)
            fl.addLayout(head)
            if widget is not None:
                widget.setMinimumHeight(min_h)
                fl.addWidget(widget, 1)
            return f, fl

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(14)
        lay.addLayout(body, 1)

        # ── left: the mean beat and its landmarks ──
        self.btn_template = QtWidgets.QPushButton("Landmarks…")
        self.btn_template.setToolTip("The P / Q / S / T search windows of this recording, on "
                                     "its mean beat")
        self.btn_template.clicked.connect(self._edit_template)
        # The mean beat on screen (PyQtGraph); the Matplotlib figure stays,
        # hidden, for the exports.
        from ecg.qt.widgets.beat_chart import BeatChart
        self.beat_chart = BeatChart()
        host.slot("beat", page, 12.0, 4.6).hide()
        left, ll = card("Mean beat ± SD · landmarks", None, 0, self.btn_template)
        hint = QtWidgets.QLabel("Landmarks…: move a search window on this mean beat")
        hint.setProperty("muted", True)
        ll.addWidget(hint)
        self.beat_chart.setMinimumHeight(380)
        ll.addWidget(self.beat_chart, 1)
        body.addWidget(left, 1)

        # ── right: measure, then the interval table ──
        right, rl = card("Intervals")
        row = QtWidgets.QHBoxLayout()
        self.cb_qtc = QtWidgets.QComboBox()
        for key, label in QTC_FORMULA_LABELS.items():
            self.cb_qtc.addItem(label, key)
        self.cb_qtc.currentIndexChanged.connect(self._qtc_changed)
        if QT_REPORTED:            # no QTc formula while QT is not reported (models.QT_REPORTED)
            row.addWidget(QtWidgets.QLabel("QTc formula"))
            row.addWidget(self.cb_qtc)
        else:
            self.cb_qtc.setVisible(False)
        row.addWidget(QtWidgets.QLabel("Rolling median (beats)"))
        self.sp_ensemble = QtWidgets.QSpinBox()
        self.sp_ensemble.setRange(1, 4096)
        self.sp_ensemble.setValue(ENSEMBLE_BEATS_DEFAULT)
        self.sp_ensemble.setToolTip("Each beat is measured on the median of N neighbouring beats; "
                                    "1 measures each beat on itself. Do not compare results "
                                    "obtained with different N.")
        self.sp_ensemble.valueChanged.connect(self._ensemble_changed)
        row.addWidget(self.sp_ensemble)
        row.addStretch(1)
        rl.addLayout(row)
        row2 = QtWidgets.QHBoxLayout()
        self.btn_measure = QtWidgets.QPushButton("Measure intervals")
        self.btn_measure.setProperty("action", True)
        self.btn_measure.setEnabled(False)
        self.btn_measure.clicked.connect(self.measure)
        row2.addWidget(self.btn_measure)
        self.chk_permissive = QtWidgets.QCheckBox("Permissive bounds")
        self.chk_permissive.setToolTip("Accept intervals up to twice the strict physiological "
                                       "bounds")
        row2.addWidget(self.chk_permissive)
        row2.addStretch(1)
        rl.addLayout(row2)
        host.lbl_ivl_status = StatusLabel(NEEDS_ANALYZE)
        host.lbl_ivl_status.setProperty("muted", True)
        host.lbl_ivl_status.setWordWrap(True)
        rl.addWidget(host.lbl_ivl_status)
        # What the medians rest on, then one line per interval (median, IQR,
        # beats measured), as the Tk interval table.
        self.lbl_measured = QtWidgets.QLabel("")
        self.lbl_measured.setWordWrap(True)
        rl.addWidget(self.lbl_measured)
        self.tbl_ivl = QtWidgets.QTableWidget(0, 5)
        self.tbl_ivl.setHorizontalHeaderLabels(["Interval", "Definition", "Median", "IQR",
                                                "Measured"])
        self.tbl_ivl.verticalHeader().setVisible(False)
        self.tbl_ivl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_ivl.horizontalHeader().setStretchLastSection(True)
        self.tbl_ivl.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.tbl_ivl.setShowGrid(False)
        rl.addWidget(self.tbl_ivl)
        self._ivl_shown: object = False
        # The QTc-RR figure and its numbers in words (PlotController._plot_qtc_rr),
        # only while QT is reported.
        host.lbl_qtc_rr = None
        if QT_REPORTED:
            qtc = host.slot("qtc_rr", page, 6.0, 2.0)
            qtc.setMinimumHeight(220)
            rl.addWidget(qtc, 1)
            host.lbl_qtc_rr = StatusLabel("")
            host.lbl_qtc_rr.setWordWrap(True)
            host.lbl_qtc_rr.setProperty("muted", True)
            rl.addWidget(host.lbl_qtc_rr)
        rl.addStretch(1)
        body.addWidget(right, 1)

        # ── below: every beat, and the distributions ──
        lay.addWidget(card("Intervals per beat", host.slot("intervals", page, 12.0, 5.4),
                           460)[0])
        from ecg.qt.widgets.dist_charts import BeatDistCharts
        self.dist_charts = BeatDistCharts()
        slot = host.slot("beat_dist", page, 12.0, 3.8)
        slot.hide()
        slot.on_clear = self.dist_charts.clear
        host._screen_charts["beat_dist"] = self.dist_charts.set_data
        lay.addWidget(card("Beat distributions", self.dist_charts, 260)[0])

    # ── parameters ──────────────────────────────────────────
    def _ensemble_changed(self, value: int) -> None:
        self.win.recording.analysis.ensemble_beats = normalize_ensemble_beats(value)

    def _qtc_changed(self) -> None:
        """A new QTc formula recomputes the column of measured intervals
        (AnalysisController.on_qtc_formula_change)."""
        a = self.win.recording.analysis
        a.qtc_formula = normalize_qtc_formula(self.cb_qtc.currentData())
        ivl = (a.results or {}).get("intervals")
        if ivl is not None and not ivl.empty:
            a.results["intervals"] = recompute_qtc_column(ivl, a.qtc_formula)
            self.win.figures.plot_ctrl.plot_intervals(a.results)

    def wave_template(self) -> WaveTemplate:
        """AnalysisController.recording_wave_template, with a Qt notice."""
        a = self.win.recording.analysis
        if a.wave_template is None:
            wt, notice = WaveTemplate.load_default_checked()
            a.wave_template = wt
            if notice:
                QtWidgets.QMessageBox.warning(self, "Saved default template not used", notice)
        return a.wave_template

    def _edit_template(self) -> None:
        from ecg.qt.dialogs.wave_editor import open_editor
        open_editor(self.win)

    # ── Measure intervals ───────────────────────────────────
    def refresh_beat(self) -> None:
        """The mean beat chart from the results and the template in use."""
        from ecg.app.plot_controller import beat_template_overlay
        from ecg.app.wave_landmarks import WaveLandmarks
        a = self.win.recording.analysis
        r = a.results
        marks: dict = {}
        if r is not None:
            wt, _note = beat_template_overlay(a.wave_template, r.get("intervals"))
            if wt is None:
                wt, _n = WaveTemplate.load_default_checked(quiet=True)
            marks = {k: (float(v[0]), float(v[1])) for k, v in wt.landmarks.items()}
        key = (id(r), tuple(sorted(marks.items())))
        if key == getattr(self, "_beat_key", None):
            return
        self._beat_key = key
        self.beat_chart.set_beat(r, marks, WaveLandmarks.WAVE_SHORT_LABELS,
                                 WaveLandmarks.WAVE_COLORS)

    def results_changed(self) -> None:
        QtCore.QTimer.singleShot(0, self.refresh_beat)
        has = self.win.recording.analysis.results is not None
        self.btn_measure.setEnabled(has)
        lbl = self.win.figures.lbl_ivl_status
        if has and lbl.text() == NEEDS_ANALYZE:
            lbl.configure(text="")          # the steps below say what comes next
        if not has:
            lbl.configure(text=NEEDS_ANALYZE)
            for name in ("beat", "intervals", "beat_dist", "qtc_rr"):
                if name in self.win.figures._slots:
                    self.win.figures._slots[name].clear()

    def refresh_banner(self) -> None:
        """TabMorphologyMixin._refresh_morphology: the steps, the interval
        table (rebuilt only when the frame changes) and the stale line."""
        self.refresh_beat()                     # a template moved in the editor
        from ecg.qt import theme
        from ecg.app.freshness import COMPUTING, OUT_OF_DATE, UP_TO_DATE, stale_banner_text
        from ecg.app.morphology_helpers import (interval_lines, measured_headline,
                                                morphology_steps, step_tip)
        host = self.win.figures
        p = theme.current()
        r = self.win.recording.analysis.results or {}
        ivl = r.get("intervals")
        beats = r.get("beat_corr")
        glyphs = {UP_TO_DATE: "●", OUT_OF_DATE: "◆", COMPUTING: "◐"}
        colors = {UP_TO_DATE: p.GREEN, OUT_OF_DATE: p.ORANGE, COMPUTING: p.BLUE}
        for lbl, step in zip(self.step_labels, morphology_steps(
                host.freshness.status("core"), host.freshness.status("intervals"),
                len(beats) if beats is not None else None)):
            glyph = "●" if step.current else glyphs.get(step.state, "○")
            color = p.BLUE if step.current else colors.get(step.state, p.MUTED)
            lbl.setText(f'<span style="color:{color}">{glyph}</span>  {step.number}   '
                        + (f"<b>{step.text}</b>" if step.current else step.text))
            lbl.setToolTip(step_tip(step))
        if ivl is not self._ivl_shown:
            self._ivl_shown = ivl
            text, tier = measured_headline(ivl)
            self.lbl_measured.setText(text)
            self.lbl_measured.setStyleSheet(
                f"color: {({'good': p.GREEN, 'partial': p.ORANGE, 'low': p.RED}).get(tier, p.MUTED)};")
            lines = interval_lines(ivl)
            self.tbl_ivl.setRowCount(len(lines))
            for i, line in enumerate(lines):
                for c, v in enumerate((line.name, line.definition, line.median, line.iqr,
                                       line.measured)):
                    self.tbl_ivl.setItem(i, c, QtWidgets.QTableWidgetItem(v))
            self.tbl_ivl.resizeColumnsToContents()
            h = self.tbl_ivl.horizontalHeader().height() + 4 + sum(       # every line shown
                self.tbl_ivl.rowHeight(r) for r in range(self.tbl_ivl.rowCount()))
            self.tbl_ivl.setFixedHeight(max(h, 60))
        self.banner.set_text(stale_banner_text(
            "Intervals", host.freshness.status("intervals"),
            host.ui.module_computed_at.get("intervals"), host.ui.peaks_edited_at,
            has_values=ivl is not None and "QT_ms" in ivl.columns))

    def measure(self) -> None:
        rec = self.win.recording
        host = self.win.figures
        ctl = host.analysis_ctrl
        rp = host._windowed_peaks()
        if rec.analysis.results is None or rp is None or len(rp) < 5:
            return
        fs, sig = rec.signal.fs, rec.signal.filtered
        rr = np.diff(rp).astype(float) / fs * 1000
        wt = self.wave_template().copy()
        qtc = ctl.qtc_formula()
        lo, hi = ctl.results_rr_bounds()
        basis = host.freshness.capture(rr_bounds=(lo, hi))
        host.lbl_ivl_status.configure(text="Delineating all beats…")
        self.btn_measure.setEnabled(False)
        self.win.run_job("Measure intervals", analyse_intervals, sig, rp, fs, rr,
                         wave_template=wt, permissive_bounds=self.chk_permissive.isChecked(),
                         qtc_formula=qtc, rr_min_ms=lo, rr_max_ms=hi,
                         ensemble_beats=ctl.ensemble_beats(), peaks_bound=True,
                         on_done=lambda df: self._measured(df, wt, qtc, basis),
                         on_end=lambda: self.btn_measure.setEnabled(
                             self.win.recording.analysis.results is not None))

    def _measured(self, df, wt: WaveTemplate, qtc: str, basis: "dict | None" = None) -> None:
        """AnalysisController.run_intervals's landing."""
        a = self.win.recording.analysis
        host = self.win.figures
        if a.results is None:
            return
        if host.analysis_ctrl.qtc_formula() != qtc:
            df = recompute_qtc_column(df, host.analysis_ctrl.qtc_formula())
        a.results["intervals"] = df
        if basis is not None:
            host.freshness.stamp("intervals", basis)
        host.analysis_ctrl.note_intervals_measured(df)
        note, color, n_ok, n_total = host.analysis_ctrl._measured_note(df, wt.provenance())
        host.lbl_ivl_status.configure(text=note, text_color=color)
        pc = host.plot_ctrl
        r = a.results
        pc.run_plot_chain([("ECG intervals", lambda: pc.plot_intervals(r)),
                           ("Beat template", lambda: pc.plot_beat_template(r)),
                           ("Metrics", lambda: pc.plot_table_views(r))],
                          on_complete=lambda: self.win.set_status(
                              f"Interval delineation done — {n_ok}/{n_total} beats"))
