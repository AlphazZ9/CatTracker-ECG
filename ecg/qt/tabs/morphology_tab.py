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

from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtWidgets

from ecg.core.analysis import (ENSEMBLE_BEATS_DEFAULT, analyse_intervals,
                               normalize_ensemble_beats, recompute_qtc_column)
from ecg.core.models import QT_REPORTED, QTC_FORMULA_LABELS, normalize_qtc_formula
from ecg.core.wave_template import WaveTemplate
from ecg.qt.figure_host import StatusLabel

if TYPE_CHECKING:
    from ecg.qt.main_window import MainWindow


class MorphologyTab(QtWidgets.QScrollArea):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.win = window
        host = window.figures
        self.setWidgetResizable(True)
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        self.setWidget(page)

        bar = QtWidgets.QFrame(objectName="Card")
        row = QtWidgets.QHBoxLayout(bar)
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
        self.chk_permissive = QtWidgets.QCheckBox("Permissive bounds")
        self.chk_permissive.setToolTip("Accept intervals up to twice the strict physiological "
                                       "bounds")
        row.addWidget(self.chk_permissive)
        self.btn_template = QtWidgets.QPushButton("Wave template…")
        self.btn_template.setToolTip("The P / Q / S / T search windows of this recording, on "
                                     "its mean beat")
        self.btn_template.clicked.connect(self._edit_template)
        row.addWidget(self.btn_template)
        self.btn_measure = QtWidgets.QPushButton("Measure intervals")
        self.btn_measure.setProperty("primary", True)
        self.btn_measure.setEnabled(False)
        self.btn_measure.clicked.connect(self.measure)
        row.addWidget(self.btn_measure)
        row.addStretch(1)
        lay.addWidget(bar)
        from ecg.qt.widgets.stale_banner import StaleBanner
        self.banner = StaleBanner(self, on_recompute=lambda: self.measure())
        lay.addWidget(self.banner)
        host.lbl_ivl_status = StatusLabel("Click Analyze first")
        host.lbl_ivl_status.setProperty("muted", True)
        lay.addWidget(host.lbl_ivl_status)
        # The three steps (TabMorphologyMixin's chips): mean beat, check the
        # landmarks, measure every beat -- the current one highlighted.
        steps = QtWidgets.QHBoxLayout()
        self.step_labels = []
        for _i in range(3):
            lbl = QtWidgets.QLabel("")
            steps.addWidget(lbl)
            self.step_labels.append(lbl)
        steps.addStretch(1)
        lay.addLayout(steps)
        # What the medians rest on, then one line per interval (median, IQR,
        # beats measured), as the Tk interval table.
        self.lbl_measured = QtWidgets.QLabel("")
        lay.addWidget(self.lbl_measured)
        self.tbl_ivl = QtWidgets.QTableWidget(0, 5)
        self.tbl_ivl.setHorizontalHeaderLabels(["Interval", "Definition", "Median", "IQR",
                                                "Measured"])
        self.tbl_ivl.verticalHeader().setVisible(False)
        self.tbl_ivl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_ivl.horizontalHeader().setStretchLastSection(True)
        self.tbl_ivl.setMaximumHeight(170)
        lay.addWidget(self.tbl_ivl)
        self._ivl_shown: object = False

        def card(title: str, widget: QtWidgets.QWidget, min_h: int) -> QtWidgets.QFrame:
            f = QtWidgets.QFrame(objectName="Card")
            fl = QtWidgets.QVBoxLayout(f)
            fl.addWidget(QtWidgets.QLabel(f"<b>{title}</b>"))
            widget.setMinimumHeight(min_h)
            fl.addWidget(widget, 1)
            return f

        lay.addWidget(card("Mean beat and landmarks", host.slot("beat", page, 12.0, 4.6), 420))
        lay.addWidget(card("Intervals per beat", host.slot("intervals", page, 12.0, 5.4), 460))
        if QT_REPORTED:
            lay.addWidget(card("QTc vs RR", host.slot("qtc_rr", page, 6.0, 2.0), 220))
        lay.addWidget(card("Beat distributions", host.slot("beat_dist", page, 12.0, 3.8), 340))

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
    def results_changed(self) -> None:
        has = self.win.recording.analysis.results is not None
        self.btn_measure.setEnabled(has)
        if not has:
            self.win.figures.lbl_ivl_status.configure(text="Click Analyze first")
            for name in ("beat", "intervals", "beat_dist", "qtc_rr"):
                if name in self.win.figures._slots:
                    self.win.figures._slots[name].clear()

    def refresh_banner(self) -> None:
        """TabMorphologyMixin._refresh_morphology: the steps, the interval
        table (rebuilt only when the frame changes) and the stale line."""
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
