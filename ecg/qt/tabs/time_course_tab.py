# -*- coding: utf-8 -*-
"""
ecg.qt.tabs.time_course_tab
---------------------------
Time course: HRV per epoch, in a sliding window, and the time-resolved
autonomic indices (CSI / CPI, optional SPWVD LF / HF).

The three computations are the Tk interface's own AnalysisController
methods (compute_epochs, compute_rolling_hrv, compute_autonomic), run
through the window's FigureHost: they read this tab's entries through the
same names (ent_epoch, _roll_metrics, ent_auto_win...), compute in a
background job (ecg.core.windowed_hrv, ecg.services.autonomic_service) and
draw with the shared figure code. Defaults are those of the Tk tab.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtWidgets

from ecg.core.autonomic import COV_METHODS
from ecg.core.models import MouseECG
from ecg.qt.figure_host import StatusLabel, TextBox, Value
from ecg.ui.autonomic_view import normalise_autonomic_params

if TYPE_CHECKING:
    from ecg.qt.main_window import MainWindow

# ecg/ui/tab_time_course.py: the metrics offered and those ticked at start-up.
WINDOW_METRICS = (("HR", "HR"), ("SDNN", "SDNN"), ("RMSSD", "RMSSD"), ("pNN6", "pNN6"),
                  ("SD1", "SD1"), ("SD2", "SD2"), ("LF_nu", "LF (n.u.)"),
                  ("HF_nu", "HF (n.u.)"), ("LF_HF", "LF/HF"))
EPOCH_DEFAULTS = ("HR", "SDNN", "RMSSD", "pNN6")
SLIDING_DEFAULTS = ("HR", "SDNN", "RMSSD")


def _spin(value: float, lo: float, hi: float, step: float = 1.0,
          decimals: int = 1) -> QtWidgets.QDoubleSpinBox:
    sp = QtWidgets.QDoubleSpinBox()
    sp.setRange(lo, hi)
    sp.setDecimals(decimals)
    sp.setSingleStep(step)
    sp.setValue(value)
    return sp


class TimeCourseTab(QtWidgets.QTabWidget):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.win = window
        host = window.figures
        ctl = host.analysis_ctrl
        self.addTab(self._epochs_page(host), "Epochs")
        self.addTab(self._sliding_page(host), "Sliding")
        self.addTab(self._autonomic_page(host), "Autonomic")
        self._buttons = (self.btn_epochs, self.btn_sliding, self.btn_auto)
        self.btn_epochs.clicked.connect(ctl.compute_epochs)
        self.btn_sliding.clicked.connect(ctl.compute_rolling_hrv)
        self.btn_auto.clicked.connect(lambda: ctl.compute_autonomic())
        self.results_changed()

    # ── pages ───────────────────────────────────────────────
    def _metrics_row(self, row: QtWidgets.QHBoxLayout, defaults) -> dict:
        row.addWidget(QtWidgets.QLabel("Metrics:"))
        out = {}
        for metric, label in WINDOW_METRICS:
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(metric in defaults)
            row.addWidget(cb)
            out[metric] = Value(cb.isChecked)
        return out

    def _results(self, lay: QtWidgets.QVBoxLayout, host, slot: str) -> TextBox:
        canvas = host.slot(slot, self, 12.0, 4.6)
        canvas.setMinimumHeight(420)
        lay.addWidget(canvas, 3)
        legend = QtWidgets.QLabel(host.analysis_ctrl.WINDOWED_LEGEND)
        legend.setProperty("muted", True)
        legend.setWordWrap(True)
        lay.addWidget(legend)
        box = TextBox()
        lay.addWidget(box, 2)
        return box

    def _epochs_page(self, host) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Window (s):"))
        self.sp_epoch = _spin(MouseECG.EPOCH_DEFAULT_S, MouseECG.EPOCH_MIN_S, 36000.0, 5.0, 0)
        row.addWidget(self.sp_epoch)
        row.addWidget(QtWidgets.QLabel("Overlap (s):"))
        self.sp_overlap = _spin(0.0, 0.0, 36000.0, 5.0, 0)
        row.addWidget(self.sp_overlap)
        self.btn_epochs = QtWidgets.QPushButton("Compute epochs")
        self.btn_epochs.setProperty("primary", True)
        row.addWidget(self.btn_epochs)
        host.lbl_epoch_info = StatusLabel("")
        host.lbl_epoch_info.setProperty("muted", True)
        row.addWidget(host.lbl_epoch_info, 1)
        host.lbl_epoch_count = StatusLabel("")
        row.addWidget(host.lbl_epoch_count)
        lay.addLayout(row)
        mrow = QtWidgets.QHBoxLayout()
        host._epoch_metrics = self._metrics_row(mrow, EPOCH_DEFAULTS)
        mrow.addStretch(1)
        lay.addLayout(mrow)
        host.ent_epoch = Value(self.sp_epoch.value)
        host.ent_overlap = Value(self.sp_overlap.value)
        host.txt_epochs = self._results(lay, host, "epochs")
        return page

    def _sliding_page(self, host) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Window (s):"))
        self.sp_roll_win = _spin(30.0, 3.0, 36000.0, 5.0, 0)
        row.addWidget(self.sp_roll_win)
        row.addWidget(QtWidgets.QLabel("Step (s):"))
        self.sp_roll_step = _spin(5.0, 1.0, 36000.0, 1.0, 0)
        row.addWidget(self.sp_roll_step)
        self.btn_sliding = QtWidgets.QPushButton("Compute sliding window")
        self.btn_sliding.setProperty("primary", True)
        row.addWidget(self.btn_sliding)
        host.lbl_roll_status = StatusLabel("")
        host.lbl_roll_status.setProperty("muted", True)
        row.addWidget(host.lbl_roll_status, 1)
        lay.addLayout(row)
        mrow = QtWidgets.QHBoxLayout()
        host._roll_metrics = self._metrics_row(mrow, SLIDING_DEFAULTS)
        mrow.addStretch(1)
        lay.addLayout(mrow)
        host.ent_roll_win = Value(self.sp_roll_win.value)
        host.ent_roll_step = Value(self.sp_roll_step.value)
        host.txt_rolling = self._results(lay, host, "rolling_hrv")
        return page

    def _autonomic_page(self, host) -> QtWidgets.QWidget:
        p = normalise_autonomic_params(host.analysis.autonomic_params or None)
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Window (s):"))
        self.sp_auto_win = _spin(float(p["window_s"]), 0.5, 600.0, 0.5, 1)
        row.addWidget(self.sp_auto_win)
        row.addWidget(QtWidgets.QLabel("Min pairs:"))
        self.sp_auto_pairs = QtWidgets.QSpinBox()
        self.sp_auto_pairs.setRange(1, 10000)
        self.sp_auto_pairs.setValue(int(p["min_pairs"]))
        row.addWidget(self.sp_auto_pairs)
        row.addWidget(QtWidgets.QLabel("Method:"))
        self.cb_auto_method = QtWidgets.QComboBox()
        self.cb_auto_method.addItems(list(COV_METHODS.values()))
        self.cb_auto_method.setCurrentText(COV_METHODS[p["method"]])
        row.addWidget(self.cb_auto_method)
        row.addWidget(QtWidgets.QLabel("ks:"))
        self.sp_ks = _spin(float(p["ks"]), 0.0, 1000.0, 0.5, 2)
        row.addWidget(self.sp_ks)
        row.addWidget(QtWidgets.QLabel("kp:"))
        self.sp_kp = _spin(float(p["kp"]), 0.0, 1000.0, 0.5, 2)
        row.addWidget(self.sp_kp)
        self.chk_lfhf = QtWidgets.QCheckBox("LF/HF (SPWVD)")
        self.chk_lfhf.setChecked(bool(p["lf_hf"]))
        row.addWidget(self.chk_lfhf)
        self.btn_auto = QtWidgets.QPushButton("Compute")
        self.btn_auto.setProperty("primary", True)
        row.addWidget(self.btn_auto)
        row.addStretch(1)
        lay.addLayout(row)
        host.lbl_auto_status = StatusLabel("")
        host.lbl_auto_status.setProperty("muted", True)
        host.lbl_auto_status.setWordWrap(True)
        lay.addWidget(host.lbl_auto_status)
        # AnalysisController reads them as entries / a combobox / a check box.
        host.ent_auto_win = Value(self.sp_auto_win.value)
        host.ent_auto_minpairs = Value(self.sp_auto_pairs.value)
        host.cb_auto_method = Value(self.cb_auto_method.currentText)
        host.ent_auto_ks = Value(self.sp_ks.value)
        host.ent_auto_kp = Value(self.sp_kp.value)
        host.chk_auto_lfhf = Value(lambda: 1 if self.chk_lfhf.isChecked() else 0)
        canvas = host.slot("autonomic", self, 12.0, 5.0)
        canvas.setMinimumHeight(460)
        lay.addWidget(canvas, 3)
        host.txt_autonomic = TextBox()
        lay.addWidget(host.txt_autonomic, 2)
        return page

    # ── state ───────────────────────────────────────────────
    def results_changed(self) -> None:
        """Epochs / Sliding need Analyze's results, Autonomic only peaks."""
        rec = self.win.recording
        has_results = rec.analysis.results is not None
        self.btn_epochs.setEnabled(has_results)
        self.btn_sliding.setEnabled(has_results)
        self.btn_auto.setEnabled(rec.detection.rpeaks_ok is not None)
        if not has_results:
            host = self.win.figures
            for name in ("epochs", "rolling_hrv"):
                host.slot(name).clear()
            host.txt_epochs.setPlainText("")
            host.txt_rolling.setPlainText("")
            rec.analysis.epoch_df = rec.analysis.rolling_hrv_df = None
