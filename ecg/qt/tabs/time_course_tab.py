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
from ecg.qt.figure_host import Field, StatusLabel, TextBox
from ecg.qt.theme import current as theme_current
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
        self.addTab(self._event_locked_page(host), "Event-locked")
        self._buttons = (self.btn_epochs, self.btn_sliding, self.btn_auto)
        # The mode shown (ui.time_course_mode, saved with sessions); the
        # Event-locked page is part of Tk's Autonomic mode.
        self.currentChanged.connect(lambda i: setattr(
            host.ui, "time_course_mode", ("Epochs", "Sliding", "Autonomic", "Autonomic")[i]))
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
            out[metric] = Field(cb)
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
        host.ent_epoch = Field(self.sp_epoch)
        host.ent_overlap = Field(self.sp_overlap)
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
        # Article-style Poincaré (LW) per Sliding window, as the Tk Autonomic
        # mode's row: in _roll_metrics, on the Sliding window and step.
        lw = QtWidgets.QHBoxLayout()
        lw.addWidget(QtWidgets.QLabel("Article-style Poincaré (LW), per window:"))
        for metric, label in (("CCD", "CCD"), ("SD1_LW", "SD1 (LW)"), ("SD2_LW", "SD2 (LW)")):
            cb = QtWidgets.QCheckBox(label)
            cb.setToolTip("LW: Ledoit-Wolf shrinkage covariance (sklearn), not the article's "
                          "estimator")
            lw.addWidget(cb)
            host._roll_metrics[metric] = Field(cb)
        lw.addStretch(1)
        lay.addLayout(lw)
        host.ent_roll_win = Field(self.sp_roll_win)
        host.ent_roll_step = Field(self.sp_roll_step)
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
        self.btn_calibrate = QtWidgets.QPushButton("Calibrate ks/kp…")
        self.btn_calibrate.setToolTip("Choose ks / kp on the cohort registry's event-locked rows "
                                      "(the paper's values are rat values, not validated in mice)")
        self.btn_calibrate.clicked.connect(self.open_calibration)
        row.addWidget(self.btn_calibrate)
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
        host.ent_auto_win = Field(self.sp_auto_win)
        host.ent_auto_minpairs = Field(self.sp_auto_pairs)
        host.cb_auto_method = Field(self.cb_auto_method)
        host.ent_auto_ks = Field(self.sp_ks)
        host.ent_auto_kp = Field(self.sp_kp)
        host.chk_auto_lfhf = Field(self.chk_lfhf, as_int=True)
        canvas = host.slot("autonomic", self, 12.0, 5.0)
        canvas.setMinimumHeight(460)
        lay.addWidget(canvas, 3)
        host.txt_autonomic = TextBox()
        lay.addWidget(host.txt_autonomic, 2)
        return page

    def _event_locked_page(self, host) -> QtWidgets.QWidget:
        """Time course > Event-locked: EventLockedController (the Tk one) on
        the host, reading these inputs under the Tk names."""
        from ecg.ui.event_locked_controller import EventLockedController
        from ecg.ui.event_locked_view import EVENT_TABLE_COLUMNS, EVENT_TABLE_NOTE, \
            EVENT_TABLE_TITLE
        import ecg.ui.event_locked_controller as elc
        from ecg.qt.figure_host import QtFileDialog
        elc.filedialog = QtFileDialog                    # Export CSV… under Qt
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Events:"))
        self.cb_evt_label = QtWidgets.QComboBox()
        self.cb_evt_label.setMinimumWidth(160)
        row.addWidget(self.cb_evt_label)
        row.addWidget(QtWidgets.QLabel("Pre (s):"))
        self.sp_evt_pre = _spin(60.0, 0.5, 36000.0, 5.0, 1)
        row.addWidget(self.sp_evt_pre)
        row.addWidget(QtWidgets.QLabel("Post (s):"))
        self.sp_evt_post = _spin(60.0, 0.5, 36000.0, 5.0, 1)
        row.addWidget(self.sp_evt_post)
        self.chk_evt_z = QtWidgets.QCheckBox("z-score")
        self.chk_evt_z.setToolTip("Traces as z-scores of each event's pre window (display only)")
        row.addWidget(self.chk_evt_z)
        self.btn_evt = QtWidgets.QPushButton("Compute event-locked")
        self.btn_evt.setProperty("primary", True)
        row.addWidget(self.btn_evt)
        row.addStretch(1)
        lay.addLayout(row)
        host.lbl_evt_status = StatusLabel("")
        host.lbl_evt_status.setProperty("muted", True)
        host.lbl_evt_status.setWordWrap(True)
        lay.addWidget(host.lbl_evt_status)
        host.cb_evt_label = Field(self.cb_evt_label)
        host.ent_evt_pre = Field(self.sp_evt_pre)
        host.ent_evt_post = Field(self.sp_evt_post)
        host.chk_evt_z = Field(self.chk_evt_z, as_int=True)
        body = QtWidgets.QHBoxLayout()
        canvas = host.slot("event_locked", self, 8.0, 5.0)
        canvas.setMinimumHeight(460)
        body.addWidget(canvas, 3)
        right = QtWidgets.QVBoxLayout()
        host.lbl_evt_table_head = StatusLabel(EVENT_TABLE_TITLE)
        right.addWidget(host.lbl_evt_table_head)
        self.tbl_evt = QtWidgets.QTableWidget(0, len(EVENT_TABLE_COLUMNS))
        self.tbl_evt.setHorizontalHeaderLabels(list(EVENT_TABLE_COLUMNS))
        self.tbl_evt.verticalHeader().setVisible(False)
        self.tbl_evt.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        right.addWidget(self.tbl_evt, 1)
        note = QtWidgets.QLabel(EVENT_TABLE_NOTE)
        note.setProperty("muted", True)
        note.setWordWrap(True)
        right.addWidget(note)
        bar = QtWidgets.QHBoxLayout()
        self.btn_evt_save = QtWidgets.QPushButton("Save to cohort registry")
        self.btn_evt_save.setToolTip("Store this recording's per-event values in the local "
                                     "registry with the Subject ID and the method / window / "
                                     "ks / kp")
        self.btn_evt_csv = QtWidgets.QPushButton("Export CSV…")
        bar.addWidget(self.btn_evt_save)
        bar.addWidget(self.btn_evt_csv)
        bar.addStretch(1)
        self.btn_to_compare = QtWidgets.QPushButton("Compare segments →")
        self.btn_to_compare.setToolTip("Open the Compare segments tab: time windows chosen by "
                                       "hand, block statistics, tachograms and Poincaré, no "
                                       "annotation needed (Ctrl+M)")
        self.btn_to_compare.clicked.connect(
            lambda: self.win.tabs.setCurrentWidget(self.win.compare))
        bar.addWidget(self.btn_to_compare)
        right.addLayout(bar)
        body.addLayout(right, 2)
        lay.addLayout(body, 1)
        host.txt_event_locked = TextBox()
        lay.addWidget(host.txt_event_locked)
        host._fill_event_rows = self._fill_event_rows
        ctl = EventLockedController(host)
        host.event_ctrl = ctl
        self.btn_evt.clicked.connect(ctl.compute)
        self.chk_evt_z.toggled.connect(lambda *_a: ctl.on_zscore_toggle())
        self.btn_evt_save.clicked.connect(ctl.save_to_registry)
        self.btn_evt_csv.clicked.connect(ctl.export_csv)
        ctl.apply_params(host.analysis.event_locked_params)
        ctl.invalidate(None)
        return page

    def open_calibration(self) -> None:
        """EventLockedController.open_calibration, with the Qt dialog."""
        from ecg.qt.dialogs.calibration_dialog import CalibrationDialog
        ctl = self.win.figures.event_ctrl
        ks, kp = ctl.current_gains()
        _fp, _sub, project = ctl._identity()
        self.calibration = CalibrationDialog(self.win, ks=ks, kp=kp,
                                             config=ctl.current_config(), project=project,
                                             on_apply=ctl.apply_gains)
        self.calibration.show()

    def _fill_event_rows(self, rows) -> None:
        """The compact per-event table (TimeCourseMixin._fill_event_rows)."""
        from PySide6 import QtGui
        t = self.tbl_evt
        t.setRowCount(0)
        warn = QtGui.QColor(theme_current().ORANGE)
        for r, row in enumerate(rows):
            t.insertRow(r)
            for c, text in enumerate((row.onset, row.hr, row.csi, row.cpi, row.valid)):
                item = QtWidgets.QTableWidgetItem(text)
                if c >= 4 and not row.is_valid:
                    item.setForeground(warn)
                t.setItem(r, c, item)
        t.resizeColumnsToContents()

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
