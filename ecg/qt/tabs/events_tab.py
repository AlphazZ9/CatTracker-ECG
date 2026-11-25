# -*- coding: utf-8 -*-
"""
ecg.qt.tabs.events_tab
----------------------
Events: abnormal-rhythm classification (bradycardia / tachycardia relative
to the recording's own baseline, pauses, premature beats, irregular runs,
AV-conduction delay when intervals were measured), its overview ribbon,
the list of events and the ECG around the one selected.

Classification is the Tk interface's AnalysisController.
run_arrhythmia_analysis (ecg.services.events_service), run through the
FigureHost; the ribbon and the ECG view are its figure code. Settings and
defaults are those of the Tk tab (baseline 30 s, bradycardia 20 %, 10
beats).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

from ecg.core.models import event_kind_label
from ecg.qt.figure_host import StatusLabel, Value

if TYPE_CHECKING:
    from ecg.qt.main_window import MainWindow

COLUMNS = ("#", "Event", "Start (s)", "End (s)", "Beats", "HR (bpm)", "Δ baseline", "Severity")


class EventsTab(QtWidgets.QWidget):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.win = window
        host = window.figures
        lay = QtWidgets.QVBoxLayout(self)

        bar = QtWidgets.QFrame(objectName="Card")
        row = QtWidgets.QHBoxLayout(bar)

        def spin(label: str, value: float, lo: float, hi: float, tip: str) -> QtWidgets.QDoubleSpinBox:
            row.addWidget(QtWidgets.QLabel(label))
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(0)
            sp.setValue(value)
            sp.setToolTip(tip)
            row.addWidget(sp)
            return sp

        self.sp_baseline = spin("Baseline (s)", 30, 5, 36000,
                                "Window at the start of the analysed span that sets the "
                                "recording's own baseline heart rate")
        self.sp_brady = spin("Brady / tachy (%)", 20, 5, 60,
                             "Deviation from the baseline heart rate that counts")
        self.sp_min_beats = spin("Min beats", 10, 3, 10000,
                                 "Shortest bradycardia / tachycardia episode, beats")
        self.btn_classify = QtWidgets.QPushButton("Classify events")
        self.btn_classify.setProperty("primary", True)
        self.btn_classify.setEnabled(False)
        self.btn_classify.clicked.connect(host.analysis_ctrl.run_arrhythmia_analysis)
        row.addWidget(self.btn_classify)
        row.addStretch(1)
        lay.addWidget(bar)
        host.ent_arr_baseline = Value(self.sp_baseline.value)
        host.ent_arr_brady_pct = Value(self.sp_brady.value)
        host.ent_arr_min_beats = Value(lambda: int(self.sp_min_beats.value()))
        host.lbl_arrhythmia_status = StatusLabel("Detect the peaks, then classify.")
        host.lbl_arrhythmia_status.setProperty("muted", True)
        host.lbl_arrhythmia_status.setWordWrap(True)
        lay.addWidget(host.lbl_arrhythmia_status)

        ribbon = host.slot("arr_ribbon", self, 12.0, 2.0)
        ribbon.setMinimumHeight(170)
        lay.addWidget(ribbon)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._selected)
        split.addWidget(self.table)
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        head = QtWidgets.QHBoxLayout()
        host.lbl_arr_event_title = StatusLabel("Select an event")
        head.addWidget(host.lbl_arr_event_title, 1)
        head.addWidget(QtWidgets.QLabel("Window (s)"))
        self.sp_win = QtWidgets.QDoubleSpinBox()
        self.sp_win.setRange(0.1, 30.0)
        self.sp_win.setValue(3.0)
        self.sp_win.valueChanged.connect(self._window_changed)
        head.addWidget(self.sp_win)
        rl.addLayout(head)
        host.ent_arr_win = Value(self.sp_win.value)
        detail = host.slot("arr_detail", right, 9.0, 4.0)
        detail.setMinimumHeight(360)
        rl.addWidget(detail, 1)
        split.addWidget(right)
        split.setSizes([480, 900])
        lay.addWidget(split, 1)
        host.on_events_list = self.show_events

    def peaks_changed(self) -> None:
        self.btn_classify.setEnabled(self.win.recording.detection.rpeaks_ok is not None)

    def show_events(self, events: list) -> None:
        """The event list (AnalysisController.build_event_cards, as a table)."""
        self.table.setRowCount(len(events))
        for i, ev in enumerate(events):
            delta = f"{ev.delta_pct:+.0f} %" if ev.baseline_hr else ""
            values = (str(i + 1), event_kind_label(ev.kind), f"{ev.t_start:.2f}",
                      f"{ev.t_end:.2f}", str(ev.n_beats), f"{ev.hr_mean:.0f}", delta,
                      ev.severity)
            for c, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                if c in (0, 2, 3, 4, 5, 6):
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                                          | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(i, c, item)
        self.table.resizeColumnsToContents()

    def _selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        self.win.figures._select_arrhythmia_event(rows[0].row())
        a = self.win.recording.analysis
        self.sp_win.blockSignals(True)
        self.sp_win.setValue(a.arr_win)        # the event may have widened it
        self.sp_win.blockSignals(False)

    def _window_changed(self, value: float) -> None:
        a = self.win.recording.analysis
        if a.arr_selected_idx >= 0:
            a.arr_win = float(value)
            self.win.figures._draw_arr_detail()

    def clear(self) -> None:
        self.table.setRowCount(0)
        for name in ("arr_ribbon", "arr_detail"):
            self.win.figures.slot(name).clear()
        self.win.figures.lbl_arr_event_title.configure(text="Select an event")
