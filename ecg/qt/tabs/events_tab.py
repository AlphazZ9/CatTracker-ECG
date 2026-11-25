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
        self.btn_copy = QtWidgets.QPushButton("Copy for Excel")
        self.btn_copy.setToolTip("The events as tab-separated text: paste them into Excel")
        self.btn_copy.clicked.connect(host.analysis_ctrl.copy_arrhythmia_tsv)
        row.addWidget(self.btn_copy)
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
        nav = host.nav_ctrl
        for text, tip, fn in (
                ("◀ Previous", "The event before the one selected", lambda: self.step_event(-1)),
                ("Next ▶", "The event after the one selected", lambda: self.step_event(+1))):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            head.addWidget(b)
        self.btn_edit = QtWidgets.QPushButton("Edit peaks here")
        self.btn_edit.setCheckable(True)
        self.btn_edit.setToolTip("Left click near a peak on the trace below: exclude / restore "
                                 "it. Right click: add a peak at the local maximum (or remove "
                                 "one added by hand). Undo: Ctrl+Z.")
        self.btn_edit.toggled.connect(self._edit_toggled)
        head.addWidget(self.btn_edit)
        for text, step in (("←", -1), ("→", +1)):
            b = QtWidgets.QPushButton(text)
            b.setToolTip("Pan the trace")
            b.setMaximumWidth(36)
            b.clicked.connect(lambda _c=False, d=step: nav.arr_navigate(d))
            head.addWidget(b)
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
        detail.canvas.mpl_connect("button_press_event", host.analysis_ctrl.on_arr_detail_click)
        rl.addWidget(detail, 1)
        split.addWidget(right)
        split.setSizes([480, 900])
        lay.addWidget(split, 1)
        host.on_events_list = self.show_events
        # What AnalysisController.on_arr_detail_click calls on the app (Tk's
        # DetectionController): the undo snapshot, the peaks rebuilt, the
        # Signal & Detection trace and the Undo buttons follow.
        # (window.detection is set once every tab is built: looked up at call time.)
        def _push_undo() -> None:
            from ecg.core.peak_editing import push_undo
            d = window.recording.detection
            window.detection.mark_peaks_edited()
            push_undo(d.edit_undo, d.edit_redo, d.manual_excluded, d.manual_added)
            window.detection._update_undo_buttons()
        host._push_edit_undo = _push_undo
        host._run_detection = lambda _thr=None: window.detection.apply_threshold()
        host._update_undo_btns = lambda: window.detection._update_undo_buttons()
        host._draw_detail = lambda *_a: None        # apply_threshold redraws it
        host.btn_arr_edit = Value(lambda: None)
        host.lbl_arr_edit_hint = Value(lambda: None)

    def _edit_toggled(self, on: bool) -> None:
        """AnalysisController.toggle_arr_edit_mode, the button holding the state."""
        a = self.win.recording.analysis
        a.arr_edit_mode = bool(on)
        self.btn_edit.setText("Edit Mode ON" if on else "Edit peaks here")
        if a.arr_selected_idx >= 0:
            self.win.figures._draw_arr_detail()

    def step_event(self, direction: int) -> None:
        """NavigationController.arr_step_event, the table row selected with it."""
        a = self.win.recording.analysis
        n = len(a.arrhythmia_events)
        if not n:
            return
        cur = a.arr_selected_idx
        idx = (0 if direction > 0 else n - 1) if cur < 0 else cur + direction
        if 0 <= idx < n:
            self.table.selectRow(idx)          # -> _selected -> select_arrhythmia_event

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
