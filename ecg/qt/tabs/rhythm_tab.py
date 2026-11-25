# -*- coding: utf-8 -*-
"""
ecg.qt.tabs.rhythm_tab
----------------------
Rhythm & HRV: the RR / HR tachogram, Poincaré, power spectrum, RR
distribution, and the time-domain / frequency / non-linear HRV tables.

The figures and texts are drawn by the Tk interface's own figure code
through the window's FigureHost (ecg.qt.figure_host), so they are the same
as in the Tk interface and the exports. Analyze and its frequency /
non-linear steps run in the main window (MainWindow.analyze).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from ecg.core.metric_table import TIER_ALL, MetricTable, rhythm_cards, rhythm_rows
from ecg.core.models import MouseECG
from ecg.qt.figure_host import Field, StatusLabel, TextBox

if TYPE_CHECKING:
    from ecg.qt.main_window import MainWindow


class _Cards(dict):
    """KPI tiles by card key, created on first use in *layout*."""

    def __init__(self, layout: QtWidgets.QHBoxLayout) -> None:
        super().__init__()
        self._layout = layout

    def __bool__(self) -> bool:      # "the page is built" for PlotController
        return True

    def __missing__(self, key: str) -> dict:
        box = QtWidgets.QFrame(objectName="Card")
        lay = QtWidgets.QVBoxLayout(box)
        name = QtWidgets.QLabel(key.split(".")[-1])
        name.setProperty("muted", True)
        value = StatusLabel("—")
        value.setStyleSheet("font-size: 16pt; font-weight: 600;")
        unit = StatusLabel("")
        unit.setProperty("muted", True)
        note = StatusLabel("")
        note.setProperty("muted", True)
        note.setWordWrap(True)
        for w in (name, value, unit, note):
            lay.addWidget(w)
        self._layout.addWidget(box)
        widgets = self[key] = {"value": value, "unit": unit, "note": note, "name": name}
        return widgets


class RhythmTab(QtWidgets.QScrollArea):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.win = window
        host = window.figures
        self.setWidgetResizable(True)
        page = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(page)
        self.setWidget(page)

        def card(title: str, widget: QtWidgets.QWidget, min_h: int = 320) -> QtWidgets.QFrame:
            f = QtWidgets.QFrame(objectName="Card")
            lay = QtWidgets.QVBoxLayout(f)
            lbl = QtWidgets.QLabel(f"<b>{title}</b>")
            lay.addWidget(lbl)
            widget.setMinimumHeight(min_h)
            lay.addWidget(widget, 1)
            return f

        self.lbl_info = QtWidgets.QLabel("Detect the peaks, then click Analyze.")
        self.lbl_info.setProperty("muted", True)
        grid.addWidget(self.lbl_info, 0, 0, 1, 2)
        cards_row = QtWidgets.QHBoxLayout()
        host._rhythm_cards = _Cards(cards_row)
        grid.addLayout(cards_row, 5, 0, 1, 2)

        # The metric table (ecg.core.metric_table.rhythm_rows), with its basis.
        table_box = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(table_box)
        tl.setContentsMargins(0, 0, 0, 0)
        head = QtWidgets.QHBoxLayout()
        host._rhythm_labels = {k: StatusLabel("") for k in ("excluded", "nonlinear_basis",
                                                            "asymmetry")}
        for lbl in host._rhythm_labels.values():
            lbl.setProperty("muted", True)
            lbl.setWordWrap(True)
        head.addWidget(host._rhythm_labels["excluded"], 1)
        self.chk_all = QtWidgets.QCheckBox("Show all metrics")
        self.chk_all.toggled.connect(self._show_all_toggled)
        head.addWidget(self.chk_all)
        tl.addLayout(head)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Metric", "Value", "Unit", "Basis"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        tl.addWidget(self.table, 1)
        tl.addWidget(host._rhythm_labels["nonlinear_basis"])
        grid.addWidget(card("Metrics", table_box, 380), 6, 0)
        asym = QtWidgets.QWidget()
        al = QtWidgets.QVBoxLayout(asym)
        al.setContentsMargins(0, 0, 0, 0)
        al.addWidget(host._rhythm_labels["asymmetry"])
        al.addWidget(host.slot("asymmetry", page, 6.0, 3.0), 1)
        grid.addWidget(card("RR asymmetry", asym, 380), 6, 1)
        host.on_rhythm_table = self.show_table
        self._table: "MetricTable | None" = None
        tacho = QtWidgets.QWidget()
        tl2 = QtWidgets.QVBoxLayout(tacho)
        tl2.setContentsMargins(0, 0, 0, 0)
        srow = QtWidgets.QHBoxLayout()
        srow.addStretch(1)
        self.btn_spikes = QtWidgets.QPushButton("▲▼ Spike markers")
        self.btn_spikes.setCheckable(True)
        self.btn_spikes.setChecked(bool(host.ui.show_spike_markers))
        self.btn_spikes.setToolTip("Show / hide the flagged beats (▲ / ▼) of the tachogram: "
                                   "the trace and the mean ± SD band stay")
        self.btn_spikes.toggled.connect(self._spikes_toggled)
        srow.addWidget(self.btn_spikes)
        self.btn_review = QtWidgets.QPushButton("Review ▶")
        self.btn_review.setToolTip("The next flagged beat (▲ / ▼), centred in Signal & "
                                   "Detection; press again for the following one")
        self.btn_review.clicked.connect(lambda: window.detection.step_spike(+1))
        srow.addWidget(self.btn_review)
        tl2.addLayout(srow)
        tl2.addWidget(host.slot("rr", page, 9.0, 4.4), 1)
        grid.addWidget(card("RR / HR tachogram", tacho, 420), 1, 0)
        grid.addWidget(card("Poincaré", host.slot("poincare", page, 5.0, 4.4), 420), 1, 1)

        psd_box = QtWidgets.QWidget()
        psd_lay = QtWidgets.QVBoxLayout(psd_box)
        psd_lay.setContentsMargins(0, 0, 0, 0)
        frow = QtWidgets.QHBoxLayout()
        self.btn_freq = QtWidgets.QPushButton("Recompute Frequency")
        self.btn_freq.setToolTip("Frequency HRV again on the current peaks, analysis window and "
                                 "RR bounds")
        self.btn_freq.clicked.connect(window.recompute_freq)
        frow.addWidget(self.btn_freq)
        host.lbl_freq_status = StatusLabel("")
        host.lbl_freq_status.setWordWrap(True)
        frow.addWidget(host.lbl_freq_status, 1)
        psd_lay.addLayout(frow)
        psd_lay.addWidget(host.slot("psd", page, 6.0, 3.6), 1)
        grid.addWidget(card("Power spectral density", psd_box, 360), 2, 0)

        dist = QtWidgets.QWidget()
        dist_lay = QtWidgets.QVBoxLayout(dist)
        dist_lay.setContentsMargins(0, 0, 0, 0)
        dist_lay.addWidget(host.slot("rr_hist", page, 5.0, 2.6), 1)
        host.txt_rr = TextBox()
        host.txt_rr.setMaximumHeight(120)
        dist_lay.addWidget(host.txt_rr)
        grid.addWidget(card("RR distribution (valid intervals)", dist, 360), 2, 1)

        raw = QtWidgets.QWidget()
        raw_lay = QtWidgets.QHBoxLayout(raw)
        raw_lay.setContentsMargins(0, 0, 0, 0)
        for title, attr in (("HRV time domain", "txt_td"), ("Frequency domain", "txt_fd"),
                            ("Non-linear", "txt_nl")):
            col = QtWidgets.QVBoxLayout()
            col.addWidget(QtWidgets.QLabel(title))
            if attr == "txt_nl":
                nrow = QtWidgets.QHBoxLayout()
                self.btn_nonlin = QtWidgets.QPushButton("Recompute Non-linear")
                self.btn_nonlin.clicked.connect(window.recompute_nonlinear)
                nrow.addWidget(self.btn_nonlin)
                nrow.addWidget(QtWidgets.QLabel("Entropy cap (RR):"))
                # Runtime cap of the O(N²) entropy block only (SampEn, ApEn, MSE…):
                # Poincaré and DFA always use every beat of the analysis window.
                self.sp_entropy = QtWidgets.QSpinBox()
                self.sp_entropy.setRange(0, 10_000_000)
                self.sp_entropy.setValue(int(MouseECG.ENTROPY_MAX_BEATS))
                self.sp_entropy.setToolTip("Valid RR intervals used by the entropy block, for "
                                           "runtime (0 = no cap). SD1 / SD2 and DFA always use "
                                           "every valid RR of the window. Takes effect at the "
                                           "next Analyze or Recompute.")
                nrow.addWidget(self.sp_entropy)
                nrow.addStretch(1)
                col.addLayout(nrow)
                host.ent_entropy_cap = Field(self.sp_entropy, text=True)
                host.lbl_nonlin_status = StatusLabel("")
                col.addWidget(host.lbl_nonlin_status)
            box = TextBox()
            setattr(host, attr, box)
            col.addWidget(box, 1)
            raw_lay.addLayout(col, 1)
        grid.addWidget(card("HRV tables (right click: copy as a table)", raw, 300), 3, 0, 1, 2)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)

    def _spikes_toggled(self, on: bool) -> None:
        """ECGApp._toggle_spike_markers: only the flagged-beat layer."""
        host = self.win.figures
        host.ui.show_spike_markers = bool(on)
        r = self.win.recording.analysis.results
        if r is not None:
            host.plot_ctrl.plot_rr(r)

    def _show_all_toggled(self, on: bool) -> None:
        self.win.figures.ui.hrv_show_all = on
        if self._table is not None:
            self.show_table(self._table)

    def show_table(self, t: MetricTable) -> None:
        """PlotController.refresh_rhythm_table, as a Qt table."""
        self._table = t
        for card in rhythm_cards(t):               # the tiles' titles
            self.win.figures._rhythm_cards[card.key]["name"].setText(card.name)
        rows = rhythm_rows(t, show_all=self.win.figures.ui.hrv_show_all)
        self.table.setRowCount(0)
        group = None
        bold = QtGui.QFont()
        bold.setBold(True)
        for row in rows:
            if row.group != group:
                group = row.group
                r = self.table.rowCount()
                self.table.insertRow(r)
                item = QtWidgets.QTableWidgetItem(group)
                item.setFont(bold)
                self.table.setItem(r, 0, item)
                self.table.setSpan(r, 0, 1, 4)
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, text in enumerate((row.label, row.value, row.unit, row.basis)):
                item = QtWidgets.QTableWidgetItem(text)
                if c == 1:
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                                          | QtCore.Qt.AlignmentFlag.AlignVCenter)
                if row.tier == TIER_ALL:
                    item.setForeground(QtGui.QColor(self.palette().color(
                        QtGui.QPalette.ColorRole.PlaceholderText)))
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()

    def clear(self) -> None:
        self._table = None
        self.table.setRowCount(0)
        for name in ("rr", "poincare", "psd", "rr_hist", "asymmetry"):
            self.win.figures.slot(name).clear()
        for attr in ("txt_rr", "txt_td", "txt_fd", "txt_nl"):
            getattr(self.win.figures, attr).setPlainText("")
        self.win.figures.lbl_freq_status.setText("")
        self.win.figures.lbl_nonlin_status.setText("")
        self.lbl_info.setText("Detect the peaks, then click Analyze.")
