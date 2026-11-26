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


CARD_NAMES = {"hr": "Heart rate", "sdnn": "SDNN", "rmssd": "RMSSD", "pnn6": "pNN6",
              "lf_hf": "LF / HF", "sd1_sd2": "SD1 / SD2"}


class _Cards(dict):
    """KPI tiles by card key, created on first use in *layout*: the heart
    rate first and in red, the others plain."""

    def __init__(self, layout: QtWidgets.QHBoxLayout) -> None:
        super().__init__()
        self._layout = layout

    def __bool__(self) -> bool:      # "the page is built" for PlotController
        return True

    def __missing__(self, key: str) -> dict:
        hero = key == "hr"
        box = QtWidgets.QFrame(objectName="Hero" if hero else "Card")
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(2)
        name = QtWidgets.QLabel(CARD_NAMES.get(key, key.split(".")[-1]))
        name.setProperty("muted", not hero)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        value = StatusLabel("—")
        value.setProperty("role", "value")
        if hero:
            value.setStyleSheet("font-size: 26pt; font-weight: 700; color: white;")
        unit = StatusLabel("")
        unit.setProperty("muted", not hero)
        row.addWidget(value, 0, QtCore.Qt.AlignmentFlag.AlignBottom)
        row.addWidget(unit, 0, QtCore.Qt.AlignmentFlag.AlignBottom)
        row.addStretch(1)
        note = StatusLabel("")
        note.setProperty("muted", not hero)
        note.setWordWrap(True)
        lay.addWidget(name)
        lay.addLayout(row)
        lay.addWidget(note)
        self._layout.addWidget(box, 3 if hero else 2)
        widgets = self[key] = {"value": value, "unit": unit, "note": note, "name": name}
        return widgets


class RhythmTab(QtWidgets.QScrollArea):
    """Rhythm & HRV: the key numbers first (heart rate in red), then the
    tachogram, Poincaré, spectrum and RR distribution, the metric table, and
    the raw text tables on demand."""

    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        from ecg.qt.widgets.shell import card, page_header
        self.win = window
        host = window.figures
        self.setWidgetResizable(True)
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(24, 16, 24, 20)
        lay.setSpacing(14)
        self.setWidget(page)

        self.lbl_info = QtWidgets.QLabel("Detect the peaks, then click Analyze.")
        lay.addLayout(page_header("Rhythm & HRV", self.lbl_info))
        cards_row = QtWidgets.QHBoxLayout()
        cards_row.setSpacing(12)
        host._rhythm_cards = _Cards(cards_row)
        lay.addLayout(cards_row)
        for key in ("hr", "sdnn", "rmssd", "pnn6", "lf_hf", "sd1_sd2"):   # in this order
            host._rhythm_cards[key]
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(14)
        # The two chart rows share the free height of the page: the charts
        # grow with the window rather than leaving a blank band below.
        grid.setRowStretch(0, 3)
        grid.setRowStretch(1, 2)
        lay.addLayout(grid)
        self._grid, self._cards_row = grid, cards_row

        # Tachogram, with its flagged-beat controls.
        spikes = QtWidgets.QWidget()
        srow = QtWidgets.QHBoxLayout(spikes)
        srow.setContentsMargins(0, 0, 0, 0)
        self.btn_spikes = QtWidgets.QPushButton("Flagged beats")
        self.btn_spikes.setCheckable(True)
        self.btn_spikes.setChecked(bool(host.ui.show_spike_markers))
        self.btn_spikes.setToolTip("Show / hide the flagged beats (▲ / ▼) of the tachogram: "
                                   "the trace and the mean ± SD band stay")
        self.btn_spikes.toggled.connect(self._spikes_toggled)
        srow.addWidget(self.btn_spikes)
        self.btn_review = QtWidgets.QPushButton("Review ›")
        self.btn_review.setToolTip("The next flagged beat (▲ / ▼), centred in Signal; press "
                                   "again for the following one")
        self.btn_review.clicked.connect(lambda: window.detection.step_spike(+1))
        srow.addWidget(self.btn_review)
        # The charts on screen (PyQtGraph); the Matplotlib figures of the same
        # results are kept, hidden, for the exports (drawn when one reads them).
        from ecg.qt.widgets.rhythm_charts import RhythmCharts
        from ecg.qt.widgets.shell import Segmented
        self.charts = RhythmCharts(on_time_clicked=self._show_beat, parent=self)
        for name in ("rr", "poincare", "psd", "rr_hist"):
            host.slot(name, page, 6.0, 4.0).hide()
        self.seg_unit = Segmented(["HR", "RR"])
        self.seg_unit.buttons[0].setChecked(True)
        for b in self.seg_unit.buttons:
            b.clicked.connect(lambda _c=False, u=b.text(): self.charts.set_unit(u))
        srow.insertWidget(0, self.seg_unit)
        self.lbl_flagged = QtWidgets.QLabel("")
        self.lbl_flagged.setProperty("muted", True)
        srow.insertWidget(0, self.lbl_flagged)
        grid.addWidget(card("Heart rate over time", self.charts.tacho, 0, spikes), 0, 0)

        # Poincaré, with the non-linear recompute and the entropy cap.
        nl = QtWidgets.QWidget()
        nrow = QtWidgets.QHBoxLayout(nl)
        nrow.setContentsMargins(0, 0, 0, 0)
        nrow.addWidget(QtWidgets.QLabel("Entropy cap (RR)"))
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
        self.btn_nonlin = QtWidgets.QPushButton("Recompute")
        self.btn_nonlin.setToolTip("Non-linear HRV again on the current peaks, window and RR "
                                   "bounds")
        self.btn_nonlin.clicked.connect(window.recompute_nonlinear)
        nrow.addWidget(self.btn_nonlin)
        host.ent_entropy_cap = Field(self.sp_entropy, text=True)
        pc_box = QtWidgets.QWidget()
        pcl = QtWidgets.QVBoxLayout(pc_box)
        pcl.setContentsMargins(0, 0, 0, 0)
        host.lbl_nonlin_status = StatusLabel("")
        host.lbl_nonlin_status.setWordWrap(True)
        pcl.addWidget(host.lbl_nonlin_status)
        prow = QtWidgets.QHBoxLayout()
        prow.addWidget(self.charts.poincare, 3)
        stats = QtWidgets.QVBoxLayout()
        self.lbl_sd = {}
        for key, label in (("sd1", "SD1"), ("sd2", "SD2"), ("ratio", "SD1 / SD2")):
            name = QtWidgets.QLabel(label)
            name.setProperty("muted", True)
            val = QtWidgets.QLabel("—")
            val.setProperty("role", "value")
            stats.addWidget(name)
            stats.addWidget(val)
            self.lbl_sd[key] = val
        stats.addStretch(1)
        prow.addLayout(stats, 1)
        pcl.addLayout(prow, 3)
        pcl.addWidget(self.charts.mse, 1)
        grid.addWidget(card("Poincaré and entropy", pc_box, 0, nl), 0, 1)

        # Spectrum.
        psd_box = QtWidgets.QWidget()
        psd_lay = QtWidgets.QVBoxLayout(psd_box)
        psd_lay.setContentsMargins(0, 0, 0, 0)
        host.lbl_freq_status = StatusLabel("")
        host.lbl_freq_status.setWordWrap(True)
        psd_lay.addWidget(host.lbl_freq_status)
        psd_lay.addWidget(self.charts.psd, 1)
        self.btn_freq = QtWidgets.QPushButton("Recompute")
        self.btn_freq.setToolTip("Frequency HRV again on the current peaks, analysis window and "
                                 "RR bounds")
        self.btn_freq.clicked.connect(window.recompute_freq)
        grid.addWidget(card("Power spectrum", psd_box, 0, self.btn_freq), 1, 0)

        grid.addWidget(card("RR distribution (valid intervals)", self.charts.hist, 0), 1, 1)

        # The metric table (ecg.core.metric_table.rhythm_rows), with its basis:
        # as tall as its rows, so the page is the only thing that scrolls.
        table_box = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(table_box)
        tl.setContentsMargins(0, 0, 0, 0)
        host._rhythm_labels = {k: StatusLabel("") for k in ("excluded", "nonlinear_basis",
                                                            "asymmetry")}
        for lbl in host._rhythm_labels.values():
            lbl.setProperty("muted", True)
            lbl.setWordWrap(True)
        tl.addWidget(host._rhythm_labels["excluded"])
        self.chk_all = QtWidgets.QCheckBox("Show all metrics")
        self.chk_all.toggled.connect(self._show_all_toggled)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Metric", "Value", "Unit", "Basis"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        tl.addWidget(self.table)
        tl.addWidget(host._rhythm_labels["nonlinear_basis"])
        tl.addStretch(1)
        grid.addWidget(card("Metrics", table_box, 0, self.chk_all), 2, 0)
        asym = QtWidgets.QWidget()
        al = QtWidgets.QVBoxLayout(asym)
        al.setContentsMargins(0, 0, 0, 0)
        al.addWidget(host._rhythm_labels["asymmetry"])
        # On screen in PyQtGraph; the Matplotlib figure stays, hidden, for
        # the exports (it took ~0.3 s to draw each time the tab was shown).
        from ecg.qt.widgets.dist_charts import AsymmetryCharts
        self.asym_charts = AsymmetryCharts()
        slot = host.slot("asymmetry", page, 6.0, 3.0)
        slot.hide()
        slot.on_clear = self.asym_charts.clear
        host._screen_charts["asymmetry"] = self.asym_charts.set_data
        al.addWidget(self.asym_charts, 1)
        grid.addWidget(card("RR asymmetry", asym, 300), 2, 1)
        host.on_rhythm_table = self.show_table
        self._table: "MetricTable | None" = None
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)

        # The raw text tables (right click: copy as a table), on demand.
        self.btn_text = QtWidgets.QPushButton("Show the text tables")
        self.btn_text.setCheckable(True)
        self.btn_text.setProperty("flat", True)
        lay.addWidget(self.btn_text, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        raw = QtWidgets.QWidget()
        raw_lay = QtWidgets.QHBoxLayout(raw)
        raw_lay.setContentsMargins(0, 0, 0, 0)
        for title, attr in (("HRV time domain", "txt_td"), ("Frequency domain", "txt_fd"),
                            ("Non-linear", "txt_nl"), ("RR intervals", "txt_rr")):
            col = QtWidgets.QVBoxLayout()
            col.addWidget(QtWidgets.QLabel(title))
            box = TextBox()
            box.setMinimumHeight(260)
            setattr(host, attr, box)
            col.addWidget(box, 1)
            raw_lay.addLayout(col, 1)
        self.text_card = card(None, raw)
        self.text_card.setVisible(False)
        self.btn_text.toggled.connect(self._text_toggled)
        lay.addWidget(self.text_card)
        lay.addStretch(1)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._fit_charts()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._fit_charts()

    def _fit_charts(self) -> None:
        """The two chart rows fill the visible height under the key numbers
        (3 : 2), so the charts fit the window whatever its size; the metric
        table follows below, reached by scrolling."""
        grid = getattr(self, "_grid", None)
        if grid is None:
            return
        top = self._cards_row.geometry().bottom() + 14
        avail = self.viewport().height() - top - grid.spacing() - 16
        avail = max(avail, 2 * 230)
        grid.setRowMinimumHeight(0, int(avail * 0.56))
        grid.setRowMinimumHeight(1, int(avail * 0.44))

    def refresh_charts(self) -> None:
        """The charts from the current results (after each step of Analyze,
        a recompute, a session restored, a theme change)."""
        host = self.win.figures
        r = self.win.recording.analysis.results
        psd = None
        if r is not None and r.get("rr_ms") is not None:
            try:
                psd = host.plot_ctrl.psd_for_display(r)
            except Exception:                        # a chart never breaks the page
                psd = None
        spikes = host.ui.rr_spike_times if host.ui.show_spike_markers else None
        self.charts.set_results(r, spikes, self.win.recording.analysis.annotations, psd)
        self.win.detection.refresh_to_check()
        n = 0 if host.ui.rr_spike_times is None else len(host.ui.rr_spike_times)
        self.lbl_flagged.setText(f"{n} beats flagged" if r is not None and n else "")
        nl = (r or {}).get("hrv_nonlin")
        vals = {}
        for key, col in (("sd1", "HRV_SD1"), ("sd2", "HRV_SD2"), ("ratio", "HRV_SD1SD2")):
            try:
                vals[key] = float(nl[col].values[0])
            except Exception:
                vals[key] = float("nan")
        import math
        for key, v in vals.items():
            unit = "" if key == "ratio" else " ms"
            self.lbl_sd[key].setText(f"{v:.2f}{unit}" if key == "ratio" and math.isfinite(v)
                                     else f"{v:.1f}{unit}" if math.isfinite(v) else "—")

    def _show_beat(self, t: float) -> None:
        """A click on the heart-rate chart: that beat, centred in Signal."""
        self.win.detection.center_on(t)

    def apply_palette(self, p) -> None:
        self.asym_charts.apply_palette(p)
        self.charts.apply_palette(p)

    def _text_toggled(self, on: bool) -> None:
        self.text_card.setVisible(on)
        self.btn_text.setText("Hide the text tables" if on else "Show the text tables")

    def _spikes_toggled(self, on: bool) -> None:
        """ECGApp._toggle_spike_markers: only the flagged-beat layer."""
        host = self.win.figures
        host.ui.show_spike_markers = bool(on)
        r = self.win.recording.analysis.results
        if r is not None:
            host.plot_ctrl.plot_rr(r)
        self.refresh_charts()

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
        self._fit_table()

    def _fit_table(self) -> None:
        """The table as tall as its rows: no scroll inside the page's scroll."""
        h = self.table.horizontalHeader().height() + 2 * self.table.frameWidth() + 2
        h += sum(self.table.rowHeight(r) for r in range(self.table.rowCount()))
        self.table.setFixedHeight(max(h, 80))

    def clear(self) -> None:
        self._table = None
        self.table.setRowCount(0)
        self._fit_table()
        for name in ("rr", "poincare", "psd", "rr_hist", "asymmetry"):
            self.win.figures.slot(name).clear()
        self.charts.set_results(None)
        for v in self.lbl_sd.values():
            v.setText("—")
        self.lbl_flagged.setText("")
        for attr in ("txt_rr", "txt_td", "txt_fd", "txt_nl"):
            getattr(self.win.figures, attr).setPlainText("")
        self.win.figures.lbl_freq_status.setText("")
        self.win.figures.lbl_nonlin_status.setText("")
        self.lbl_info.setText("Detect the peaks, then click Analyze.")
