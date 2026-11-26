# -*- coding: utf-8 -*-
"""
ecg.qt.tabs.report_tab
----------------------
Report: what the exports will contain (provenance, one line per module with
its state, its key result and an Include tick) and the text report, with
the exports -- Excel, PDF, ZIP of figures, GraphPad Prism, RR series (CSV),
abnormal events (PDF).

Everything comes from the Tk interface's ExportController (through the
FigureHost): the same data, the same files. Its file dialogs and message
boxes are Qt ones.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from ecg.qt.figure_host import TextBox

if TYPE_CHECKING:
    from ecg.qt.main_window import MainWindow

EXPORTS = (("Excel…", "export_excel"), ("PDF report…", "export_pdf_report"),
           ("Figures (ZIP)…", "export_zip"), ("Prism…", "export_prism"),
           ("RR series (CSV)…", "export_rr_csv"), ("Events (PDF)…", "export_arrhythmia_pdf"),
           ("Copy text", "copy_report"), ("Save text…", "save_report_txt"))


class ReportTab(QtWidgets.QWidget):
    """What goes out (one line per module, its state, an Include tick), the
    formats (pick several, export them in one go to one folder), the
    provenance every file carries, and the text report on demand."""

    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        from ecg.qt.widgets.shell import card
        self.win = window
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(24, 16, 24, 16)
        lay.setSpacing(12)
        ctl = window.figures.export_ctrl
        self._ctl = ctl
        # The Tk arrangement: Provenance on the left; on the right what is in
        # this report (one line per module, its state, its result, Include)
        # with the export buttons under it; the text report on demand.
        self.buttons: "dict[str, QtWidgets.QPushButton]" = {}
        for label, method in EXPORTS:
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(lambda _c=False, m=method: self._export(m))
            self.buttons[method] = b
        self.buttons["export_excel"].setText("Export Excel")
        self.buttons["export_excel"].setProperty("primary", True)
        self.buttons["export_prism"].setText("GraphPad Prism")
        self.buttons["export_pdf_report"].setText("PDF report")
        self.buttons["save_report_txt"].setText("Save .txt")

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(14)
        lay.addLayout(body)
        self.provenance = QtWidgets.QTableWidget(0, 2)
        self.provenance.horizontalHeader().setVisible(False)
        self.provenance.verticalHeader().setVisible(False)
        self.provenance.horizontalHeader().setStretchLastSection(True)
        self.provenance.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.provenance.setShowGrid(False)
        self.provenance.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        body.addWidget(card("Provenance", self.provenance), 2)

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(10)
        self.lines = QtWidgets.QTableWidget(0, 4)
        self.lines.setHorizontalHeaderLabels(["Include", "Module", "State", "Key result"])
        self.lines.verticalHeader().setVisible(False)
        self.lines.horizontalHeader().setStretchLastSection(True)
        self.lines.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.lines.setShowGrid(False)
        self.lines.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.lines.itemChanged.connect(self._tick_changed)
        rl.addWidget(self.lines)
        bar = QtWidgets.QHBoxLayout()
        for m in ("export_excel", "export_prism", "export_pdf_report", "copy_report",
                  "save_report_txt"):
            bar.addWidget(self.buttons[m])
        more = QtWidgets.QToolButton()
        more.setText("More")
        more.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QtWidgets.QMenu(more)
        for m in ("export_zip", "export_rr_csv", "export_arrhythmia_pdf"):
            b = self.buttons[m]
            act = menu.addAction(b.text())
            act.triggered.connect(b.click)
        menu.aboutToShow.connect(lambda: [a.setEnabled(self.buttons[m].isEnabled()) for a, m in
                                          zip(menu.actions(), ("export_zip", "export_rr_csv",
                                                               "export_arrhythmia_pdf"))])
        more.setMenu(menu)
        bar.addWidget(more)
        bar.addStretch(1)
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_refresh.setProperty("flat", True)
        self.btn_refresh.clicked.connect(self.refresh)
        bar.addWidget(self.btn_refresh)
        rl.addLayout(bar)
        note = QtWidgets.QLabel("This screen, the text report, the PDF and the Metrics tables "
                                "of Excel and Prism read one table: each value with its unit, "
                                "its basis n and its state. The other Excel sheets and Prism "
                                "tables hold the series those values come from.")
        note.setProperty("muted", True)
        note.setWordWrap(True)
        rl.addWidget(note)
        self.lbl_left_out = QtWidgets.QLabel("")
        self.lbl_left_out.setWordWrap(True)
        self.warn_left_out = QtWidgets.QFrame(objectName="Warning")
        wl = QtWidgets.QHBoxLayout(self.warn_left_out)
        wl.setContentsMargins(14, 8, 14, 8)
        wl.addWidget(self.lbl_left_out)
        self.warn_left_out.hide()
        rl.addWidget(self.warn_left_out)
        rl.addStretch(1)
        body.addWidget(card("What is in this report", right), 3)
        for m in ("export_zip", "export_rr_csv", "export_arrhythmia_pdf"):
            self.buttons[m].setParent(self)
            self.buttons[m].hide()

        self.btn_text = QtWidgets.QPushButton("▸ Text report")
        self.btn_text.setCheckable(True)
        self.btn_text.setProperty("flat", True)
        self.btn_text.setStyleSheet("text-align: left; font-weight: 600;")
        lay.addWidget(self.btn_text, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.text = TextBox()
        self.text.setMinimumHeight(320)
        self.text.hide()
        self.btn_text.toggled.connect(lambda on: (self.text.setVisible(on),
                                                  self.btn_text.setText(
                                                      ("▾" if on else "▸") + " Text report")))
        lay.addWidget(self.text, 1)
        lay.addStretch(0)
        self._filling = False

    def _export(self, method: str) -> None:
        getattr(self._ctl, method)()
        self.refresh()

    def refresh(self) -> None:
        """ExportController.report_view: what the exports hold now."""
        if self.win.recording.signal.filepath is None:
            return
        view = self._ctl.report_view()
        self._filling = True
        self.provenance.setRowCount(len(view.provenance))
        for i, (name, value) in enumerate(view.provenance):
            self.provenance.setItem(i, 0, QtWidgets.QTableWidgetItem(str(name)))
            self.provenance.setItem(i, 1, QtWidgets.QTableWidgetItem(str(value)))
        self.provenance.resizeColumnToContents(0)
        self.lines.setRowCount(len(view.lines))
        for i, line in enumerate(view.lines):
            tick = QtWidgets.QTableWidgetItem("")
            tick.setData(QtCore.Qt.ItemDataRole.UserRole, (line.key, line.state))
            if line.can_include:
                tick.setFlags(tick.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                tick.setCheckState(QtCore.Qt.CheckState.Checked if line.include
                                   else QtCore.Qt.CheckState.Unchecked)
            else:
                tick.setFlags(tick.flags() & ~QtCore.Qt.ItemFlag.ItemIsEnabled)
            self.lines.setItem(i, 0, tick)
            for c, text in enumerate((line.label, line.state, line.result), start=1):
                self.lines.setItem(i, c, QtWidgets.QTableWidgetItem(text))
        self.lines.resizeColumnsToContents()
        self.text.setPlainText(view.text)
        self.lbl_left_out.setText(("Left out of the exports: " + ", ".join(view.left_out))
                                  if view.left_out else "")
        self.warn_left_out.setVisible(bool(view.left_out))
        has = self.win.recording.analysis.results is not None
        for m in ("export_excel", "export_pdf_report", "export_zip", "export_prism",
                  "copy_report", "save_report_txt"):
            self.buttons[m].setEnabled(has)
        self.buttons["export_rr_csv"].setEnabled(self.win.recording.detection.rpeaks_ok is not None)
        self.buttons["export_arrhythmia_pdf"].setEnabled(
            bool(self.win.recording.analysis.arrhythmia_events))
        h = self.lines.horizontalHeader().height() + 4 + sum(
            self.lines.rowHeight(r) for r in range(self.lines.rowCount()))
        self.lines.setFixedHeight(max(h, 80))
        self.provenance.resizeColumnToContents(0)
        self._filling = False

    def _tick_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        """An Include tick, kept with the state it was given in
        (UIState.report_include, report_view.include_choices)."""
        if self._filling or item.column() != 0:
            return
        key, state = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self.win.figures.ui.report_include[key] = [
            item.checkState() == QtCore.Qt.CheckState.Checked, state]
        self.refresh()
