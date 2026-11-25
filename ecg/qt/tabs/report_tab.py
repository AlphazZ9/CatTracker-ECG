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

from PySide6 import QtCore, QtWidgets

from ecg.qt.figure_host import TextBox

if TYPE_CHECKING:
    from ecg.qt.main_window import MainWindow

EXPORTS = (("Excel…", "export_excel"), ("PDF report…", "export_pdf_report"),
           ("Figures (ZIP)…", "export_zip"), ("Prism…", "export_prism"),
           ("RR series (CSV)…", "export_rr_csv"), ("Events (PDF)…", "export_arrhythmia_pdf"),
           ("Copy text", "copy_report"), ("Save text…", "save_report_txt"))


class ReportTab(QtWidgets.QWidget):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.win = window
        lay = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QFrame(objectName="Card")
        row = QtWidgets.QHBoxLayout(bar)
        self.buttons: "dict[str, QtWidgets.QPushButton]" = {}
        ctl = window.figures.export_ctrl
        for label, method in EXPORTS:
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(lambda _c=False, m=method: self._export(m))
            row.addWidget(b)
            self.buttons[method] = b
        row.addStretch(1)
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh)
        row.addWidget(self.btn_refresh)
        lay.addWidget(bar)
        self._ctl = ctl

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        top = QtWidgets.QWidget()
        tl = QtWidgets.QHBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        self.provenance = QtWidgets.QTableWidget(0, 2)
        self.provenance.setHorizontalHeaderLabels(["Provenance", ""])
        self.provenance.verticalHeader().setVisible(False)
        self.provenance.horizontalHeader().setStretchLastSection(True)
        self.provenance.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        tl.addWidget(self.provenance, 1)
        self.lines = QtWidgets.QTableWidget(0, 4)
        self.lines.setHorizontalHeaderLabels(["Include", "Module", "State", "Result"])
        self.lines.verticalHeader().setVisible(False)
        self.lines.horizontalHeader().setStretchLastSection(True)
        self.lines.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.lines.itemChanged.connect(self._tick_changed)
        tl.addWidget(self.lines, 1)
        split.addWidget(top)
        self.text = TextBox()
        split.addWidget(self.text)
        split.setSizes([300, 500])
        lay.addWidget(split, 1)
        self.lbl_left_out = QtWidgets.QLabel("")
        self.lbl_left_out.setProperty("warn", True)
        lay.addWidget(self.lbl_left_out)
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
        has = self.win.recording.analysis.results is not None
        for m in ("export_excel", "export_pdf_report", "export_zip", "export_prism",
                  "copy_report", "save_report_txt"):
            self.buttons[m].setEnabled(has)
        self.buttons["export_rr_csv"].setEnabled(self.win.recording.detection.rpeaks_ok is not None)
        self.buttons["export_arrhythmia_pdf"].setEnabled(
            bool(self.win.recording.analysis.arrhythmia_events))
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
