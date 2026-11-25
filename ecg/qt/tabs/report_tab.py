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


FORMATS = (("export_excel", "Excel", "Every metric with its basis, and its series"),
           ("export_pdf_report", "PDF report", "One page: ECG, heart rate, key metrics"),
           ("export_prism", "GraphPad Prism", "A new file, or your template filled"),
           ("export_zip", "Figures (ZIP)", "Excel and every figure as PNG 300 dpi"),
           ("export_rr_csv", "RR series", "CSV, one line per beat"),
           ("export_arrhythmia_pdf", "Events PDF", "One annotated strip per episode"))
DEFAULT_FORMATS = ("export_excel", "export_pdf_report", "export_prism")


class _InFolder:
    """A file chooser that answers every save with the suggested file name
    in one folder (Export N files: one question for all of them)."""

    def __init__(self, folder: str, fallback) -> None:
        self.folder, self.fallback = folder, fallback

    def asksaveasfilename(self, initialfile: str = "", defaultextension: str = "", **_k) -> str:
        import os
        name = initialfile or f"export{defaultextension}"
        if defaultextension and not os.path.splitext(name)[1]:
            name += defaultextension
        return os.path.join(self.folder, name)

    def askdirectory(self, **_k) -> str:
        return self.folder

    def __getattr__(self, name):
        return getattr(self.fallback, name)


class ReportTab(QtWidgets.QWidget):
    """What goes out (one line per module, its state, an Include tick), the
    formats (pick several, export them in one go to one folder), the
    provenance every file carries, and the text report on demand."""

    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        from ecg.qt.widgets.shell import card, page_header
        self.win = window
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(24, 16, 24, 16)
        lay.setSpacing(12)
        sub = QtWidgets.QLabel("Choose what goes out, then the formats. Every file carries the "
                               "same provenance.")
        head = page_header("Report & exports", sub)
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh)
        head.addWidget(self.btn_refresh, 0, QtCore.Qt.AlignmentFlag.AlignBottom)
        self.btn_export = QtWidgets.QPushButton("Export")
        self.btn_export.setProperty("primary", True)
        self.btn_export.clicked.connect(self.export_selected)
        head.addWidget(self.btn_export, 0, QtCore.Qt.AlignmentFlag.AlignBottom)
        lay.addLayout(head)
        ctl = window.figures.export_ctrl
        self._ctl = ctl
        # One button per export (each exports at once, asking where): the
        # tiles' context menus and the search use them.
        self.buttons: "dict[str, QtWidgets.QPushButton]" = {}
        for label, method in EXPORTS:
            b = QtWidgets.QPushButton(label, self)
            b.clicked.connect(lambda _c=False, m=method: self._export(m))
            b.hide()
            self.buttons[method] = b

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(14)
        lay.addLayout(body, 1)
        left = QtWidgets.QVBoxLayout()
        self.lines = QtWidgets.QTableWidget(0, 4)
        self.lines.setHorizontalHeaderLabels(["Include", "Module", "State", "Result"])
        self.lines.verticalHeader().setVisible(False)
        self.lines.horizontalHeader().setStretchLastSection(True)
        self.lines.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.lines.setShowGrid(False)
        self.lines.itemChanged.connect(self._tick_changed)
        box = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(box)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.lines)
        bl.addStretch(1)
        left.addWidget(card("What goes out", box), 1)
        self.lbl_left_out = QtWidgets.QLabel("")
        self.lbl_left_out.setWordWrap(True)
        self.warn_left_out = QtWidgets.QFrame(objectName="Warning")
        wl = QtWidgets.QHBoxLayout(self.warn_left_out)
        wl.setContentsMargins(14, 8, 14, 8)
        wl.addWidget(self.lbl_left_out)
        self.warn_left_out.hide()
        left.addWidget(self.warn_left_out)
        body.addLayout(left, 1)

        right = QtWidgets.QVBoxLayout()
        right.setSpacing(14)
        tiles = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(tiles)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)
        self.tiles: "dict[str, QtWidgets.QPushButton]" = {}
        for i, (method, title, detail) in enumerate(FORMATS):
            t = QtWidgets.QPushButton(f"{title}\n{detail}")
            t.setProperty("tile", True)
            t.setCheckable(True)
            t.setChecked(method in DEFAULT_FORMATS)
            t.setMinimumHeight(70)
            t.setToolTip(f"Ticked: part of Export. Right click: export only the {title}.")
            t.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            t.customContextMenuRequested.connect(
                lambda _pos, m=method, tt=title: self._tile_menu(m, tt))
            t.toggled.connect(lambda _on: self._sync_export_button())
            grid.addWidget(t, i // 2, i % 2)
            self.tiles[method] = t
        right.addWidget(card("Formats", tiles))
        self.provenance = QtWidgets.QTableWidget(0, 2)
        self.provenance.setHorizontalHeaderLabels(["", ""])
        self.provenance.horizontalHeader().setVisible(False)
        self.provenance.verticalHeader().setVisible(False)
        self.provenance.horizontalHeader().setStretchLastSection(True)
        self.provenance.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.provenance.setShowGrid(False)
        copy = QtWidgets.QPushButton("Copy")
        copy.clicked.connect(self._copy_provenance)
        right.addWidget(card("Provenance", self.provenance, 0, copy), 1)
        body.addLayout(right, 1)

        # The text report, on demand.
        row = QtWidgets.QHBoxLayout()
        self.btn_text = QtWidgets.QPushButton("Show the text report")
        self.btn_text.setCheckable(True)
        self.btn_text.setProperty("flat", True)
        row.addWidget(self.btn_text)
        row.addStretch(1)
        self.lbl_folder = QtWidgets.QLabel("")
        self.lbl_folder.setProperty("muted", True)
        row.addWidget(self.lbl_folder)
        lay.addLayout(row)
        tbar = QtWidgets.QWidget()
        tb = QtWidgets.QHBoxLayout(tbar)
        tb.setContentsMargins(0, 0, 0, 0)
        for label, method in (("Copy text", "copy_report"), ("Save text…", "save_report_txt")):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(lambda _c=False, m=method: self._export(m))
            tb.addWidget(b)
        self.text = TextBox()
        self.text.setMinimumHeight(320)
        self.text_card = card("Text report", self.text, 0, tbar)
        self.text_card.hide()
        self.btn_text.toggled.connect(lambda on: (self.text_card.setVisible(on),
                                                  self.btn_text.setText(
                                                      "Hide the text report" if on
                                                      else "Show the text report")))
        lay.addWidget(self.text_card, 1)
        self._filling = False
        self._last_folder = ""
        self._sync_export_button()

    def _tile_menu(self, method: str, title: str) -> None:
        menu = QtWidgets.QMenu(self)
        act = menu.addAction(f"Export only the {title}…")
        act.setEnabled(self.buttons[method].isEnabled())
        act.triggered.connect(self.buttons[method].click)
        menu.exec(QtGui.QCursor.pos())

    def selected(self) -> "list[str]":
        return [m for m, t in self.tiles.items() if t.isChecked() and t.isEnabled()]

    def _sync_export_button(self) -> None:
        n = len(self.selected())
        self.btn_export.setText(f"Export {n} file{'s' if n != 1 else ''}" if n else "Export")
        self.btn_export.setEnabled(n > 0)

    def export_selected(self) -> None:
        """The ticked formats, in one folder asked once."""
        from ecg.app import dialogs
        methods = self.selected()
        if not methods:
            return
        folder = dialogs.filedialog.askdirectory(title="Export to which folder?",
                                                 initialdir=self._last_folder)
        if not folder:
            return
        self._last_folder = folder
        previous = dialogs.filedialog.backend
        dialogs.install(file_dialog=_InFolder(folder, previous))
        try:
            for m in methods:
                getattr(self._ctl, m)()
        finally:
            dialogs.install(file_dialog=previous)
        self.lbl_folder.setText(f"Last export: {folder}")
        self.refresh()

    def _copy_provenance(self) -> None:
        rows = [f"{self.provenance.item(r, 0).text()}\t{self.provenance.item(r, 1).text()}"
                for r in range(self.provenance.rowCount())
                if self.provenance.item(r, 0) and self.provenance.item(r, 1)]
        QtWidgets.QApplication.clipboard().setText("\n".join(rows))
        self.win.set_status("Provenance copied (paste into Excel).")

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
        for m, t in self.tiles.items():
            t.setEnabled(self.buttons[m].isEnabled())
        h = self.lines.horizontalHeader().height() + 4 + sum(
            self.lines.rowHeight(r) for r in range(self.lines.rowCount()))
        self.lines.setFixedHeight(max(h, 80))
        self.provenance.resizeColumnToContents(0)
        self._sync_export_button()
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
