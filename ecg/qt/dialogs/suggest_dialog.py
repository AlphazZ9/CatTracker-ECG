# -*- coding: utf-8 -*-
"""
ecg.qt.dialogs.suggest_dialog
-----------------------------
Annotations > Suggest from channels (the Tk AnnotationSuggestDialog): the
file's marker / keyboard channels and the stimulus pulses of its other
channels, read in a background job (ecg.services.
annotation_suggest_service.collect); the user ticks the ones to add and
can rename them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from PySide6 import QtCore, QtWidgets

from ecg.qt import theme
from ecg.services import annotation_suggest_service as svc

if TYPE_CHECKING:
    from ecg.qt.main_window import MainWindow


class SuggestDialog(QtWidgets.QDialog):
    """The suggestions, each with a tick and an editable label; Add selected
    calls *on_add(list of annotations)*."""

    def __init__(self, parent: QtWidgets.QWidget, window: "MainWindow",
                 on_add: "Callable[[list[dict]], None]") -> None:
        super().__init__(parent)
        self.setWindowTitle("Suggest annotations from channels")
        self.resize(720, 460)
        self._on_add = on_add
        self.suggestions: "list[dict]" = []
        lay = QtWidgets.QVBoxLayout(self)
        self.lbl = QtWidgets.QLabel("Reading the marker, keyboard and stimulus channels…")
        self.lbl.setWordWrap(True)
        self.lbl.setProperty("muted", True)
        lay.addWidget(self.lbl)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Start (s)", "End (s)", "Label"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table, 1)
        bar = QtWidgets.QHBoxLayout()
        for text, on in (("Tick all", True), ("Untick all", False)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _c=False, v=on: self.tick(v))
            bar.addWidget(b)
        hint = QtWidgets.QLabel("A lone marker lasts 1 s; a train spans its markers or pulses.")
        hint.setProperty("muted", True)
        bar.addWidget(hint, 1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.reject)
        bar.addWidget(close)
        self.btn_add = QtWidgets.QPushButton("Add selected")
        self.btn_add.setProperty("primary", True)
        self.btn_add.setEnabled(False)
        self.btn_add.clicked.connect(self.add)
        bar.addWidget(self.btn_add)
        lay.addLayout(bar)

        sig = window.recording.signal
        t = sig.time
        t_max = float(t[-1]) if t is not None and len(t) else None
        args = (sig.filepath, sig.channel, sig.crop, t_max,
                list(window.recording.analysis.annotations))
        window.run_job("Suggest annotations", lambda progress_cb=None: svc.collect(*args),
                       on_done=self.show_result, on_end=self._ended)
        self._got = False

    def _ended(self) -> None:
        if not self._got and self.lbl.text().startswith("Reading"):
            self.show_result("Could not read the channels (see the status bar).")

    def show_result(self, result: "tuple[list[dict], list[str]] | str") -> None:
        self._got = True
        self.table.setRowCount(0)
        self.suggestions = []
        if isinstance(result, str):
            self.lbl.setText(result)
            self.lbl.setStyleSheet(f"color: {theme.current().ORANGE};")
            return
        found, read = result
        if not found:
            self.lbl.setText("No marker, keyboard or stimulus-pulse event that is not annotated "
                             "already." if read else
                             "This file has no marker or keyboard channel and no channel of "
                             "stimulus pulses (Spike2 .smr or its MATLAB export).")
            return
        self.lbl.setProperty("muted", False)
        self.lbl.setText(f"{len(found)} suggestion{'s' if len(found) != 1 else ''} from "
                         f"{', '.join(read)}: untick the ones to leave out, edit a label "
                         "if needed.")
        self.table.setRowCount(len(found))
        flags = QtCore.Qt.ItemFlag
        for i, s in enumerate(found):
            for c, text in enumerate((f"{s['t_start']:.3f}", f"{s['t_end']:.3f}")):
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(flags.ItemIsEnabled | flags.ItemIsSelectable
                              | (flags.ItemIsUserCheckable if c == 0 else flags.NoItemFlags))
                if c == 0:
                    item.setCheckState(QtCore.Qt.CheckState.Checked)
                self.table.setItem(i, c, item)
            self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(s["label"]))
        self.table.resizeColumnsToContents()
        self.suggestions = list(found)
        self.btn_add.setEnabled(True)

    def tick(self, on: bool) -> None:
        state = QtCore.Qt.CheckState.Checked if on else QtCore.Qt.CheckState.Unchecked
        for i in range(self.table.rowCount()):
            self.table.item(i, 0).setCheckState(state)

    def picked(self) -> "list[tuple[dict, str]]":
        return [(s, self.table.item(i, 2).text()) for i, s in enumerate(self.suggestions)
                if self.table.item(i, 0).checkState() == QtCore.Qt.CheckState.Checked]

    def add(self) -> None:
        out = svc.to_annotations(self.picked())
        if out:
            self._on_add(out)
        self.accept()


def open_suggestions(manager: QtWidgets.QWidget, window: "MainWindow",
                     on_add: "Callable[[list[dict]], None]") -> "Optional[SuggestDialog]":
    if not window.recording.signal.filepath:
        QtWidgets.QMessageBox.information(manager, "Suggest annotations",
                                          "Open a recording first.")
        return None
    dlg = SuggestDialog(manager, window, on_add)
    dlg.show()
    return dlg
