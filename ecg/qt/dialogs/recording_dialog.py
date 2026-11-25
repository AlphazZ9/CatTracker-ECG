# -*- coding: utf-8 -*-
"""
ecg.qt.dialogs.recording_dialog
-------------------------------
The Recording sheet (the Tk one opened from the file chip): the animal's
Subject ID and sex, the project the recording belongs to, its notes, and
the file's channel list.

The fields are published on the window's FigureHost under the Tk widget
names (ent_subject, ent_sex, ent_project_name), so the exports, the
sessions, the cohort registry and the group table read them as in Tk. The
Subject ID and the sex are filled from the file name on opening
(ecg.services.recording_fields), never over a value typed here. The
project spans many recordings: opening another file keeps it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6 import QtWidgets

from ecg.io.subject_name import DEFAULT_SUBJECT, normalise_sex
from ecg.qt.figure_host import Field

if TYPE_CHECKING:
    from ecg.qt.main_window import MainWindow

log = logging.getLogger("ecg")


class RecordingDialog(QtWidgets.QDialog):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.win = window
        self.setWindowTitle("Recording")
        host = window.figures
        form = QtWidgets.QFormLayout(self)
        self.ed_subject = QtWidgets.QLineEdit(DEFAULT_SUBJECT)
        form.addRow("Subject ID", self.ed_subject)
        self.ed_sex = QtWidgets.QLineEdit()
        self.ed_sex.setMaximumWidth(60)
        self.ed_sex.setToolTip("Sex of the animal: M (male), F (female) or empty. Read from the "
                               "file name (a standalone M or F next to the number: '1146 F', "
                               "'M_206') unless typed here; written to the group table.")
        form.addRow("Sex (M / F)", self.ed_sex)
        self.ed_project = QtWidgets.QLineEdit()
        self.ed_project.setToolTip("A label for the whole study (many recordings): kept when "
                                   "another file is opened; groups the cohort registry")
        form.addRow("Project", self.ed_project)
        form.addRow(QtWidgets.QLabel("<b>Notes</b> (this recording: drug, dose, time…)"))
        self.txt_notes = QtWidgets.QPlainTextEdit()
        self.txt_notes.setMinimumHeight(120)
        form.addRow(self.txt_notes)
        row = QtWidgets.QHBoxLayout()
        self.btn_save_notes = QtWidgets.QPushButton("Save notes")
        self.btn_save_notes.clicked.connect(self.save_notes)
        row.addWidget(self.btn_save_notes)
        self.btn_channels = QtWidgets.QPushButton("Channels of the file…")
        self.btn_channels.clicked.connect(self.show_channels)
        row.addWidget(self.btn_channels)
        row.addStretch(1)
        form.addRow(row)
        self.lbl_session = QtWidgets.QLabel("")
        self.lbl_session.setProperty("muted", True)
        self.lbl_session.setWordWrap(True)
        form.addRow(self.lbl_session)
        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.hide)
        form.addRow(close)

        host.ent_subject = Field(self.ed_subject)
        host.ent_sex = Field(self.ed_sex)
        host.ent_project_name = Field(self.ed_project)
        host._subject_autofill = None
        host._sex_autofill = None
        host._on_project_name_change = window.update_file_label
        self.ed_subject.editingFinished.connect(window.update_file_label)
        self.ed_project.editingFinished.connect(window.update_file_label)
        self.ed_sex.editingFinished.connect(self._normalise_sex)

    def _normalise_sex(self) -> None:
        """'f', 'female', 'femelle'... become 'F' (and the same for M)."""
        text = self.ed_sex.text()
        norm = normalise_sex(text)
        if norm != text:
            self.ed_sex.setText(norm)

    # ── notes ───────────────────────────────────────────────
    def file_opened(self, path: str) -> None:
        """The recording's notes (the registry's, as SignalController.load_path)."""
        from ecg.io.db import _DB_AVAILABLE, get_notes
        notes = get_notes(path) if _DB_AVAILABLE else ""
        self.win.figures.session.recording_notes = notes
        self.txt_notes.setPlainText(notes)

    def notes_restored(self) -> None:
        self.txt_notes.setPlainText(self.win.figures.session.recording_notes or "")

    def save_notes(self) -> None:
        from ecg.io.db import _DB_AVAILABLE, set_notes
        path = self.win.recording.signal.filepath
        if not path:
            QtWidgets.QMessageBox.information(self, "Notes", "Load a file first.")
            return
        notes = self.txt_notes.toPlainText().strip()
        self.win.figures.session.recording_notes = notes
        if _DB_AVAILABLE:
            set_notes(path, notes)
        self.win.session_dirty = True
        self.win.set_status("Notes saved ✓")

    def show_channels(self) -> None:
        from ecg.io.loaders import list_channels
        path = self.win.recording.signal.filepath
        if not path:
            QtWidgets.QMessageBox.warning(self, "No file", "Open a .mat file first.")
            return
        try:
            QtWidgets.QMessageBox.information(self, f"Channels — {Path(path).name}",
                                              list_channels(path))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", str(exc))
