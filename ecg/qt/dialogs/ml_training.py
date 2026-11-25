# -*- coding: utf-8 -*-
"""
ecg.qt.dialogs.ml_training
--------------------------
The ML R-peak detector dialog (ecg/ui/dialogs.py MLTrainingDialog) in Qt:
how to train it, the training examples (each removable), Add / Remove the
open recording, and Train model with its held-out check. Operates on the
pooled training cache and the saved model (ecg.core.ml_detector), with
the Tk dialog's texts and gate (ecg.ui.ml_examples).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PySide6 import QtWidgets

from ecg.core.ml_detector import MLPeakModel, delete_training_sample, train_model
from ecg.io.db import _DB_AVAILABLE, set_verified
from ecg.io.session import _file_fingerprint
from ecg.qt.figure_host import StatusLabel
from ecg.ui.ml_examples import status_text, train_gate, train_result_text

if TYPE_CHECKING:
    from ecg.qt.main_window import MainWindow


def training_examples() -> "list[dict]":
    """MLTrainingDialog._examples: each example with its file name and path."""
    from ecg.ui.dialogs import MLTrainingDialog
    return MLTrainingDialog._examples(None)   # type: ignore[arg-type]


class MLTrainingDialog(QtWidgets.QDialog):
    def __init__(self, window: "MainWindow") -> None:
        from ecg.ui.dialogs import MLTrainingDialog as _Tk
        super().__init__(window)
        self.win = window
        self.setWindowTitle("ML R-Peak Detector")
        self.resize(760, 640)
        lay = QtWidgets.QVBoxLayout(self)
        how = QtWidgets.QLabel("<b>How to train it</b><br>" + "<br>".join(_Tk.HOW_TO))
        how.setWordWrap(True)
        lay.addWidget(how)
        self.lbl_status = StatusLabel("")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)
        self.lbl_model = QtWidgets.QLabel("")
        self.lbl_model.setWordWrap(True)
        self.lbl_model.setProperty("muted", True)
        lay.addWidget(self.lbl_model)
        row = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add open recording to training examples")
        self.btn_add.clicked.connect(self.add_open)
        self.btn_remove_open = QtWidgets.QPushButton("Remove open recording")
        self.btn_remove_open.clicked.connect(self.remove_open)
        row.addWidget(self.btn_add)
        row.addWidget(self.btn_remove_open)
        row.addStretch(1)
        lay.addLayout(row)
        self.lbl_list = QtWidgets.QLabel("Training examples")
        lay.addWidget(self.lbl_list)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Recording", "Added", "Peaks", "Rate", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(False)
        lay.addWidget(self.table, 1)
        self.lbl_gate = StatusLabel("")
        lay.addWidget(self.lbl_gate)
        bar = QtWidgets.QHBoxLayout()
        self.btn_train = QtWidgets.QPushButton("Train model")
        self.btn_train.setProperty("primary", True)
        self.btn_train.clicked.connect(self.train)
        bar.addWidget(self.btn_train)
        self.lbl_result = StatusLabel("")
        self.lbl_result.setWordWrap(True)
        bar.addWidget(self.lbl_result, 1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.accept)
        bar.addWidget(close)
        lay.addLayout(bar)
        self.refresh()

    # ── state ───────────────────────────────────────────────
    def refresh(self) -> None:
        examples = training_examples()
        model = MLPeakModel.load()
        meta = model.meta if model is not None else None
        _short, full, _tone = status_text(examples, meta)
        self.lbl_status.configure(text=full)
        if meta is not None:
            acc = meta.get("holdout_accuracy")
            acc_txt = (f" — {acc * 100:.0f} % of candidate peaks classified correctly on "
                       "held-out data" if isinstance(acc, (int, float)) else "")
            check = ""
            if meta.get("holdout_confusion_matrix"):
                check = (f"\nFound {meta.get('holdout_recall', 0.0) * 100:.0f} % of your "
                         f"R-peaks; {meta.get('holdout_precision', 0.0) * 100:.0f} % of the "
                         f"peaks it marked are yours (checked on "
                         f"{meta.get('holdout_n_test', 0):,} candidate peaks kept aside).")
            self.lbl_model.setText(
                f"Current model: trained {str(meta.get('trained_at', '?'))[:16].replace('T', ' ')}"
                f" on {meta.get('n_training_files', '?')} training examples{acc_txt}.{check}")
        else:
            self.lbl_model.setText("")
        ok, gate = train_gate(len(examples))
        self.btn_train.setEnabled(ok)
        self.lbl_gate.configure(text=gate)
        self.lbl_list.setText(f"Training examples ({len(examples)})")
        self.table.setRowCount(0)
        for i, e in enumerate(examples):
            self.table.insertRow(i)
            rate = e.get("rate_bpm")
            for c, text in enumerate((e["name"], str(e.get("added_at", ""))[:10],
                                      f"{e.get('n_peaks', 0):,}",
                                      f"{rate:.0f} bpm" if rate else "—")):
                self.table.setItem(i, c, QtWidgets.QTableWidgetItem(text))
            b = QtWidgets.QPushButton("Remove")
            b.clicked.connect(lambda _c=False, x=e: self.remove(x["fingerprint"], x["filepath"],
                                                                 x["name"]))
            self.table.setCellWidget(i, 4, b)
        self.table.resizeColumnsToContents()
        rec = self.win.recording
        has_peaks = rec.signal.filtered is not None and rec.detection.rpeaks_ok is not None \
            and len(rec.detection.rpeaks_ok) > 0
        self.btn_add.setEnabled(has_peaks)
        path = rec.signal.filepath
        self.btn_remove_open.setEnabled(bool(path) and self.win.session.ctrl.is_training_example(
            path))

    def _busy(self) -> bool:
        from ecg.ui.ml_examples import BUSY_MESSAGE
        if self.win.jobs.running:
            self.lbl_result.configure(text=f"{BUSY_MESSAGE}: an operation of the main window "
                                           "is still running.")
            return True
        return False

    # ── examples ────────────────────────────────────────────
    def remove(self, fingerprint: str, filepath: "Optional[str]", name: str = "",
               confirm: bool = True) -> bool:
        if self._busy():
            return False
        if confirm and QtWidgets.QMessageBox.question(
                self, "Remove training example",
                f"Remove {name or 'this recording'} from the training examples?\n\n"
                "The model changes only when you train it again."
        ) != QtWidgets.QMessageBox.StandardButton.Yes:
            return False
        delete_training_sample(fingerprint)
        if filepath and _DB_AVAILABLE:
            set_verified(filepath, False)
        open_path = self.win.recording.signal.filepath
        if open_path and (open_path == filepath or _file_fingerprint(open_path) == fingerprint):
            self.win.figures.session.verified_for_training = False
        self.refresh()
        return True

    def remove_open(self) -> None:
        path = self.win.recording.signal.filepath
        if path:
            self.remove(_file_fingerprint(path), path, path.rsplit("/", 1)[-1])

    def add_open(self) -> None:
        """SessionController.add_training_example, in a background job."""
        if self._busy():
            return
        rec = self.win.recording
        s, d = rec.signal, rec.detection
        if s.filepath is None or s.filtered is None or d.rpeaks_ok is None:
            return
        ctrl = self.win.session.ctrl
        updating = ctrl.is_training_example(s.filepath)
        gen = self.win.generation
        self.btn_add.setEnabled(False)

        def done(result: dict) -> None:
            if gen == self.win.generation:
                self.win.figures.session.verified_for_training = True
            what = f"{result.get('n_peaks')} peaks"
            self.lbl_result.configure(
                text=f"{'Training example updated' if updating else 'Added to the training examples'}"
                     f" — {what}. Retrain the model (Train model) to use it.")
            self.refresh()
        self.win.run_job("Training example", ctrl.save_for_training_worker, s.filepath,
                         s.filtered, s.fs, d.rpeaks_ok.copy(), progress=False, on_done=done,
                         on_end=self.refresh)

    # ── train ───────────────────────────────────────────────
    def train(self) -> None:
        if self._busy():
            return
        self.btn_train.setEnabled(False)
        self.btn_train.setText("Training…")

        def done(result: dict) -> None:
            self.lbl_result.configure(text=train_result_text(result))

        def end() -> None:
            self.btn_train.setText("Train model")
            self.refresh()
        self.win.run_job("Train model", train_model, progress=False, on_done=done, on_end=end)
