"""The Qt ML detector dialog: Add open recording writes the training
example (SessionController.save_for_training_worker), it is listed, and
Remove deletes it -- in a temporary training folder."""
from pathlib import Path

import pytest
from PySide6 import QtWidgets

from ecg.core import analysis
from ecg.qt.dialogs.ml_training import MLTrainingDialog
from ecg.qt.main_window import MainWindow

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")


def test_add_and_remove_training_example(qtbot, tmp_path, monkeypatch):
    import ecg.core.ml_detector as ml
    import ecg.io.db as db
    monkeypatch.setattr(ml, "TRAINING_DATA_DIR", tmp_path / "ml_training")
    monkeypatch.setattr(ml, "MODEL_PATH", tmp_path / "model.skops")
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "registry.db")
    monkeypatch.setattr(db, "_schema_ready", False)
    monkeypatch.setattr(db, "_schema_retry_at", 0.0)
    w = MainWindow()
    qtbot.addWidget(w)
    w.open_path(str(FIXTURE), check_session=False)
    w.detection.detect()
    qtbot.waitUntil(lambda: w.recording.detection.rpeaks_ok is not None, timeout=60000)
    dlg = MLTrainingDialog(w)
    qtbot.addWidget(dlg)
    assert dlg.table.rowCount() == 0 and not dlg.btn_train.isEnabled()
    dlg.add_open()
    qtbot.waitUntil(lambda: dlg.table.rowCount() == 1, timeout=120000)
    assert dlg.table.item(0, 0).text() == FIXTURE.name
    assert dlg.btn_remove_open.isEnabled()
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes))
    dlg.remove_open()
    assert dlg.table.rowCount() == 0
    w.jobs.cancel_all()
    w.jobs.wait(5000)
