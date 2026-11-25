"""Time course > Autonomic > Calibrate ks/kp in Qt: the Tk dialog's pure
helpers on the registry rows; Apply writes the knees into the ks / kp boxes
(nothing recomputed); Export writes the configuration's rows."""
import pandas as pd
import pytest
from PySide6 import QtWidgets

from ecg.qt.dialogs import calibration_dialog as cd
from ecg.qt.main_window import MainWindow
from ecg.ui import event_locked_view as ev
from tests.ui.test_event_locked_view import _cohort, _row


@pytest.fixture
def rows(monkeypatch):
    data = _cohort() + [_row("M0", 5.0, 0.1, 0.1, method="exact")]

    def _list(label=None, project_name=None, valid_only=True):
        return [r for r in data if (label is None or r["label"] == label)
                and (project_name is None or r["project_name"] == project_name)]
    monkeypatch.setattr(cd, "list_event_rows", _list)
    monkeypatch.setattr(cd, "list_event_labels", lambda: ["dobu"])
    written = []
    monkeypatch.setattr(cd, "export_event_rows_csv",
                        lambda path, rs: written.append((path, list(rs))) or len(rs))
    return data, written


def test_calibration_dialog_is_the_cohort_calibration(qtbot, rows, monkeypatch):
    data, written = rows
    w = MainWindow()
    qtbot.addWidget(w)
    tc = w.time_course
    tc.sp_ks.setValue(3.0)
    tc.sp_kp.setValue(12.0)
    tc.open_calibration()
    dlg = tc.calibration
    assert dlg.cb_label.currentText() == "dobu"                 # the label with most events
    assert dlg.cb_config.count() == 2
    cfg = dlg._configs[dlg.cb_config.currentText()]
    ref = ev.cohort_calibration(ev.registry_frame(data), 3.0, 12.0,
                                aggregation=ev.DEFAULT_AGGREGATION, config=cfg)
    pd.testing.assert_frame_equal(dlg.cal.tests, ref.tests)
    assert dlg.cal.knees == ref.knees
    assert dlg.txt.toPlainText() == ev.calibration_text(ref)
    # per event
    dlg.agg_buttons["event"].setChecked(True)
    assert dlg.cal.aggregation == "event" and dlg.cal.counts["n_units"] == 16
    # Apply: the knees into the Autonomic boxes, not recomputed
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes))
    ks, kp, _t, enabled = ev.apply_plan(dlg.cal)
    assert enabled
    dlg.apply()
    assert tc.sp_ks.value() == pytest.approx(ks, abs=0.01)
    assert tc.sp_kp.value() == pytest.approx(kp, abs=0.01)
    assert "not applied yet" in w.figures.lbl_auto_status.text()
    assert w.recording.analysis.autonomic_result is None
    # Copy and export
    dlg.copy()
    assert "knee_csi" in QtWidgets.QApplication.clipboard().text()
    dlg.export("out.csv")
    assert written and all(ev.row_config(r) == cfg for r in written[0][1])
    assert len(written[0][1]) == 16
    # The other configuration
    dlg.cb_config.setCurrentIndex(1)
    assert dlg.cal.counts["n_events"] == 1


def test_empty_registry_says_how_to_fill_it(qtbot, monkeypatch):
    monkeypatch.setattr(cd, "list_event_rows", lambda **k: [])
    monkeypatch.setattr(cd, "list_event_labels", lambda: [])
    w = MainWindow()
    qtbot.addWidget(w)
    w.time_course.open_calibration()
    dlg = w.time_course.calibration
    assert "No valid event-locked row" in dlg.txt.toPlainText()
    assert dlg.cal.counts["n_events"] == 0
