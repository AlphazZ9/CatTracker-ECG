"""The Recording sheet, the recent recordings and Cohort Trends in Qt:
Subject ID / sex filled from the file name as in Tk (never over a typed
value), the project kept across files, notes in the registry, and all of
them saved in and restored from the session."""
import shutil
from pathlib import Path

import pytest

from ecg.core import analysis
from ecg.io import db
from ecg.io.session import load_session
from ecg.qt.main_window import MainWindow

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"


def _window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _named(tmp_path, name):
    p = tmp_path / name
    shutil.copy(FIXTURE, p)
    return str(p)


def test_subject_and_sex_come_from_the_file_name_unless_typed(qtbot, tmp_path):
    w = _window(qtbot)
    sheet = w.recording_sheet
    w.open_path(_named(tmp_path, "Stim BNST - 1146 F.mat"), check_session=False)
    assert (sheet.ed_subject.text(), sheet.ed_sex.text()) == ("1146", "F")
    assert w.subject() == "1146" and "1146" in w.lbl_file.text()
    # the next file's name replaces what the app filled
    w.open_path(_named(tmp_path, "M_206 baseline.mat"), check_session=False)
    assert (sheet.ed_subject.text(), sheet.ed_sex.text()) == ("206", "M")
    # a typed value is kept, and the status says so
    sheet.ed_subject.setText("mouse A")
    w.open_path(_named(tmp_path, "Stim BNST - 1146 F.mat"), check_session=False)
    assert sheet.ed_subject.text() == "mouse A"
    assert "Kept as typed" in w.statusBar().currentMessage()
    # the project spans recordings
    sheet.ed_project.setText("BNST")
    sheet.ed_project.editingFinished.emit()
    w.open_path(_named(tmp_path, "M_206 baseline.mat"), check_session=False)
    assert sheet.ed_project.text() == "BNST" and w.lbl_file.text().startswith("[BNST]")
    sheet.ed_sex.setText("female")
    sheet.ed_sex.editingFinished.emit()
    assert sheet.ed_sex.text() == "F"
    assert w.figures.session.recent_files[:2] == [str(tmp_path / "M_206 baseline.mat"),
                                                   str(tmp_path / "Stim BNST - 1146 F.mat")]


def test_notes_are_saved_in_the_registry(qtbot, tmp_path):
    w = _window(qtbot)
    path = _named(tmp_path, "rec.mat")
    db.upsert_recording(path, "fp", "", {"hr_mean": 500.0})
    w.open_path(path, check_session=False)
    w.recording_sheet.txt_notes.setPlainText("  dobutamine 1 mg/kg at 600 s ")
    w.recording_sheet.save_notes()
    assert db.get_notes(path) == "dobutamine 1 mg/kg at 600 s"
    w2 = _window(qtbot)
    w2.open_path(path, check_session=False)
    assert w2.recording_sheet.txt_notes.toPlainText() == "dobutamine 1 mg/kg at 600 s"
    assert w2.figures.session.recording_notes == "dobutamine 1 mg/kg at 600 s"


@pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")
def test_the_session_keeps_subject_sex_and_project(qtbot, tmp_path):
    path = _named(tmp_path, "Stim BNST - 1146 F.mat")
    w = _window(qtbot)
    w.open_path(path, check_session=False)
    w.recording_sheet.ed_subject.setText("M7")
    w.recording_sheet.ed_project.setText("P1")
    w.detection.detect()
    qtbot.waitUntil(lambda: w.recording.detection.rpeaks_ok is not None, timeout=60000)
    w.session.save()
    qtbot.waitUntil(lambda: w.session.saved_at is not None, timeout=60000)
    state = load_session(path)
    assert state["recording_fields"]["subject"] == "M7"
    assert state["recording_fields"]["sex"] == "F"
    assert state["recording_fields"]["project_name"] == "P1"
    w.jobs.cancel_all()
    w.jobs.wait(5000)

    w2 = _window(qtbot)
    w2.open_path(path, check_session=False)
    assert w2.recording_sheet.ed_subject.text() == "1146"        # the file name's
    w2.session.saved_at = None
    w2.session.restore(state, path)
    qtbot.waitUntil(lambda: w2.session.saved_at is not None, timeout=60000)
    sheet = w2.recording_sheet
    assert (sheet.ed_subject.text(), sheet.ed_sex.text(), sheet.ed_project.text()) == \
        ("M7", "F", "P1")
    assert "M7" in w2.lbl_file.text()
    w2.jobs.cancel_all()
    w2.jobs.wait(5000)


def test_recent_recordings_and_cohort_trends(qtbot, tmp_path, monkeypatch):
    from ecg.qt.dialogs import recent_dialog as rd
    a, b = _named(tmp_path, "a.mat"), _named(tmp_path, "b.mat")
    db.upsert_recording(a, "fa", "", {"hr_mean": 512.4, "sdnn": 4.25, "duration_s": 150.0},
                        notes="baseline", project_name="P1")
    db.upsert_recording(b, "fb", "", {"hr_mean": 480.0, "sdnn": 3.0, "rmssd": 2.0},
                        project_name="")
    db.upsert_recording(str(tmp_path / "gone.mat"), "fg", "", {"hr_mean": 1.0})
    w = _window(qtbot)
    w._fill_recent_menu()
    names = [act.text() for act in w.menu_recent.actions() if act.text()]
    assert {"a.mat", "b.mat"} <= set(names) and "gone.mat" not in names
    w.open_recent()
    dlg = w.recent_dialog
    files = [dlg.table.item(i, 0).text() for i in range(dlg.table.rowCount())]
    assert sorted(files) == ["a.mat", "b.mat"]
    i = files.index("a.mat")
    assert [dlg.table.item(i, c).text() for c in range(5)] == \
        ["a.mat", "512 bpm", "4.2 ms", "150 s", "baseline"]
    dlg.open_trends()
    fig = dlg.trends.canvas.figure
    assert len(fig.axes) == 3 and fig.axes[0].get_ylabel() == "Heart rate (bpm)"
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert set(labels) == {"P1", "Ungrouped"}
    dlg.open_row(i)
    assert w.recording.signal.filepath == a
