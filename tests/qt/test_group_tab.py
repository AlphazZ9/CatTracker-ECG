"""Group in Qt: Add open recording writes the Tk table (same function),
the table reads back, and each analysis is that of ecg.io.group_compare
with the tab's choices."""
from pathlib import Path

import pandas as pd
import pytest

from ecg.core import analysis
from ecg.io import group_compare as gc
from ecg.io import group_data as gd
from ecg.qt.main_window import MainWindow

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")


def test_add_open_recording_and_analyses(qtbot, tmp_path, monkeypatch):
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    assert w.group.tbl_res.horizontalHeaderItem(0).text() == "Open a group table"
    w.open_path(str(FIXTURE), check_session=False)
    w.detection.cb_method.setCurrentIndex(0)
    w.detection.detect()
    qtbot.waitUntil(lambda: w.recording.detection.rpeaks_ok is not None, timeout=60000)
    w.analyze()
    qtbot.waitUntil(lambda: w.statusBar().currentMessage() == "Analysis done", timeout=180000)
    w.compare.run()
    qtbot.waitUntil(lambda: bool(w.compare.segs), timeout=60000)

    table = tmp_path / "group.xlsx"
    from PySide6 import QtWidgets

    def _no_box(*a, **k):
        raise AssertionError(f"message box: {a[1:]}")
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", staticmethod(_no_box))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(table), "")))
    w.group.add_open()
    assert table.exists() and w.group.sheets is not None
    sheets = gd.read_sheets(str(table))
    assert len(sheets[gd.SHEET_ANIMALS]) == 1
    for name in (gd.SHEET_SEGMENTS, gd.SHEET_WHOLE):
        assert len(sheets[name])

    g = w.group
    g.cb_by.setCurrentText(gc.BY_NONE)
    g.cb_an.setCurrentText(gc.AN_WHOLE)
    ref = gc.analyse_whole(sheets, gd.animals_frame(sheets), gc.BY_NONE, None)
    pd.testing.assert_frame_equal(g.res.display, ref.display)
    g.cb_an.setCurrentText(gc.AN_SEGMENTS)
    ref = gc.analyse_segments(sheets, gd.animals_frame(sheets), gc.BY_NONE,
                              g.cb_a.currentText(), g.cb_b.currentText(), None,
                              pairs=gc.PAIRS_ONE, leave_out=True)
    pd.testing.assert_frame_equal(g.res.display, ref.display)
    # an animal's group is written to the file
    g.set_animal((sheets[gd.SHEET_ANIMALS]["subject"].iloc[0],
                  sheets[gd.SHEET_ANIMALS]["file"].iloc[0]), "group", "Control")
    assert gd.read_sheets(str(table))[gd.SHEET_ANIMALS]["group"].iloc[0] == "Control"
    w.jobs.cancel_all()
    w.jobs.wait(5000)
