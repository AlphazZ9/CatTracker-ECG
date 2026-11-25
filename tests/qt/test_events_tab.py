"""Events in the Qt interface: the reference's classification (after the
intervals, as the reference: the AV-delay check reads their PR), the list
and the ECG around a selected event."""
from pathlib import Path

import pytest

from ecg.core import analysis
from ecg.qt.main_window import MainWindow
from tests.regression import golden

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")


def test_classification_list_and_event_view(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    w.open_path(str(FIXTURE))
    w.detection.cb_method.setCurrentIndex(0)
    w.detection.detect()
    qtbot.waitUntil(lambda: w.recording.detection.rpeaks_ok is not None, timeout=60000)
    w.analyze()
    qtbot.waitUntil(lambda: w.statusBar().currentMessage() == "Analysis done", timeout=120000)
    w.morphology.measure()
    qtbot.waitUntil(lambda: w.statusBar().currentMessage().startswith(
        "Interval delineation done"), timeout=120000)
    a = w.recording.analysis
    w.events.btn_classify.click()
    qtbot.waitUntil(lambda: a.premature_summary is not None, timeout=60000)

    ref, manifest = golden.load(ROOT / "tests" / "regression" / "reference", "synthetic_mouse_ecg")
    new = golden.flatten({"events": a.arrhythmia_events, "summary": a.premature_summary}, "events")
    diffs = golden.compare({k: ref[k] for k in new}, new,
                           manifest["tolerance"]["rtol"], manifest["tolerance"]["atol"])
    assert diffs == [], diffs[:5]
    assert a.arrhythmia_run["av_note"] == ref["events/av_note"]

    assert w.events.table.rowCount() == len(a.arrhythmia_events) > 0
    assert w.figures.slot("arr_ribbon").figure.axes
    w.events.table.selectRow(0)
    assert a.arr_selected_idx == 0
    assert w.figures.slot("arr_detail").figure.axes
    assert "#1" in w.figures.lbl_arr_event_title.text()
    # Next / Previous, Copy for Excel
    ev = w.events
    if len(a.arrhythmia_events) > 1:
        ev.step_event(+1)
        assert a.arr_selected_idx == 1
        ev.step_event(-1)
    assert a.arr_selected_idx == 0
    from PySide6 import QtWidgets
    QtWidgets.QApplication.clipboard().clear()
    ev.btn_copy.click()
    assert a.arrhythmia_tsv and QtWidgets.QApplication.clipboard().text() == a.arrhythmia_tsv
    # Edit peaks here: a left click on a peak of the event trace excludes it
    from types import SimpleNamespace
    d, fs = w.recording.detection, w.recording.signal.fs
    ev.btn_edit.setChecked(True)
    e0 = a.arrhythmia_events[0]
    inside = d.rpeaks_ok[(d.rpeaks_ok / fs >= e0.t_start) & (d.rpeaks_ok / fs <= e0.t_end)]
    peak = int(inside[0] if len(inside) else d.rpeaks_ok[10])
    n = len(d.rpeaks_ok)
    w.figures.analysis_ctrl.on_arr_detail_click(SimpleNamespace(xdata=peak / fs, button=1))
    assert peak in d.manual_excluded and len(d.rpeaks_ok) == n - 1
    assert w.detection.btn_undo.isEnabled()
    w.detection.undo()
    assert peak not in d.manual_excluded and len(d.rpeaks_ok) == n
    w.jobs.wait(5000)
