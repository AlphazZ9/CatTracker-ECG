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
    # Tk's wheel zoom on the event trace: 0.8 times the window per notch,
    # one redraw after the burst, the window box following.
    from matplotlib.backend_bases import MouseEvent
    ev.btn_edit.setChecked(False)
    slot = w.figures.slot("arr_detail")
    assert slot.canvas.wheel_always
    ax = slot.figure.axes[0]
    x0, x1 = ax.get_xlim()
    px, py = ax.transData.transform(((x0 + x1) / 2, 0.0))
    for _ in range(2):
        slot.canvas.callbacks.process("scroll_event", MouseEvent(
            "scroll_event", slot.canvas, px, py, button="up", step=1))
    qtbot.wait(400)
    y0, y1 = slot.figure.axes[0].get_xlim()
    assert abs((y1 - y0) - (x1 - x0) * 0.64) < 0.02 * (x1 - x0)
    assert abs(ev.sp_win.value() - a.arr_win) < 0.01
    w.jobs.wait(5000)
