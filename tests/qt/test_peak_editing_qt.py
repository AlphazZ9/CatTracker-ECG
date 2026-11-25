"""Editing peaks by hand in the Qt interface gives what the Tk interface
gives: the same click sequence (left / right, free placement, zoom levels,
undo / redo) leaves the same peaks, edit sets and undo history in both."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ecg.core import analysis
from ecg.qt.main_window import MainWindow
from tests.regression import golden

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")


@pytest.fixture
def qt_window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1200, 700)
    w.show()
    w.open_path(str(FIXTURE))
    w.detection.cb_method.setCurrentIndex(0)          # Auto: the Tk stub's FilterParams default
    w.detection.detect()
    qtbot.waitUntil(lambda: w.recording.detection.rpeaks_ok is not None, timeout=60000)
    yield w
    w.jobs.cancel_all()
    w.jobs.wait(5000)


def _tk_app(qt_rec):
    """The Tk DetectionController on the same detection (golden's stub app)."""
    app = golden._new_app(FIXTURE)
    app.signal = copy.deepcopy(qt_rec.signal)
    app.detection = copy.deepcopy(qt_rec.detection)
    app.detection.edit_mode = True
    app._set_status = lambda t, c=None: None
    app.sl_thr = golden._Entry(0.5)
    app._EDIT_UNDO_LIMIT = 50
    return app


def test_the_same_clicks_give_the_same_peaks_as_tk(qtbot, qt_window):
    tab, rec = qt_window.detection, qt_window.recording
    tab.btn_edit.setChecked(True)
    tk = _tk_app(rec)
    rng = np.random.default_rng(5)
    peaks_s = rec.detection.rpeaks_ok / rec.signal.fs
    for i in range(120):
        win = float(rng.choice([1.0, 2.0, 10.0]))
        t = float(rng.choice(peaks_s) + rng.normal(0, 0.01)) if rng.random() < 0.6 \
            else float(rng.uniform(5, 140))
        button = str(rng.choice(["left", "left", "right", "right", "other"]))
        free = bool(rng.random() < 0.15)
        op = rng.random()
        tab.chk_free.setChecked(free)
        tk.detection.edit_free_placement = free
        tk.ent_window = golden._Entry(win)
        if op < 0.1:
            tab.undo()
            tk.detection_ctrl.undo_edit()
        elif op < 0.15:
            tab.redo()
            tk.detection_ctrl.redo_edit()
        else:
            tab.plot.set_window(t - win / 2, t + win / 2)
            t0, t1 = tab.plot.window()
            tk.ent_window = golden._Entry(t1 - t0)     # the width the Qt view really shows
            tab._clicked(t, button)
            tk.detection_ctrl.on_detail_click(SimpleNamespace(
                xdata=t, button={"left": 1, "right": 3, "other": 2}[button]))
        q, k = rec.detection, tk.detection
        assert q.manual_excluded == k.manual_excluded and q.manual_added == k.manual_added, i
        assert np.array_equal(q.rpeaks_ok, k.rpeaks_ok), i
        assert q.edit_undo == k.edit_undo and q.edit_redo == k.edit_redo, i
    assert rec.detection.manual_added or rec.detection.manual_excluded


def test_clicks_do_nothing_outside_edit_mode(qt_window):
    tab, d = qt_window.detection, qt_window.recording.detection
    before = d.rpeaks_ok.copy()
    tab._clicked(float(before[100]) / qt_window.recording.signal.fs, "left")
    assert np.array_equal(d.rpeaks_ok, before) and not d.manual_excluded


def test_an_edit_drops_the_analysis_results_and_undo_restores_the_peaks(qt_window):
    tab, rec = qt_window.detection, qt_window.recording
    d = rec.detection
    rec.analysis.results = {"hr": {}}
    before = d.rpeaks_ok.copy()
    tab.btn_edit.setChecked(True)
    tab.plot.set_window(40, 42)
    tab._clicked(float(before[300]) / rec.signal.fs, "left")
    assert rec.analysis.results is None and int(before[300]) not in d.rpeaks_ok
    assert tab.btn_undo.isEnabled()
    tab.undo()
    assert np.array_equal(d.rpeaks_ok, before) and tab.btn_redo.isEnabled()
