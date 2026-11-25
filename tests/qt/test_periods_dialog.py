"""Annotations and noisy regions: the Tk dialogs' checks and stored dicts,
and noisy regions reaching Analyze (their RR intervals left out)."""
from pathlib import Path

import pytest

from ecg.core import analysis
from ecg.qt.dialogs.periods_dialog import PeriodDialog, PeriodsManager, validate_period
from ecg.qt.main_window import MainWindow

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w.jobs.cancel_all()
    w.jobs.wait(5000)


def test_validation_is_that_of_the_tk_dialogs():
    assert validate_period("1.23456", "2", 10) == ((1.2346, 2.0), "")
    assert validate_period("a", "2", 10)[0] is None
    assert validate_period("3", "2", 10) == (None, "Start must be less than end.")
    assert validate_period("-1", "2", 10)[0] is None
    assert validate_period("1", "11.5", 10)[0] is None
    assert validate_period("1", "10.9", 10)[0] == (1.0, 10.9)


def test_period_dialog_builds_the_tk_dicts(qtbot, window):
    d = PeriodDialog(window, "annotation", 100.0, default=(1.0, 3.0))
    qtbot.addWidget(d)
    assert (d.ed_start.text(), d.ed_end.text()) == ("1.000", "3.000")
    d.ed_text.setText(" drug ")
    d._ok()
    assert d.result_period == {"t_start": 1.0, "t_end": 3.0, "label": "drug",
                               "color": d.cb_color.itemData(0)}
    n = PeriodDialog(window, "noise", 100.0, existing={"t_start": 5, "t_end": 6, "note": "x"})
    qtbot.addWidget(n)
    n.ed_end.setText("7.5")
    n._ok()
    assert n.result_period == {"t_start": 5.0, "t_end": 7.5, "note": "x"}


@pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")
def test_noisy_region_is_left_out_of_analyze(qtbot, window, monkeypatch):
    window.open_path(str(FIXTURE))
    window.detection.detect()
    qtbot.waitUntil(lambda: window.recording.detection.rpeaks_ok is not None, timeout=60000)
    mgr = PeriodsManager(window, "noise", window.detection.noise_changed)
    qtbot.addWidget(mgr)
    region = {"t_start": 20.0, "t_end": 40.0, "note": ""}

    def fake_exec(self):
        self.result_period = dict(region)
        return True
    monkeypatch.setattr(PeriodDialog, "exec", fake_exec)
    mgr.add()
    assert window.recording.analysis.noise_regions == [region]
    assert window.session_dirty
    assert window.detection.btn_noise.text() == "Noisy regions (1)"
    assert window.analysis_params()["noise_regions"] == [region]


@pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")
def test_mark_noise_drag_and_right_click(qtbot, window):
    """DetectionController's Mark Noise: a drag adds a region (rounded to
    1 ms, sorted, shorter than 50 ms ignored); a right click inside removes it."""
    window.open_path(str(FIXTURE), check_session=False)
    window.detection.detect()
    qtbot.waitUntil(lambda: window.recording.detection.rpeaks_ok is not None, timeout=60000)
    det = window.detection
    det.btn_mark_noise.setChecked(True)
    assert det.btn_edit.isChecked() and det.plot.span_mode
    det._noise_drawn(10.0, 12.34567)
    det._noise_drawn(5.0, 5.02)               # shorter than 50 ms: a click, not a region
    det._noise_drawn(2.0, 3.0)
    assert window.recording.analysis.noise_regions == [
        {"t_start": 2.0, "t_end": 3.0, "note": ""},
        {"t_start": 10.0, "t_end": 12.346, "note": ""}]
    peaks_before = window.recording.detection.rpeaks_ok.copy()
    det._clicked(2.5, "right")
    assert len(window.recording.analysis.noise_regions) == 1
    det._clicked(20.0, "left")                # no peak edit while Mark Noise is on
    assert (window.recording.detection.rpeaks_ok == peaks_before).all()


def test_suggest_from_channels_adds_the_ticked_suggestions(qtbot, window, monkeypatch):
    from ecg.io.export import ANNOTATION_COLORS
    from ecg.services import annotation_suggest_service as svc
    found = [{"t_start": 10.0, "t_end": 11.0, "label": "Keyboard '1'", "source": "Keyboard"},
             {"t_start": 60.0, "t_end": 61.0, "label": "Stim (20 pulses)", "source": "Stim"},
             {"t_start": 70.0, "t_end": 71.0, "label": "Stim (5 pulses)", "source": "Stim"}]
    seen = []
    monkeypatch.setattr(svc, "collect", lambda *a: seen.append(a) or (found, ["Keyboard", "Stim"]))
    window.open_path(str(FIXTURE), check_session=False)
    window.recording.analysis.annotations[:] = [
        {"t_start": 30.0, "t_end": 31.0, "label": "drug", "color": "#E53935"}]
    m = PeriodsManager(window, "annotation", window.detection.annotations_changed)
    qtbot.addWidget(m)
    m.suggest()
    dlg = m.suggest_dialog
    qtbot.waitUntil(lambda: dlg.btn_add.isEnabled(), timeout=10000)
    assert seen[0][0] == str(FIXTURE) and seen[0][4][0]["label"] == "drug"
    from PySide6 import QtCore
    dlg.table.item(0, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    dlg.table.item(1, 2).setText("Stim 20 Hz")
    dlg.add()
    anns = window.recording.analysis.annotations
    assert [(a["t_start"], a["label"]) for a in anns] == \
        [(30.0, "drug"), (60.0, "Stim 20 Hz"), (70.0, "Stim (5 pulses)")]
    # one colour per source channel
    assert anns[1]["color"] == anns[2]["color"] == ANNOTATION_COLORS[0][0]
    assert m.table.rowCount() == 3 and window.session_dirty
