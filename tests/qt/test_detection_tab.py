"""The Qt window: the eight tabs in the Tk order, and Detect Peaks / the
threshold giving exactly the app's peaks (the reference outputs)."""
from pathlib import Path

import numpy as np
import pytest

from ecg.core import analysis
from ecg.core.detection import apply_threshold
from ecg.qt.main_window import MainWindow
from ecg.ui.tab_keys import TAB_ORDER
from tests.regression import golden

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w.jobs.cancel_all()
    w.jobs.wait(5000)


def _detect(qtbot, w):
    w.open_path(str(FIXTURE))
    w.detection.detect()
    qtbot.waitUntil(lambda: w.recording.detection.rpeaks_ok is not None, timeout=60000)


def test_tabs_are_those_of_the_tk_interface(window):
    shown = [window.tabs.tabText(i).replace("&&", "&") for i in range(window.tabs.count())]
    assert shown == list(TAB_ORDER)


def test_detect_peaks_gives_the_reference_peaks(qtbot, window):
    _detect(qtbot, window)
    ref, _ = golden.load(ROOT / "tests" / "regression" / "reference", "synthetic_mouse_ecg")
    d = window.recording.detection
    # The Qt Method box starts on SG + Derivative, no filtering: the
    # reference's "sg_derivative" scenario.
    assert np.array_equal(d.rpeaks_ok, ref["detect/sg_derivative/rpeaks_ok"])
    assert np.array_equal(d.all_candidates, ref["detect/sg_derivative/all_cands"])
    assert not window.recording.signal.raw.flags.writeable


def test_threshold_change_rebuilds_the_peaks_as_the_app_does(qtbot, window):
    _detect(qtbot, window)
    s, d = window.recording.signal, window.recording.detection
    window.detection.sp_thr.setValue(0.9)
    qtbot.waitUntil(lambda: abs(d.thresh_amp - apply_threshold(
        s.filtered, d.all_candidates, d.all_prominences, 0.9, fs=s.fs)[2]) < 1e-12, timeout=3000)
    expected, rejected, _ = apply_threshold(s.filtered, d.all_candidates, d.all_prominences,
                                            0.9, fs=s.fs)
    assert np.array_equal(d.rpeaks_ok, expected) and np.array_equal(d.rpeaks_rej, rejected)
    assert window.detection.sl_thr.value() == 90


def test_a_result_for_another_recording_is_discarded(qtbot, window):
    window.open_path(str(FIXTURE))
    window.detection.detect()
    window.open_path(str(FIXTURE))                 # a new recording while it runs
    window.jobs.wait(60000)
    qtbot.wait(100)
    assert window.recording.detection.rpeaks_ok is None
