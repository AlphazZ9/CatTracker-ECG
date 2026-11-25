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


def test_check_agreement_counts_and_circles_the_unconfirmed_beats(qtbot, window, monkeypatch):
    from ecg.core import ml_detector
    from ecg.qt.dialogs import agreement_dialog
    from ecg.services.agreement_service import method_agreement
    monkeypatch.setattr(ml_detector.MLPeakModel, "load",
                        staticmethod(lambda: (_ for _ in ()).throw(ml_detector.NoModelError())))
    _detect(qtbot, window)
    s, d = window.recording.signal, window.recording.detection
    det = window.detection
    assert det.btn_agreement.isEnabled()
    det.check_agreement()
    qtbot.waitUntil(lambda: d.agreement_pct is not None, timeout=60000)
    from ecg.core.models import MouseECG
    host = window.figures
    min_rr = host._safe_float(host.ent_minrr, MouseECG.MIN_RR_MS)
    expected = method_agreement(s.filtered, s.fs, d.rpeaks_ok, min_rr,
                                det.cb_method.currentText().lower())
    assert expected["ok"] and "SG+Derivative (already the active method)" in expected["methods_skipped"]
    assert np.array_equal(d.agreement_disagree_samples, expected["disagree_samples"])
    dlg = det.agreement_dialog
    assert isinstance(dlg, agreement_dialog.AgreementDialog)
    assert [b.value() for b in dlg.bars] == expected["histogram"][::-1]
    x, _y = det.plot.unconfirmed.getData()
    assert np.allclose(x, expected["disagree_samples"] / s.fs)
    if dlg.times:
        dlg.jump_to(dlg.times[0])
        t0, t1 = det.plot.window()
        assert t0 <= dlg.times[0] <= t1
    # A new detection drops the check.
    det.detect()
    qtbot.waitUntil(lambda: d.agreement_pct is None, timeout=60000)
    assert len(det.plot.unconfirmed.getData()[0]) == 0


def test_rr_hr_quality_strip(qtbot, window):
    from ecg.core.analysis import compute_beat_correlation
    _detect(qtbot, window)
    s, d = window.recording.signal, window.recording.detection
    det = window.detection
    assert det.btn_strip.isChecked() and det.plot.strip[0].isVisible() is not False
    qtbot.waitUntil(lambda: d._beat_corr_for is d.rpeaks_ok, timeout=10000)
    x, y = det.plot.rr_curve.getData()
    assert np.allclose(y, np.diff(d.rpeaks_ok) / s.fs * 1000.0)
    assert np.allclose(x, d.rpeaks_ok[1:] / s.fs)
    lo, hi = window.figures.analysis_ctrl.hrv_rr_bounds(quiet=True)
    ref = compute_beat_correlation(s.filtered, d.rpeaks_ok, s.fs, lo, hi)
    _qx, qy = det.plot.q_curve.getData()
    assert np.allclose(qy, ref["beat_corr"], equal_nan=True)
    det.btn_strip.setChecked(False)
    assert not window.figures.ui.rrhr_strip_visible
    assert all(not p.isVisible() for p in det.plot.strip)
