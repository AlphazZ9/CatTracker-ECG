"""Time course in the Qt interface: Epochs, Sliding and Autonomic give the
reference outputs (default settings and every metric), drawn by the shared
figure code."""
from pathlib import Path

import pytest

from ecg.core import analysis
from ecg.qt.main_window import MainWindow
from tests.regression import golden

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")


@pytest.fixture(scope="module")
def ref():
    return golden.load(ROOT / "tests" / "regression" / "reference", "synthetic_mouse_ecg")


@pytest.fixture
def analysed(qtbot):
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
    yield w
    w.jobs.cancel_all()
    w.jobs.wait(5000)


def _check(frame, prefix, ref_pair):
    ref, manifest = ref_pair
    new = golden.flatten(frame, prefix)
    # compute_autonomic records where it ran (analysis window, peak span) in
    # the result; the reference computes the indices directly, without them.
    extra = [k for k in new if k not in ref]
    assert all(("analysis_window" in k or "peak_span_s" in k) for k in extra), extra
    new = {k: v for k, v in new.items() if k in ref}
    tol = manifest["tolerance"]
    diffs = golden.compare({k: ref[k] for k in new}, new, tol["rtol"], tol["atol"])
    if diffs and golden.package_changes(manifest):
        pytest.xfail("library versions differ from the reference")
    assert diffs == [], diffs[:5]


def test_epochs_default_and_every_metric(qtbot, analysed, ref):
    tc, a = analysed.time_course, analysed.recording.analysis
    tc.btn_epochs.click()
    qtbot.waitUntil(lambda: a.epoch_df is not None, timeout=60000)
    _check(a.epoch_df, "epochs/default", ref)
    host = analysed.figures
    assert host.slot("epochs").figure.axes and host.txt_epochs.toPlainText()
    from PySide6.QtWidgets import QCheckBox
    for b in tc.widget(0).findChildren(QCheckBox):
        b.setChecked(True)
    a.epoch_df = None
    tc.btn_epochs.click()
    qtbot.waitUntil(lambda: a.epoch_df is not None, timeout=60000)
    _check(a.epoch_df, "epochs/all_metrics", ref)


def test_sliding_default(qtbot, analysed, ref):
    tc, a = analysed.time_course, analysed.recording.analysis
    tc.btn_sliding.click()
    qtbot.waitUntil(lambda: a.rolling_hrv_df is not None, timeout=60000)
    _check(a.rolling_hrv_df, "sliding/default", ref)
    assert analysed.figures.slot("rolling_hrv").figure.axes


def test_autonomic_default_with_spwvd(qtbot, analysed, ref):
    tc, a = analysed.time_course, analysed.recording.analysis
    tc.chk_lfhf.setChecked(True)
    tc.btn_auto.click()
    qtbot.waitUntil(lambda: a.autonomic_result is not None, timeout=120000)
    from ecg.core.autonomic import to_dataframe
    _check(to_dataframe(a.autonomic_result), "autonomic/frame", ref)
    _check(a.autonomic_spwvd, "autonomic/spwvd", ref)
    qtbot.waitUntil(lambda: bool(analysed.figures.slot("autonomic").figure.axes), timeout=10000)
