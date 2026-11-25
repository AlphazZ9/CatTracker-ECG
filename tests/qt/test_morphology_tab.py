"""Measure intervals in the Qt interface: the reference's per-beat intervals
(rolling median of 32 beats, built-in wave template), and the figures."""
from pathlib import Path

import pytest

from ecg.core import analysis
from ecg.qt.main_window import MainWindow
from tests.regression import golden

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")


def test_measure_intervals_gives_the_reference_intervals(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    w.open_path(str(FIXTURE))
    w.detection.cb_method.setCurrentIndex(0)          # Auto: the reference's scenario
    w.detection.detect()
    qtbot.waitUntil(lambda: w.recording.detection.rpeaks_ok is not None, timeout=60000)
    w.analyze()
    qtbot.waitUntil(lambda: w.statusBar().currentMessage() == "Analysis done", timeout=120000)
    assert w.morphology.btn_measure.isEnabled() and w.morphology.sp_ensemble.value() == 32
    w.morphology.measure()
    qtbot.waitUntil(lambda: w.statusBar().currentMessage().startswith(
        "Interval delineation done"), timeout=120000)
    ivl = w.recording.analysis.results["intervals"]
    ref, manifest = golden.load(ROOT / "tests" / "regression" / "reference", "synthetic_mouse_ecg")
    new = golden.flatten(ivl, "intervals/n32")
    tol = manifest["tolerance"]
    diffs = golden.compare({k: ref[k] for k in new}, new, tol["rtol"], tol["atol"])
    if diffs and golden.package_changes(manifest):
        pytest.xfail("library versions differ from the reference")
    assert diffs == []
    for name in ("beat", "intervals"):
        assert w.figures.slot(name).figure.axes, name
    w.jobs.wait(5000)
