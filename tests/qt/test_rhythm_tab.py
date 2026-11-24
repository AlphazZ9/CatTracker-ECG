"""Analyze in the Qt interface: the same numbers as the app's reference
outputs (core, frequency and non-linear HRV), and the Rhythm & HRV figures
and tables drawn by the shared figure code."""
from pathlib import Path

import pytest

from ecg.core import analysis
from ecg.qt.main_window import MainWindow
from tests.regression import golden

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")


@pytest.fixture
def analysed(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    w.open_path(str(FIXTURE))
    w.detection.cb_method.setCurrentIndex(0)      # Auto: the reference's analysed scenario
    w.detection.detect()
    qtbot.waitUntil(lambda: w.recording.detection.rpeaks_ok is not None, timeout=60000)
    w.analyze()

    def _done() -> bool:          # analyse_core leaves empty freq / non-linear frames
        r = w.recording.analysis.results or {}
        return all(r.get(k) is not None and not r[k].empty for k in ("hrv_freq", "hrv_nonlin"))
    qtbot.waitUntil(_done, timeout=120000)
    qtbot.waitUntil(lambda: w.statusBar().currentMessage() == "Analysis done", timeout=30000)
    yield w
    w.jobs.cancel_all()
    w.jobs.wait(5000)


def test_analyze_gives_the_reference_results(analysed):
    ref, manifest = golden.load(ROOT / "tests" / "regression" / "reference", "synthetic_mouse_ecg")
    r = analysed.recording.analysis.results
    # The reference's analysis/core/results also holds the frequency, non-linear
    # and interval frames added after Analyze: compare those parts separately.
    later = ("/hrv_freq/", "/hrv_nonlin/", "/intervals/")
    core = {k: v for k, v in golden.flatten(r, "analysis/core/results").items()
            if not any(x in k for x in later)}
    new = dict(core)
    new.update(golden.flatten(r["hrv_freq"], "analysis/frequency"))
    new.update(golden.flatten(r["hrv_nonlin"], "analysis/nonlinear"))
    tol = manifest["tolerance"]
    diffs = golden.compare({k: ref[k] for k in new}, new, tol["rtol"], tol["atol"])
    if diffs and golden.package_changes(manifest):
        pytest.xfail("library versions differ from the reference")
    assert diffs == []


def test_rhythm_figures_and_tables_are_drawn(analysed):
    host = analysed.figures
    for name in ("rr", "poincare", "psd", "rr_hist"):
        assert host.slot(name).figure.axes, name
    assert "RMSSD" in host.txt_td.toPlainText()
    assert host.txt_fd.toPlainText() and host.txt_nl.toPlainText()
    assert host.txt_td.tsv                       # "copy as a table" available
    from ecg.core.metric_table import rhythm_rows
    tab = analysed.rhythm
    n_rows = len(rhythm_rows(tab._table))
    assert tab.table.rowCount() > n_rows        # the rows + their group headers
    assert host._rhythm_cards and host.slot("asymmetry").figure.axes


def test_a_peak_edit_clears_the_results(qtbot, analysed):
    tab, rec = analysed.detection, analysed.recording
    tab.btn_edit.setChecked(True)
    tab.plot.set_window(40, 42)
    tab._clicked(float(rec.detection.rpeaks_ok[300]) / rec.signal.fs, "left")
    assert rec.analysis.results is None
    assert not analysed.figures.slot("rr").figure.axes
    assert analysed.btn_analyze.isEnabled()
