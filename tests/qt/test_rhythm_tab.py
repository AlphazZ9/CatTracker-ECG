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


def test_analyze_result_dropped_when_peaks_change_meanwhile(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.open_path(str(FIXTURE), check_session=False)
    w.detection.detect()
    qtbot.waitUntil(lambda: w.recording.detection.rpeaks_ok is not None, timeout=60000)
    w.analyze()
    w.detection.apply_threshold()                  # the threshold applied while it runs
    qtbot.waitUntil(lambda: not w._active, timeout=120000)
    assert w.recording.analysis.results is None
    assert "peaks changed" in w.statusBar().currentMessage()
    w.jobs.cancel_all()
    w.jobs.wait(5000)


def test_recompute_entropy_cap_and_spike_markers(qtbot, analysed):
    import numpy as np
    import pandas as pd
    from ecg.core.analysis import analyse_hrv_nonlinear
    w = analysed
    r = w.recording.analysis.results
    before = r["hrv_freq"].copy()
    tab = w.rhythm
    # Recompute Frequency alone: the same numbers, non-linear untouched
    nl = r["hrv_nonlin"]
    tab.btn_freq.click()
    qtbot.waitUntil(lambda: w.statusBar().currentMessage() == "Frequency HRV recomputed",
                    timeout=60000)
    pd.testing.assert_frame_equal(r["hrv_freq"], before)
    assert r["hrv_nonlin"] is nl
    # the entropy cap reaches the non-linear computation
    tab.sp_entropy.setValue(300)
    ctl = w.figures.analysis_ctrl
    assert ctl.entropy_cap() == 300
    tab.btn_nonlin.click()
    qtbot.waitUntil(lambda: w.statusBar().currentMessage() == "Non-linear HRV recomputed",
                    timeout=120000)
    lo, hi = ctl.results_rr_bounds()
    ref = analyse_hrv_nonlinear(w.figures._windowed_peaks(), w.recording.signal.fs,
                                max_beats=300, rr_min_ms=lo, rr_max_ms=hi,
                                breaks=ctl.hrv_breaks())
    pd.testing.assert_frame_equal(r["hrv_nonlin"], ref)
    # spike markers: the layer toggles, Review steps through the flagged beats
    spikes = w.figures.ui.rr_spike_times
    tab.btn_spikes.setChecked(not tab.btn_spikes.isChecked())
    assert w.figures.ui.show_spike_markers == tab.btn_spikes.isChecked()
    if spikes is not None and len(spikes):
        tab.btn_review.click()
        t0, t1 = w.detection.plot.window()
        assert any(t0 <= s <= t1 for s in np.asarray(spikes))
        assert w.tabs.currentWidget() is w.detection


def test_clear_edits_and_go_to(qtbot, analysed):
    w = analysed
    det, d = w.detection, w.recording.detection
    peaks = d.rpeaks_ok.copy()
    det.btn_edit.setChecked(True)
    det._clicked(float(peaks[50] / w.recording.signal.fs), "left")
    assert len(d.manual_excluded) == 1 and len(d.rpeaks_ok) == len(peaks) - 1
    det.clear_edits()
    assert not d.manual_excluded and (d.rpeaks_ok == peaks).all()
    assert w.recording.analysis.results is None
    det.undo()                                         # Clear Edits is undoable
    assert len(d.manual_excluded) == 1
    det.go_to(100.0)
    t0, _t1 = det.plot.window()
    assert abs(t0 - 100.0) < 1e-6
    det.go_to(1e6)                                     # kept inside the recording
    _t0, t1 = det.plot.window()
    assert t1 <= float(w.recording.signal.time[-1]) + 1e-6


def test_tab_states_follow_the_freshness(qtbot, analysed):
    from ecg.app.tab_keys import TAB_HRV, TAB_ORDER
    w = analysed
    w.refresh_tab_states()
    i = TAB_ORDER.index(TAB_HRV)
    assert w.tabs.tabToolTip(i).startswith(f"{TAB_HRV}: up to date")
    det, d = w.detection, w.recording.detection
    det.btn_edit.setChecked(True)
    det._clicked(float(d.rpeaks_ok[30] / w.recording.signal.fs), "left")
    w.refresh_tab_states()
    assert not w.tabs.tabToolTip(i).startswith(f"{TAB_HRV}: up to date")


def test_the_tachograms_zoom_and_come_back(qtbot):
    """Ctrl + wheel zooms the time of both tachograms (one axis), Reset zoom
    brings every chart back to its drawn view; a plain wheel is left to the
    page (it scrolls)."""
    import numpy as np
    from PySide6 import QtCore, QtGui
    from ecg.qt.widgets.rhythm_charts import RhythmCharts
    charts = RhythmCharts()
    for c in (charts.tacho, charts.tacho_rr):
        qtbot.addWidget(c)
        c.resize(600, 200)
        c.show()
    rng = np.random.default_rng(0)
    rr = 115 + rng.normal(0, 3, 600)
    charts.set_results({"rr_ms": rr, "rr_t_s": np.cumsum(rr) / 1000.0,
                        "rr_valid": np.ones(len(rr), bool)})
    home = charts.tacho.getPlotItem().getViewBox().viewRange()[0]

    def wheel(chart, ctrl: bool) -> None:
        pos = QtCore.QPointF(300, 100)
        mods = (QtCore.Qt.KeyboardModifier.ControlModifier if ctrl
                else QtCore.Qt.KeyboardModifier.NoModifier)
        ev = QtGui.QWheelEvent(pos, chart.mapToGlobal(pos.toPoint()).toPointF(),
                               QtCore.QPoint(0, 0), QtCore.QPoint(0, 120),
                               QtCore.Qt.MouseButton.NoButton, mods,
                               QtCore.Qt.ScrollPhase.NoScrollPhase, False)
        chart.wheelEvent(ev)

    wheel(charts.tacho, ctrl=False)
    assert charts.tacho.getPlotItem().getViewBox().viewRange()[0] == home
    wheel(charts.tacho, ctrl=True)
    zoomed = charts.tacho.getPlotItem().getViewBox().viewRange()[0]
    assert zoomed[1] - zoomed[0] < home[1] - home[0]
    # the RR lane follows (one time axis)
    assert np.allclose(charts.tacho_rr.getPlotItem().getViewBox().viewRange()[0], zoomed)
    charts.reset_zoom()
    assert np.allclose(charts.tacho.getPlotItem().getViewBox().viewRange()[0], home)
