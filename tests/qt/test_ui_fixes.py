"""Interface fixes of the UI audit (lot 1): the dark theme on the figures,
figures drawn when shown, the prompts that follow the state, and a new
detection that asks before replacing the peaks edited by hand."""
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import pytest
from PySide6 import QtGui, QtWidgets

from ecg.qt.main_window import MainWindow
from ecg.qt.tabs import events_tab, morphology_tab

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    yield w
    w.jobs.cancel_all()
    w.jobs.wait(5000)


def _detect(qtbot, w):
    w.open_path(str(FIXTURE))
    w.detection.detect()
    qtbot.waitUntil(lambda: w.recording.detection.rpeaks_ok is not None, timeout=60000)


def test_a_figure_of_a_hidden_tab_is_drawn_when_read(window):
    calls = []
    slot = window.figures.slot("rr")                   # Rhythm & HRV: not the tab shown
    assert not slot.isVisible()
    slot.update(lambda fig: calls.append(fig.add_subplot()))
    assert calls == []                                 # waits for its tab
    assert len(slot.figure.axes) == 1 and len(calls) == 1   # read: drawn once
    assert len(slot.fig.axes) == 1 and len(calls) == 1


def test_a_figure_is_drawn_when_its_tab_is_shown(window, qtbot):
    calls = []
    slot = window.figures.slot("asymmetry")            # on Rhythm & HRV
    slot.update(lambda fig: calls.append(fig.add_subplot()))
    window.tabs.setCurrentWidget(window.rhythm)
    qtbot.waitUntil(lambda: len(calls) == 1, timeout=5000)


def test_the_dark_theme_reaches_the_figures(window, qtbot):
    window.tabs.setCurrentWidget(window.rhythm)
    shown, hidden = window.figures.slot("asymmetry"), window.figures.slot("beat")
    for slot in (shown, hidden):
        slot.update(lambda fig: fig.add_subplot())
    window.toggle_theme()
    dark = mcolors.to_rgba(matplotlib.rcParams["figure.facecolor"])
    assert dark != mcolors.to_rgba("white")
    for slot in (shown, hidden):
        assert mcolors.to_rgba(slot.figure.get_facecolor()) == dark


def test_the_prompts_follow_the_state(window, qtbot):
    host = window.figures
    assert host.lbl_arrhythmia_status.text() == events_tab.NEEDS_PEAKS
    _detect(qtbot, window)
    assert host.lbl_arrhythmia_status.text() == events_tab.READY
    assert host.lbl_ivl_status.text() == morphology_tab.NEEDS_ANALYZE
    window.recording.analysis.results = {"hr": {}}      # as after Analyze
    window.morphology.results_changed()
    assert host.lbl_ivl_status.text() == ""
    window.open_path(str(FIXTURE))                      # another recording: back to the start
    assert host.lbl_arrhythmia_status.text() == events_tab.NEEDS_PEAKS
    assert host.lbl_ivl_status.text() == morphology_tab.NEEDS_ANALYZE


def test_detecting_again_asks_before_dropping_hand_edits(window, qtbot, monkeypatch):
    _detect(qtbot, window)
    d = window.recording.detection
    peak = int(d.rpeaks_ok[10])
    d.manual_excluded.add(peak)                         # a peak excluded by hand
    asked = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda box: asked.append(box.text()))
    before = d.rpeaks_ok
    window.detection.detect()                           # the box is closed: nothing happens
    assert asked and "1 excluded" in asked[0]
    assert not window._active                           # no detection started
    assert d.rpeaks_ok is before and peak in d.manual_excluded


def test_detect_peaks_is_ctrl_d_not_space(window):
    keys = {s.key().toString() for s in window.findChildren(QtGui.QShortcut)}
    assert "Ctrl+D" in keys and "Space" not in keys


def test_rhythm_charts_and_report_buttons(window, qtbot):
    """The Rhythm charts follow Analyze; the Report has the Tk export buttons."""
    _detect(qtbot, window)
    window.analyze()
    qtbot.waitUntil(lambda: window.statusBar().currentMessage() == "Analysis done",
                    timeout=120000)
    charts = window.rhythm.charts
    x, _y = charts.t_line.getData()
    assert x is not None and len(x) > 100
    rep = window.report
    window.tabs.setCurrentWidget(rep)
    qtbot.wait(100)
    assert rep.buttons["export_excel"].isVisible() and rep.buttons["export_excel"].isEnabled()
    assert rep.buttons["export_prism"].text() == "GraphPad Prism"


def test_command_search_finds_pages_and_runs_them(window):
    window.open_palette()
    window._palette.edit.setText("go morphology")
    assert window._palette.list.count() == 1
    window._palette._run_current()
    assert window.tabs.currentWidget() is window.morphology


def test_signal_quality_sheet_and_silent_host_attributes(window, qtbot, caplog):
    """The Signal quality values of the Tk panel are filled after Analyze;
    an attribute the shared code reads and Qt lacks is logged once."""
    import logging
    _detect(qtbot, window)
    window.analyze()
    qtbot.waitUntil(lambda: window.statusBar().currentMessage() == "Analysis done",
                    timeout=120000)
    vals = window.figures._quality_vals
    assert vals["sq_corr"].text() not in ("", "—")
    assert "/" in vals["sq_badbeats"].text()
    assert window.detection.btn_quality.text().startswith("Quality ")
    assert window.figures.slot("quality_hist").figure.axes
    with caplog.at_level(logging.INFO, logger="ecg"):
        window.figures.some_tk_only_widget
        window.figures.some_tk_only_widget
    hits = [r for r in caplog.records if "some_tk_only_widget" in r.getMessage()]
    assert len(hits) == 1


def test_the_header_shows_the_version(window):
    import ecg
    labels = [w.text() for w in window.findChildren(QtWidgets.QLabel)]
    assert ecg.__version__ in labels
