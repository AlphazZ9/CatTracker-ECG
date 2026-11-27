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
    slot = window.figures.slot("intervals")            # on Morphology & Intervals
    slot.update(lambda fig: calls.append(fig.add_subplot()))
    window.tabs.setCurrentWidget(window.morphology)
    qtbot.waitUntil(lambda: len(calls) == 1, timeout=5000)


def test_the_dark_theme_reaches_the_figures(window, qtbot):
    window.tabs.setCurrentWidget(window.morphology)
    shown, hidden = window.figures.slot("intervals"), window.figures.slot("beat")
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


def test_the_quality_and_distribution_charts_are_drawn_on_screen(window, qtbot):
    """Quality panel, RR asymmetry and beat distributions: PyQtGraph on
    screen, their Matplotlib figures kept hidden for the exports."""
    _detect(qtbot, window)
    window.analyze()
    qtbot.waitUntil(lambda: window.statusBar().currentMessage() == "Analysis done",
                    timeout=60000)
    det, rh, mo = window.detection, window.rhythm, window.morphology
    assert det.quality_charts._series is not None
    assert len(det.quality_charts.t_points.data) > 100
    assert rh.asym_charts._data is not None
    assert mo.dist_charts._data is not None
    for name in ("quality_time", "quality_hist", "asymmetry", "beat_dist"):
        assert not window.figures.slot(name).isVisible()
        assert window.figures.slot(name).figure.axes        # drawn when read (exports)
    window.figures.slot("quality_time").clear()             # results dropped: emptied too
    assert det.quality_charts._series is None
    window.figures.slot("asymmetry").clear()
    assert rh.asym_charts._data is None


def test_no_controller_call_falls_into_the_void(window, qtbot, caplog):
    """Every name the shared controllers read on a whole run (detect,
    Analyze, every tab, Classify events) exists in Qt: none falls back to
    FigureHost's silent no-op (audit G4)."""
    import logging
    caplog.set_level(logging.INFO, logger="ecg")
    _detect(qtbot, window)
    window.analyze()
    qtbot.waitUntil(lambda: window.statusBar().currentMessage() == "Analysis done",
                    timeout=120000)
    for i in range(window.tabs.count()):
        window.tabs.setCurrentIndex(i)
        qtbot.wait(100)
    window.events.btn_classify.click()
    qtbot.waitUntil(lambda: window.recording.analysis.arrhythmia_run is not None,
                    timeout=60000)
    assert window.figures.lbl_arr_baseline.text().startswith("Baseline")
    void = [r.getMessage() for r in caplog.records if "is not provided in Qt" in r.getMessage()]
    assert void == []


def test_the_signal_is_on_screen_as_soon_as_the_file_is_opened(window, qtbot):
    """Opening a file shows its raw signal (Tk's load_raw_only) before any
    Detect Peaks; Detect then shows the filtered one."""
    window.open_path(str(FIXTURE), check_session=False)
    qtbot.waitUntil(lambda: window.recording.signal.raw is not None, timeout=30000)
    det = window.detection
    x, _y = det.plot.trace.getOriginalDataset()
    assert x is not None and len(x) == len(window.recording.signal.raw)
    assert det.seg_trace.buttons[1].isChecked()              # "Raw" is what is shown
    assert window.recording.detection.rpeaks_ok is None
    det.detect()
    qtbot.waitUntil(lambda: window.recording.detection.rpeaks_ok is not None, timeout=60000)
    assert det.seg_trace.buttons[0].isChecked()


def test_the_trace_draws_what_is_in_view_only(window, qtbot):
    """Clip-to-view and downsampling hold on the plots (PlotItem.addItem used
    to switch them off): a 2 s view draws a few thousand points, not the
    whole recording (an hour at 2 kHz froze every move for ~1 s)."""
    _detect(qtbot, window)
    plot = window.detection.plot
    plot.set_window(10.0, 12.0)
    qtbot.wait(100)
    x, _y = plot.trace.getData()
    assert len(x) < 0.05 * len(window.recording.signal.filtered)
    assert plot.trace.opts["clipToView"] and plot.trace.opts["autoDownsample"]
    assert plot.rr_curve.opts["clipToView"]


def test_no_menu_bar_and_no_context_strip(window):
    """More room for the charts: the commands are in the header (Open, ▾,
    Save, Export, ⋯) and Ctrl+K; the analysis window is on the Signal page."""
    assert not window.menuBar().isVisible()
    assert not hasattr(window, "strip")
    more = [a.text() for a in window.btn_more.menu().actions() if a.text()]
    for name in ("Settings…", "ML detector…", "Light / dark mode", "About"):
        assert name in more
    assert window.detection.ed_win0.isVisibleTo(window.detection)


def test_figures_scroll_the_page_and_keep_tk_toolbars(window):
    """A plain wheel over a Matplotlib figure goes to the page (it used to
    stop there); Autonomic, Event-locked and Compare have Tk's toolbar."""
    from PySide6 import QtCore, QtGui
    for name in ("autonomic", "event_locked", "compare"):
        assert window.figures.slot(name).toolbar is not None, name
    canvas = window.figures.slot("epochs").canvas
    pos = QtCore.QPointF(10, 10)
    ev = QtGui.QWheelEvent(pos, pos, QtCore.QPoint(0, 0), QtCore.QPoint(0, 120),
                           QtCore.Qt.MouseButton.NoButton,
                           QtCore.Qt.KeyboardModifier.NoModifier,
                           QtCore.Qt.ScrollPhase.NoScrollPhase, False)
    ev.accept()
    canvas.wheelEvent(ev)
    assert not ev.isAccepted()


def test_signal_page_navigation_buttons(window, qtbot):
    """⏮ ◀ ▶ ⏭ (Tk's): start, one view back / on, end."""
    _detect(qtbot, window)
    det = window.detection
    det.plot.set_window(20.0, 22.0)
    det.page(+1)
    assert abs(det.plot.window()[0] - 22.0) < 1e-6
    det.page(-1)
    assert abs(det.plot.window()[0] - 20.0) < 1e-6
    det.go_to(det._end())
    assert abs(det.plot.window()[1] - det._end()) < 1e-6
