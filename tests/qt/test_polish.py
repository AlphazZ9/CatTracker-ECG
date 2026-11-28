"""The look of the window: labels cut with "…" (whole in text() and the
tooltip), the "?" help, empty figures offering their command, the quality
charts framed on the beats, landmark names that never overlap, a warning
shown once, the quality score coloured by its grade."""
import numpy as np
from PySide6 import QtWidgets

from ecg.qt.widgets.beat_chart import label_rows
from ecg.qt.widgets.mpl_canvas import MplCanvas
from ecg.qt.widgets.quality_charts import QualityCharts, scale_start
from ecg.qt.widgets.shell import ElidedLabel, help_badge


def test_elided_label_keeps_its_whole_text(qtbot):
    text = "synthetic_mouse_ecg.mat · ECG · 2000 Hz · subject_01"
    lbl = ElidedLabel(text)
    qtbot.addWidget(lbl)
    lbl.resize(90, 20)
    lbl.show()
    lbl.grab()                                         # paints it
    assert lbl.text() == text and text in lbl.toolTip()
    assert lbl.minimumSizeHint().width() <= 80
    lbl.resize(600, 20)
    lbl.grab()
    assert lbl.toolTip() == ""                         # whole again: no tooltip


def test_help_badge_holds_its_text():
    badge = help_badge("Click: that beat")
    assert badge.text() == "?" and badge.toolTip() == "Click: that beat"


def test_empty_figure_offers_its_command_while_it_can_run(qtbot):
    canvas = MplCanvas()
    qtbot.addWidget(canvas)
    src = QtWidgets.QPushButton("Compute")
    qtbot.addWidget(src)
    ran = []
    canvas.set_placeholder("Click Compute: the indices appear here.")
    canvas.set_placeholder_action("Compute", lambda: ran.append(1), src)
    assert canvas._stack.currentWidget() is canvas._empty
    assert not canvas._empty_btn.isHidden() and canvas._empty_btn.isEnabled()
    src.setEnabled(False)
    assert not canvas._empty_btn.isEnabled()
    src.setEnabled(True)
    canvas._empty_btn.click()
    assert ran == [1]
    canvas.update_figure(lambda fig: fig.add_subplot().plot([0, 1]))
    assert canvas._stack.currentWidget() is not canvas._empty


def test_quality_scales_start_under_the_bulk_of_the_beats(qtbot):
    rng = np.random.default_rng(0)
    v = np.r_[0.97 + 0.01 * rng.standard_normal(1000), [0.2, 0.3, 0.5]]
    lo = scale_start(v)
    assert 0.5 <= lo <= 0.85 and lo < np.percentile(v, 1)
    assert scale_start(np.full(10, 0.2)) == 0.5          # never under the floor
    charts = QualityCharts()
    t = np.arange(len(v), dtype=float)
    charts.set_series({"t": t, "v": v, "roll_t": t, "roll": v, "all": v})
    shown = len(charts.t_points.data) + len(charts.t_below.data)
    assert shown == len(v) and len(charts.t_below.data) == int((v < lo).sum())
    heights = list(charts.h_bars.opts["height"]) + list(charts.h_below.opts["height"])
    assert sum(heights) == len(v)                        # every beat counted once


def test_landmark_names_never_overlap():
    spans = [(-65, -55), (-50, -40), (-16, -6), (-8, 2), (-2, 8), (3, 13), (9, 19), (48, 58)]
    rows = label_rows(spans)
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            if rows[i] == rows[j]:
                (a0, a1), (b0, b1) = spans[i], spans[j]
                assert a1 <= b0 or b1 <= a0


def test_a_warning_shows_once(qtbot):
    from ecg.qt.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w.set_status("Something went wrong", warn=True)
    assert w.statusBar().currentMessage() == "Something went wrong"
    assert "transparent" in w.statusBar().styleSheet()   # the notice shows it
    assert w.toast.text() == "Something went wrong"
    w.toast.hide()
    assert "transparent" not in w.statusBar().styleSheet()
    w.set_status("Added peak", warn=True, notice=False)  # a routine note: no notice
    assert not w.toast.isVisible() and w.statusBar().currentMessage() == "Added peak"
