"""ecg.qt.widgets: the ECG plot shows one hour at 2 kHz without touching the
data, and the Matplotlib canvas replays a draw_fn(fig)."""
import time

import numpy as np

from ecg.qt.widgets.ecg_plot import EcgPlot
from ecg.qt.widgets.mpl_canvas import MplCanvas

FS = 2000.0


def test_one_hour_is_shown_and_the_signal_is_untouched(qtbot):
    sig = np.random.default_rng(0).normal(size=int(3600 * FS)).astype(np.float64)
    before = sig.copy()
    plot = EcgPlot()
    qtbot.addWidget(plot)
    plot.resize(1200, 500)
    plot.show()
    t0 = time.perf_counter()
    plot.set_signal(sig, FS)
    plot.set_peaks(accepted=np.arange(0, len(sig), 240))
    plot.set_window(1800.0, 1810.0)
    qtbot.wait(50)
    elapsed = time.perf_counter() - t0
    assert np.array_equal(sig, before)                       # display never alters data
    assert plot.window() == (1800.0, 1810.0)
    lo, hi = plot.band.getRegion()
    assert (round(lo, 6), round(hi, 6)) == (1800.0, 1810.0)  # overview band follows
    assert elapsed < 10.0, elapsed


def test_peaks_are_drawn_at_index_over_fs(qtbot):
    plot = EcgPlot()
    qtbot.addWidget(plot)
    sig = np.sin(np.arange(4000) / 50.0)
    plot.set_signal(sig, FS)
    plot.set_peaks(accepted=np.array([100, 2000, 99999]))     # out-of-range ignored
    x, y = plot.accepted.getData()
    assert np.allclose(x, [0.05, 1.0]) and np.allclose(y, sig[[100, 2000]])


def test_matplotlib_canvas_replays_its_draw_function(qtbot):
    canvas = MplCanvas()
    qtbot.addWidget(canvas)
    calls = []

    def draw(fig):
        calls.append(1)
        fig.add_subplot().plot([0, 1], [1, 0])
    canvas.update_figure(draw)
    canvas.redraw()
    assert len(calls) == 2 and len(canvas.figure.axes) == 1


# ── zoom and navigation as the Tk detail view ────────────────

class _Wheel:
    """A wheel notch over the trace (QGraphicsSceneWheelEvent's calls)."""

    def __init__(self, delta, pos):
        self._d, self._p = delta, pos

    def delta(self):
        return self._d

    def orientation(self):
        from PySide6 import QtCore
        return QtCore.Qt.Orientation.Vertical

    def pos(self):
        return self._p

    def accept(self):
        pass


class _Drag:
    """A finished (or ongoing) mouse drag on the trace's view box."""

    def __init__(self, p0, p1, finish=True, right=False, start=False):
        self.p0, self.p1, self.finish, self.right, self.start = p0, p1, finish, right, start

    def button(self):
        from PySide6 import QtCore
        return (QtCore.Qt.MouseButton.RightButton if self.right
                else QtCore.Qt.MouseButton.LeftButton)

    def buttonDownPos(self, *_a):
        return self.p0

    def pos(self):
        return self.p1

    def isStart(self):
        return self.start

    def isFinish(self):
        return self.finish

    def accept(self):
        pass


def _plot(qtbot, seconds=60.0):
    plot = EcgPlot()
    qtbot.addWidget(plot)
    plot.resize(1000, 500)
    plot.show()
    t = np.arange(int(seconds * FS)) / FS
    plot.set_signal(np.sin(2 * np.pi * 8 * t) + 0.1 * np.sin(2 * np.pi * 0.3 * t), FS)
    plot.set_window(10.0, 12.0)
    qtbot.wait(20)
    return plot


def _view_point(plot, t, y=0.0):
    from PySide6 import QtCore
    vb = plot.detail.getViewBox()
    return vb.mapFromView(QtCore.QPointF(t, y))


def test_wheel_zooms_the_time_around_the_pointer_as_tk(qtbot):
    from ecg.qt.widgets.ecg_plot import MIN_WINDOW_S
    plot = _plot(qtbot)
    plot._wheel(_Wheel(120, _view_point(plot, 11.5)))      # one notch up: 0.8 times
    t0, t1 = plot.window()
    assert abs((t1 - t0) - 1.6) < 1e-9
    assert abs((11.5 - t0) / (t1 - t0) - 0.75) < 1e-9       # the pointer's time stays put
    plot._wheel(_Wheel(-120, _view_point(plot, 11.5)))     # one notch down: 1.25 times
    t0, t1 = plot.window()
    assert abs((t1 - t0) - 2.0) < 1e-9
    for _ in range(40):                                    # never under the minimum...
        plot._wheel(_Wheel(120, _view_point(plot, 11.0)))
    t0, t1 = plot.window()
    assert abs((t1 - t0) - MIN_WINDOW_S) < 1e-9
    for _ in range(40):                                    # ...nor past the recording
        plot._wheel(_Wheel(-120, _view_point(plot, 11.0)))
    assert plot.window() == (0.0, 60.0 - 1 / FS)


def test_zoom_box_back_forward_and_home(qtbot):
    plot = _plot(qtbot)
    fitted_y = plot.detail.viewRange()[1]
    plot.set_nav_mode("zoom")
    a, b = _view_point(plot, 10.5, 0.8), _view_point(plot, 11.0, -0.4)
    plot._drag(_Drag(a, b))
    (x0, x1), (y0, y1) = plot.detail.viewRange()
    assert abs(x0 - 10.5) < 1e-6 and abs(x1 - 11.0) < 1e-6
    assert abs(y0 + 0.4) < 1e-6 and abs(y1 - 0.8) < 1e-6
    plot.flush_fit()                                       # a height set by hand stays
    assert abs(plot.detail.viewRange()[1][1] - 0.8) < 1e-6
    # A flat box: the time only (the height fitted to what is in view).
    flat_a, flat_b = _view_point(plot, 10.6, 0.2), _view_point(plot, 10.8, 0.2)
    plot._drag(_Drag(flat_a, flat_b))
    x0, x1 = plot.detail.viewRange()[0]
    assert abs(x0 - 10.6) < 1e-6 and abs(x1 - 10.8) < 1e-6
    assert plot.act_back.isEnabled() and not plot.act_forward.isEnabled()
    plot.go_back()
    assert abs(plot.detail.viewRange()[0][0] - 10.5) < 1e-6
    assert abs(plot.detail.viewRange()[1][1] - 0.8) < 1e-6
    plot.go_forward()
    assert abs(plot.detail.viewRange()[0][0] - 10.6) < 1e-6
    plot.go_home()
    assert plot.window() == (10.0, 12.0)
    assert np.allclose(plot.detail.viewRange()[1], fitted_y)
    # Right button: zoom out, the view shrinking into the box.
    a, b = _view_point(plot, 10.5, 0.0), _view_point(plot, 11.5, 0.0)
    plot._drag(_Drag(a, b, right=True))
    x0, x1 = plot.detail.viewRange()[0]
    assert abs((x1 - x0) - 4.0) < 1e-6 and abs(x0 - 9.0) < 1e-6
    # A move of the time window is the new Home and drops the tool views.
    plot.set_window(20.0, 22.0)
    assert not plot.act_back.isEnabled()
    plot.go_home()
    assert plot.window() == (20.0, 22.0)


def test_pan_tool_moves_both_axes_and_tools_exclude_each_other(qtbot):
    plot = _plot(qtbot)
    modes = []
    plot.nav_mode_changed.connect(modes.append)
    vb = plot.detail.getViewBox()
    assert vb.state["mouseEnabled"] == [True, False]       # no tool: the time only
    plot.act_pan.trigger()
    assert plot.nav_mode() == "pan" and vb.state["mouseEnabled"] == [True, True]
    plot.act_zoom.trigger()
    assert plot.nav_mode() == "zoom" and not plot.act_pan.isChecked()
    plot.act_zoom.trigger()
    assert plot.nav_mode() == "" and modes == ["pan", "zoom", ""]
    assert vb.state["mouseEnabled"] == [True, False]


def test_edit_preview_and_the_tk_marks(qtbot):
    from ecg.qt.widgets.ecg_plot import ADD_COLOR, CHANGE_COLOR
    plot = _plot(qtbot)
    plot.show_edit_preview(int(10.7 * FS), "add")
    assert plot.pv_point.isVisible() and plot.pv_line.isVisible()
    assert plot.pv_text.textItem.toPlainText() == "10.700 s"
    assert plot.pv_point.opts["brush"].color().name() == ADD_COLOR.lower()
    plot.show_edit_preview(int(10.7 * FS), "replace")
    assert plot.pv_point.opts["brush"].color().name() == CHANGE_COLOR.lower()
    plot.show_edit_preview(None)
    assert not plot.pv_point.isVisible() and not plot.pv_text.isVisible()
    # A star on a peak added by hand, a cross on one excluded, the accepted
    # ones paler when less sure, the beats the review removed, the threshold.
    acc = np.array([int(10.2 * FS), int(10.4 * FS), int(10.6 * FS)])
    plot.set_peaks(accepted=acc, excluded=[int(10.8 * FS)], added=[int(11.0 * FS)],
                   accepted_alpha=np.array([0.95, 0.35, 0.65]))
    assert plot.added.opts["symbol"] == "star" and len(plot.added.getData()[0]) == 1
    assert plot.excluded.opts["symbol"] == "x" and len(plot.excluded.getData()[0]) == 1
    alphas = [b.color().alpha() for b in plot.accepted.data["brush"]]
    assert alphas[0] > alphas[2] > alphas[1]
    plot.set_artifacts({"ectopic": np.array([int(11.2 * FS)])})
    assert len(plot.artifact_marks["ectopic"].getData()[0]) == 1
    plot.set_threshold(0.5)
    assert plot.threshold_line.isVisible() and plot.threshold_line.value() == 0.5
    plot.set_threshold(None)
    assert not plot.threshold_line.isVisible()


def test_save_image_writes_the_view(qtbot, tmp_path):
    plot = _plot(qtbot)
    out = plot.save_image(str(tmp_path / "view.png"))
    assert out and (tmp_path / "view.png").stat().st_size > 1000
