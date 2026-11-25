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
