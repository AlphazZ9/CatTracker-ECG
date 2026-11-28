"""The two charts of the Signal quality panel, in PyQtGraph: each beat's
correlation with the recording's template over time (with its rolling mean
and the 0.90 threshold) and their histogram.

Both scales start a little under the bulk of the beats (scale_start), so
the band near 1 where most beats lie is readable; the few beats under it
show at the edge (triangles over time, one red bar in the histogram). The
beats are green (>= 0.95), amber (0.90-0.95) or red (< 0.90).

They replace on screen the Matplotlib figures quality_time / quality_hist,
which stay, hidden, for the exports: drawn on every step of Analyze, those
froze the window for a tenth of a second or more each time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ecg.qt import theme
from ecg.qt.widgets.rhythm_charts import _alpha, _Chart

THRESHOLD = 0.90
GOOD = 0.95          # at or above: a clean beat (green); THRESHOLD to GOOD: amber; below: red
FLOOR = 0.5          # the scales never start lower: what is below is shown at the edge


class QualityCharts(QtCore.QObject):
    """Correlation over time (``time``) and its histogram (``hist``)."""

    def __init__(self, parent: "Optional[QtCore.QObject]" = None) -> None:
        super().__init__(parent)
        self.time = _Chart()
        self.time.setMinimumHeight(130)
        self.time.setLabel("bottom", "Time", units="s")
        self.time.setLabel("left", "Correlation")
        self.t_points = pg.ScatterPlotItem(size=3, pen=None)
        # Beats below the scale: a small triangle at its bottom edge.
        self.t_below = pg.ScatterPlotItem(size=6, symbol="t", pen=None)
        self.t_roll = pg.PlotDataItem()
        self.t_thr = pg.InfiniteLine(pos=THRESHOLD, angle=0, movable=False)
        for item in (self.t_points, self.t_below, self.t_roll, self.t_thr):
            self.time.addItem(item)

        self.hist = _Chart()
        self.hist.setMinimumHeight(110)
        self.hist.setLabel("bottom", "Correlation with template")
        self.hist.setLabel("left", "Beats")
        self.h_bars = pg.BarGraphItem(x=[], height=[], width=1.0)
        # The beats below the scale's start, in one bar at its left (red).
        self.h_below = pg.BarGraphItem(x=[], height=[], width=1.0)
        self.h_thr = pg.InfiniteLine(pos=THRESHOLD, angle=90, movable=False)
        self.hist.addItem(self.h_bars)
        self.hist.addItem(self.h_below)
        self.hist.addItem(self.h_thr)

        for chart in (self.time, self.hist):
            for ax in ("left", "bottom"):
                chart.getAxis(ax).setStyle(tickFont=_small_font())
        self._series: "Optional[dict]" = None
        self.apply_palette(theme.current())
        self.clear()

    # ── data ────────────────────────────────────────────────
    def clear(self) -> None:
        self._series = None
        self.t_points.setData([], [])
        self.t_below.setData([], [])
        self.t_roll.setData([], [])
        self.h_bars.setOpts(x=[], height=[], width=1.0)
        self.h_below.setOpts(x=[], height=[], width=1.0)
        self.t_thr.hide()
        self.h_thr.hide()
        self.time.say("After Analyze")
        self.hist.say("")

    def set_series(self, q: dict) -> None:
        """*q*: ecg.app.plot_controller.quality_plot_series."""
        self._series = q
        self._draw()

    def _draw(self) -> None:
        q = self._series
        if q is None:
            return
        t = np.asarray(q["t"], dtype=float)
        v = np.asarray(q["v"], dtype=float)
        n = min(len(t), len(v))
        t, v = t[:n], v[:n]
        if not n:
            self.clear()
            self.time.say("No quality data for the current window")
            return
        self.time.say("")
        # The scale starts a little under the bulk of the beats (at most at
        # 0.85, never under FLOOR), so the band near 1 where most beats are
        # is readable; the few beats under it sit at the bottom edge.
        lo = scale_start(v)
        inside = v >= lo
        cls = np.where(v >= GOOD, 0, np.where(v >= THRESHOLD, 1, 2))
        brushes = np.array(self._brushes, dtype=object)[cls[inside]]
        self.t_points.setData(t[inside], v[inside], brush=list(brushes))
        below = ~inside & np.isfinite(v)
        self.t_below.setData(t[below], np.full(int(below.sum()), lo + 0.004))
        self.t_roll.setData(np.asarray(q["roll_t"], float), np.asarray(q["roll"], float))
        self.t_thr.show()
        self.time.setYRange(lo, 1.005, padding=0)
        self.time.setXRange(float(t[0]), float(t[-1]), padding=0.01)

        allv = np.asarray(q["all"], dtype=float)
        allv = allv[np.isfinite(allv)]
        if len(allv):
            h_lo = scale_start(allv)
            n_bins = min(30, max(10, len(allv) // 4))
            edges = np.linspace(h_lo, 1.0, n_bins + 1)
            counts, _e = np.histogram(np.clip(allv, h_lo, 1.0)[allv >= h_lo], bins=edges)
            width = float(edges[1] - edges[0])
            self.h_bars.setOpts(x=(edges[:-1] + edges[1:]) / 2, height=counts,
                                width=width * 0.9)
            n_below = int((allv < h_lo).sum())
            self.h_below.setOpts(x=[h_lo - width * 0.6] if n_below else [],
                                 height=[n_below] if n_below else [], width=width * 0.9)
            self.h_thr.show()
            self.hist.say("")
            self.hist.setXRange(h_lo - (width * 1.3 if n_below else 0.0), 1.0, padding=0.02)
            self.hist.setYRange(0, float(max(counts.max(), n_below)) * 1.05, padding=0)
            self.hist.setToolTip(f"{n_below} beat(s) below {h_lo:.2f}: the red bar at the left"
                                 if n_below else "")

    # ── look ────────────────────────────────────────────────
    def apply_palette(self, p: "theme.Palette") -> None:
        self.time.style(p)
        self.hist.style(p)
        # Three plain colours that read on light and dark backgrounds alike:
        # clean beats green, between the threshold and 0.95 amber, below red.
        self._brushes = [pg.mkBrush(_alpha(c, 150)) for c in (p.GREEN, p.ORANGE, p.RED)]
        self.t_below.setBrush(pg.mkBrush(p.RED))
        self.t_roll.setPen(pg.mkPen(p.TEXT, width=1))
        thr = pg.mkPen(p.MUTED, width=1, style=QtCore.Qt.PenStyle.DashLine)
        self.t_thr.setPen(thr)
        self.h_thr.setPen(thr)
        self.h_bars.setOpts(brush=pg.mkBrush(_alpha(p.GREEN, 190)), pen=pg.mkPen(None))
        self.h_below.setOpts(brush=pg.mkBrush(_alpha(p.RED, 190)), pen=pg.mkPen(None))
        self._draw()


def scale_start(values: np.ndarray) -> float:
    """Where the correlation scales start: a little under the lowest
    percent of the beats, at most 0.85 (the 0.90 line always shows), never
    under FLOOR."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return 0.85
    low = float(np.percentile(v, 1)) - 0.02
    return float(min(0.85, max(FLOOR, np.floor(low * 50) / 50)))


def _small_font():
    f = QtWidgets.QApplication.font()
    f.setPointSizeF(max(7.0, f.pointSizeF() - 2))
    return f
