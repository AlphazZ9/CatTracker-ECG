"""The two charts of the Signal quality panel, in PyQtGraph: each beat's
correlation with the recording's template over time (with its rolling mean
and the 0.90 threshold) and their histogram.

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
_LEVELS = 8          # colour steps of the beats (viridis, 0.70 → 1.00)


class QualityCharts(QtCore.QObject):
    """Correlation over time (``time``) and its histogram (``hist``)."""

    def __init__(self, parent: "Optional[QtCore.QObject]" = None) -> None:
        super().__init__(parent)
        self.time = _Chart()
        self.time.setMinimumHeight(130)
        self.time.setLabel("bottom", "Time", units="s")
        self.time.setLabel("left", "Correlation")
        self.t_points = pg.ScatterPlotItem(size=3, pen=None)
        self.t_roll = pg.PlotDataItem()
        self.t_thr = pg.InfiniteLine(pos=THRESHOLD, angle=0, movable=False)
        for item in (self.t_points, self.t_roll, self.t_thr):
            self.time.addItem(item)

        self.hist = _Chart()
        self.hist.setMinimumHeight(110)
        self.hist.setLabel("bottom", "Correlation with template")
        self.hist.setLabel("left", "Beats")
        self.h_bars = pg.BarGraphItem(x=[], height=[], width=1.0)
        self.h_thr = pg.InfiniteLine(pos=THRESHOLD, angle=90, movable=False)
        self.hist.addItem(self.h_bars)
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
        self.t_roll.setData([], [])
        self.h_bars.setOpts(x=[], height=[], width=1.0)
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
        lvl = np.clip(((v - 0.70) / 0.30 * (_LEVELS - 1)).round(), 0, _LEVELS - 1)
        brushes = np.array(self._brushes, dtype=object)[np.nan_to_num(lvl).astype(int)]
        self.t_points.setData(t, v, brush=brushes)
        self.t_roll.setData(np.asarray(q["roll_t"], float), np.asarray(q["roll"], float))
        self.t_thr.show()
        lo = max(0.0, float(np.nanmin(v)) - 0.05) if np.isfinite(v).any() else 0.0
        self.time.setYRange(lo, 1.02, padding=0)
        self.time.setXRange(float(t[0]), float(t[-1]), padding=0.01)

        allv = np.asarray(q["all"], dtype=float)
        if len(allv):
            # Same bin rule as the Matplotlib figure (and the Beat Template).
            counts, edges = np.histogram(allv, bins=min(40, max(10, len(allv) // 4)))
            self.h_bars.setOpts(x=(edges[:-1] + edges[1:]) / 2, height=counts,
                                width=float(edges[1] - edges[0]) * 0.92)
            self.h_thr.show()
            self.hist.say("")
            self.hist.setXRange(min(float(edges[0]), THRESHOLD - 0.02), 1.0, padding=0.02)
            self.hist.setYRange(0, float(counts.max()) * 1.05, padding=0)

    # ── look ────────────────────────────────────────────────
    def apply_palette(self, p: "theme.Palette") -> None:
        self.time.style(p)
        self.hist.style(p)
        cmap = pg.colormap.get("viridis")
        self._brushes = [pg.mkBrush(_alpha(cmap.map(i / (_LEVELS - 1), mode="qcolor").name(),
                                           110)) for i in range(_LEVELS)]
        self.t_roll.setPen(pg.mkPen(p.BLUE, width=1))
        thr = pg.mkPen(p.ORANGE, width=1, style=QtCore.Qt.PenStyle.DotLine)
        self.t_thr.setPen(thr)
        self.h_thr.setPen(thr)
        self.h_bars.setOpts(brush=pg.mkBrush(_alpha(p.GREEN, 190)), pen=pg.mkPen(None))
        self._draw()


def _small_font():
    f = QtWidgets.QApplication.font()
    f.setPointSizeF(max(7.0, f.pointSizeF() - 2))
    return f
