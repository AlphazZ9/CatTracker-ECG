"""PyQtGraph versions, on screen, of two Matplotlib figures that took a
quarter to half a second to draw each time their tab was shown:

* AsymmetryCharts — RR asymmetry (Rhythm & HRV): the shares of
  decelerations / accelerations among the pairs with ΔRR ≠ 0 (50 % =
  symmetry) and the signed ΔRR histogram;
* BeatDistCharts — Beat distributions (Morphology): R-peak amplitude and
  correlation with the template.

The Matplotlib figures (slots ``asymmetry`` and ``beat_dist``) stay, hidden,
for the exports. Data: ecg.app.plot_controller.PlotController._on_screen.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ecg.qt import theme
from ecg.qt.widgets.rhythm_charts import _alpha, _Chart


def _hist(values: np.ndarray, bins: int) -> "tuple[np.ndarray, np.ndarray, float]":
    counts, edges = np.histogram(values, bins=bins)
    return (edges[:-1] + edges[1:]) / 2, counts, float(edges[1] - edges[0])


class _Pair(QtWidgets.QWidget):
    """Two charts side by side, each with a caption above it."""

    def __init__(self, parent: "Optional[QtWidgets.QWidget]" = None) -> None:
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)
        self.left, self.right = _Chart(), _Chart()
        self.cap_left, self.cap_right = QtWidgets.QLabel(""), QtWidgets.QLabel("")
        for cap, chart in ((self.cap_left, self.left), (self.cap_right, self.right)):
            col = QtWidgets.QVBoxLayout()
            col.setSpacing(2)
            cap.setProperty("muted", True)
            cap.setWordWrap(True)
            col.addWidget(cap)
            chart.setMinimumHeight(180)
            col.addWidget(chart, 1)
            lay.addLayout(col, 1)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(600, 240)


class AsymmetryCharts(_Pair):
    def __init__(self, parent: "Optional[QtWidgets.QWidget]" = None) -> None:
        super().__init__(parent)
        self.left.setLabel("left", "% of pairs with ΔRR ≠ 0")
        self.left.getAxis("bottom").setTicks([[(0, "Decel. (RR↑)"),
                                              (1, "Accel. (RR↓)")]])
        self.bars = pg.BarGraphItem(x=[0, 1], height=[0, 0], width=0.5)
        self.half = pg.InfiniteLine(pos=50, angle=0, movable=False)
        self.vals = [pg.TextItem("", anchor=(0.5, 1.0)) for _ in range(2)]
        for item in (self.bars, self.half, *self.vals):
            self.left.addItem(item)
        self.left.setXRange(-0.6, 1.6, padding=0)
        self.left.setYRange(0, 105, padding=0)
        self.right.setLabel("bottom", "ΔRR (ms)")
        self.right.setLabel("left", "Beats")
        self.h_acc = pg.BarGraphItem(x=[], height=[], width=1.0)
        self.h_dec = pg.BarGraphItem(x=[], height=[], width=1.0)
        self.zero = pg.InfiniteLine(pos=0, angle=90, movable=False)
        for item in (self.h_acc, self.h_dec, self.zero):
            self.right.addItem(item)
        self._data: "Optional[dict]" = None
        self._empty = True
        self.apply_palette(theme.current())
        self.clear()

    def clear(self, text: str = "Click Analyze: decelerations vs accelerations") -> None:
        self._data = None
        self.bars.setOpts(height=[0, 0])
        for v in self.vals:
            v.setText("")
        self.h_acc.setOpts(x=[], height=[])
        self.h_dec.setOpts(x=[], height=[])
        self.cap_left.setText("RR asymmetry")
        self.cap_right.setText("ΔRR distribution")
        self.left.say(text)
        self.right.say("")

    def set_data(self, d: "Optional[dict]") -> None:
        if d is None:
            from ecg.core.metric_table import MIN_PAIRS_ASYMMETRY
            self.clear(f"Needs ≥ {MIN_PAIRS_ASYMMETRY} successive valid RR pairs")
            return
        self._data = d
        self._draw()

    def _draw(self) -> None:
        d = self._data
        if d is None:
            return
        n_off = d["n_off"]
        vals = [d["n_dec"] / n_off * 100, d["n_acc"] / n_off * 100] if n_off else [0.0, 0.0]
        self.bars.setOpts(height=vals)
        for item, x, v in zip(self.vals, (0, 1), vals):
            item.setText(f"{v:.1f} %" if n_off else "—")
            item.setPos(x, v + 2)
        self.left.say("")
        # Porta PI / Guzik GI are on the line above the charts.
        self.cap_left.setText(f"Shares of the pairs with ΔRR ≠ 0 (dashed: 50 %, symmetry) · "
                              f"no change: {d['n_zero']} of {d['n_tot']} pairs")
        drr = np.asarray(d["drr"], dtype=float)
        top = 1.0
        for item, part in ((self.h_acc, drr[drr < 0]), (self.h_dec, drr[drr > 0])):
            if len(part):
                x, c, w = _hist(part, 30)
                item.setOpts(x=x, height=c, width=w * 0.92)
                top = max(top, float(c.max()))
            else:
                item.setOpts(x=[], height=[])
        self.right.setYRange(0, top * 1.05, padding=0)
        self.cap_right.setText("ΔRR distribution: accelerations (blue), "
                               "decelerations (red)")

    def apply_palette(self, p: "theme.Palette") -> None:
        self.left.style(p)
        self.right.style(p)
        self.bars.setOpts(brushes=[pg.mkBrush(_alpha(p.ACCENT, 205)),
                                   pg.mkBrush(_alpha(p.BLUE, 205))], pen=pg.mkPen(None))
        dash = pg.mkPen(p.BORDER2, width=1, style=QtCore.Qt.PenStyle.DashLine)
        self.half.setPen(dash)
        self.zero.setPen(dash)
        for v in self.vals:
            v.setColor(p.TEXT)
        self.h_acc.setOpts(brush=pg.mkBrush(_alpha(p.BLUE, 180)), pen=pg.mkPen(None))
        self.h_dec.setOpts(brush=pg.mkBrush(_alpha(p.ACCENT, 180)), pen=pg.mkPen(None))
        self._draw()


class BeatDistCharts(_Pair):
    def __init__(self, parent: "Optional[QtWidgets.QWidget]" = None) -> None:
        super().__init__(parent)
        self.left.setLabel("bottom", "R-peak amplitude (norm.)")
        self.left.setLabel("left", "Count")
        self.right.setLabel("bottom", "Correlation with template")
        self.right.setLabel("left", "Count")
        self.h_amp = pg.BarGraphItem(x=[], height=[], width=1.0)
        self.left.addItem(self.h_amp)
        self.h_corr = pg.BarGraphItem(x=[], height=[], width=1.0)
        self.mean = pg.InfiniteLine(pos=0, angle=90, movable=False)
        self.thr = pg.InfiniteLine(pos=0.90, angle=90, movable=False)
        for item in (self.h_corr, self.mean, self.thr):
            self.right.addItem(item)
        self._data: "Optional[dict]" = None
        self.apply_palette(theme.current())
        self.clear()

    def clear(self) -> None:
        self._data = None
        self.h_amp.setOpts(x=[], height=[])
        self.h_corr.setOpts(x=[], height=[])
        self.mean.hide()
        self.thr.hide()
        self.cap_left.setText("R-peak amplitude")
        self.cap_right.setText("Beat morphology")
        self.left.say("Click Analyze: the beats' amplitudes and shapes")
        self.right.say("")

    def set_data(self, d: "Optional[dict]") -> None:
        if d is None:
            self.clear()
            return
        self._data = d
        self._draw()

    def _draw(self) -> None:
        d = self._data
        if d is None:
            return
        # Every scored beat (n_beats counts the ghost traces only).
        n = max(len(d["beat_corr"]), 1)
        amps = d["peak_amps"][np.isfinite(d["peak_amps"])]
        corr = d["beat_corr"][np.isfinite(d["beat_corr"])]
        self.left.say("")
        for chart, item, values, bins in (
                (self.left, self.h_amp, amps, min(50, max(10, n // 4))),
                (self.right, self.h_corr, corr, min(40, max(10, n // 4)))):
            if len(values):
                x, c, w = _hist(values, bins)
                item.setOpts(x=x, height=c, width=w * 0.92)
                chart.setYRange(0, float(c.max()) * 1.05, padding=0)
            else:
                item.setOpts(x=[], height=[])
        self.mean.setValue(d["mean_corr"])
        self.mean.show()
        self.thr.show()
        many = d["n_bad"] > n * 0.1
        self.cap_right.setText(
            f"Beat morphology: {d['n_bad']} beats < 0.90 · mean {d['mean_corr']:.3f} "
            f"(dashed) · 0.90 threshold (dotted)")
        self.cap_right.setProperty("muted", not many)
        self.cap_right.setStyleSheet(f"color: {theme.current().ORANGE};" if many else "")

    def apply_palette(self, p: "theme.Palette") -> None:
        self.left.style(p)
        self.right.style(p)
        self.h_amp.setOpts(brush=pg.mkBrush(_alpha(p.BLUE, 190)), pen=pg.mkPen(None))
        self.h_corr.setOpts(brush=pg.mkBrush(_alpha(p.GREEN, 190)), pen=pg.mkPen(None))
        self.mean.setPen(pg.mkPen(p.ACCENT, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.thr.setPen(pg.mkPen(p.ORANGE, width=1, style=QtCore.Qt.PenStyle.DotLine))
        self._draw()
