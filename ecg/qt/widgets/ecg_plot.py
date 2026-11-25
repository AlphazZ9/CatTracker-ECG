# -*- coding: utf-8 -*-
"""
ecg.qt.widgets.ecg_plot
-----------------------
The interactive ECG trace (PyQtGraph): a detail view of the signal with its
R-peaks (accepted, rejected candidates, excluded and added by hand) and the
noisy regions, above an overview of the whole recording whose highlighted
band is the detail's time window (drag it to navigate).

Display only. The signal arrays are never modified or copied: PyQtGraph
downsamples what it draws ("peak" mode keeps each pixel column's minimum and
maximum, so no QRS disappears) and clips it to the visible range. A one-hour
recording at 2 kHz (7.2 million samples) scrolls without delay.

Peaks are given as sample indices and drawn at index / fs seconds. A click
on the trace emits ``time_clicked(t_s, button)`` (button: "left", "right" or
"other") -- the edit rules (ecg.core.peak_editing.edit_at_click) belong to
the caller.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ecg.qt import theme

DEFAULT_WINDOW_S = 10.0
MIN_WINDOW_S = 0.1          # as the Tk views (ecg.ui.state.MIN_VIEW_WINDOW_S)


def _alpha(color: str, alpha: int) -> QtGui.QColor:
    """*color* with opacity *alpha* (0-255): shading that shows the trace
    under it, on light and dark backgrounds alike."""
    c = QtGui.QColor(color)
    c.setAlpha(alpha)
    return c


class EcgPlot(QtWidgets.QWidget):
    """Detail trace + overview of one signal."""

    time_clicked = QtCore.Signal(float, str)        # time (s), "left" / "right" / "other"
    window_changed = QtCore.Signal(float, float)    # t0, t1 (s)
    span_drawn = QtCore.Signal(float, float)        # a left drag in span mode: t0, t1 (s)

    def __init__(self, parent: "QtWidgets.QWidget | None" = None) -> None:
        super().__init__(parent)
        self._fs: float = 1.0
        self._signal: "Optional[np.ndarray]" = None
        self._time: "Optional[np.ndarray]" = None
        self._regions: "list[pg.LinearRegionItem]" = []
        self._annotations: list = []
        self._syncing = False

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.view = pg.GraphicsLayoutWidget()
        lay.addWidget(self.view)
        self.detail = self.view.addPlot(row=0, col=0)
        self.overview = self.view.addPlot(row=1, col=0)
        self.view.ci.layout.setRowStretchFactor(0, 5)
        self.view.ci.layout.setRowStretchFactor(1, 1)
        for p in (self.detail, self.overview):
            p.showGrid(x=True, y=True, alpha=0.15)
            p.setMenuEnabled(False)
        self.detail.setLabel("bottom", "Time", units="s")
        self.detail.setMouseEnabled(x=True, y=False)
        self.overview.setMouseEnabled(x=False, y=False)
        self.overview.hideAxis("left")

        self.trace = pg.PlotDataItem(pen=pg.mkPen(width=1))
        self.trace.setDownsampling(auto=True, method="peak")
        self.trace.setClipToView(True)
        self.detail.addItem(self.trace)
        self.over_trace = pg.PlotDataItem(pen=pg.mkPen(width=1))
        self.over_trace.setDownsampling(auto=True, method="peak")
        self.overview.addItem(self.over_trace)

        self.accepted = pg.ScatterPlotItem(size=7, symbol="o")
        self.rejected = pg.ScatterPlotItem(size=5, symbol="o")
        self.excluded = pg.ScatterPlotItem(size=9, symbol="x")
        self.added = pg.ScatterPlotItem(size=8, symbol="t")
        for item in (self.rejected, self.accepted, self.excluded, self.added):
            self.detail.addItem(item)

        self.band = pg.LinearRegionItem(values=(0, DEFAULT_WINDOW_S), movable=True)
        self.overview.addItem(self.band)
        self.band.sigRegionChanged.connect(self._band_moved)
        self.detail.sigXRangeChanged.connect(self._detail_moved)
        self.detail.sigYRangeChanged.connect(lambda *_a: self._place_labels())
        self.detail.scene().sigMouseClicked.connect(self._clicked)
        # Span mode (Mark Noise): a left drag draws a span instead of panning.
        self.span_mode = False
        self._span: "Optional[pg.LinearRegionItem]" = None
        vb = self.detail.getViewBox()
        self._pan_drag = vb.mouseDragEvent
        vb.mouseDragEvent = self._drag
        self.apply_palette(theme.current())

    # ── data ────────────────────────────────────────────────
    def set_signal(self, signal: np.ndarray, fs: float,
                   time: "Optional[np.ndarray]" = None) -> None:
        """Show *signal* sampled at *fs* Hz (*time*: its time axis in s, if
        already built; else i / fs)."""
        self._signal, self._fs = signal, float(fs)
        self._time = time if time is not None else np.arange(len(signal)) / self._fs
        self.trace.setData(self._time, signal)
        self.over_trace.setData(self._time, signal)
        end = float(self._time[-1]) if len(self._time) else 0.0
        self.overview.setXRange(0, end, padding=0)
        self.set_window(0.0, min(DEFAULT_WINDOW_S, end))

    def set_peaks(self, accepted: "Optional[np.ndarray]" = None,
                  rejected: "Optional[np.ndarray]" = None,
                  excluded: "Optional[np.ndarray]" = None,
                  added: "Optional[np.ndarray]" = None) -> None:
        """Mark the peaks (sample indices) on the detail trace."""
        for item, peaks in ((self.accepted, accepted), (self.rejected, rejected),
                            (self.excluded, excluded), (self.added, added)):
            if self._signal is None or peaks is None or len(peaks) == 0:
                item.setData([], [])
                continue
            idx = np.asarray(peaks, dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < len(self._signal))]
            item.setData(idx / self._fs, self._signal[idx])

    def set_noise_regions(self, regions: "list[dict]") -> None:
        """Shade the noisy regions ({t_start, t_end} in s)."""
        for r in self._regions:
            self.detail.removeItem(r)
        self._regions = []
        p = theme.current()
        for r in regions or []:
            item = pg.LinearRegionItem(values=(float(r["t_start"]), float(r["t_end"])),
                                       movable=False, brush=pg.mkBrush(_alpha(p.ORANGE, 50)))
            item.setZValue(-10)
            self.detail.addItem(item)
            self._regions.append(item)

    def set_annotations(self, annotations: "list[dict]") -> None:
        """Shade the annotated periods ({t_start, t_end, label, color}) in
        their colour, with their label."""
        for item in self._annotations:
            self.detail.removeItem(item)
        self._annotations = []
        for a in annotations or []:
            t0, t1 = float(a["t_start"]), float(a["t_end"])
            color = a.get("color") or "#888888"
            item = pg.LinearRegionItem(values=(t0, t1), movable=False,
                                       brush=pg.mkBrush(_alpha(color, 40)),
                                       pen=pg.mkPen(color))
            item.setZValue(-20)
            self.detail.addItem(item)
            self._annotations.append(item)
            if a.get("label"):
                text = pg.TextItem(str(a["label"]), color=color, anchor=(0, 0))
                text.setZValue(-19)
                text.t_start = t0
                self.detail.addItem(text)
                self._annotations.append(text)
        self._place_labels()

    def _place_labels(self) -> None:
        """The labels at the top of the view, whatever its y range."""
        top = self.detail.viewRange()[1][1]
        for item in self._annotations:
            if isinstance(item, pg.TextItem):
                item.setPos(item.t_start, top)

    def clear(self) -> None:
        self.set_annotations([])
        self.set_noise_regions([])
        self._signal = self._time = None
        for item in (self.trace, self.over_trace):
            item.setData([], [])
        self.set_peaks()
        self.set_noise_regions([])

    # ── window ──────────────────────────────────────────────
    def window(self) -> "tuple[float, float]":
        (t0, t1), _ = self.detail.viewRange()
        return float(t0), float(t1)

    def set_window(self, t0: float, t1: float) -> None:
        if t1 - t0 < MIN_WINDOW_S:
            t1 = t0 + MIN_WINDOW_S
        self.detail.setXRange(t0, t1, padding=0)

    def _band_moved(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        t0, t1 = self.band.getRegion()
        self.detail.setXRange(t0, t1, padding=0)
        self._syncing = False

    def _detail_moved(self, *_a) -> None:
        t0, t1 = self.window()
        if not self._syncing:
            self._syncing = True
            self.band.setRegion((t0, t1))
            self._syncing = False
        self._fit_y(t0, t1)
        self.window_changed.emit(t0, t1)

    def _fit_y(self, t0: float, t1: float) -> None:
        if self._signal is None or not len(self._signal):
            return
        i0 = max(0, int(t0 * self._fs))
        i1 = min(len(self._signal), int(np.ceil(t1 * self._fs)) + 1)
        if i1 - i0 < 2:
            return
        seg = self._signal[i0:i1]
        lo, hi = float(np.nanmin(seg)), float(np.nanmax(seg))
        pad = 0.08 * (hi - lo or 1.0)
        self.detail.setYRange(lo - pad, hi + pad, padding=0)

    # ── interaction ─────────────────────────────────────────
    def _drag(self, ev, axis=None) -> None:
        if not self.span_mode or ev.button() != QtCore.Qt.MouseButton.LeftButton:
            self._pan_drag(ev, axis)
            return
        ev.accept()
        vb = self.detail.getViewBox()
        x0 = vb.mapSceneToView(ev.buttonDownScenePos()).x()
        x1 = vb.mapSceneToView(ev.scenePos()).x()
        a, b = sorted((float(x0), float(x1)))
        if self._span is None:
            self._span = pg.LinearRegionItem(values=(a, b), movable=False,
                                             brush=pg.mkBrush(_alpha("#9AA0A6", 90)))
            self.detail.addItem(self._span)
        self._span.setRegion((a, b))
        if ev.isFinish():
            self.detail.removeItem(self._span)
            self._span = None
            self.span_drawn.emit(a, b)

    def _clicked(self, ev) -> None:
        if self._signal is None or not self.detail.sceneBoundingRect().contains(ev.scenePos()):
            return
        x = float(self.detail.vb.mapSceneToView(ev.scenePos()).x())
        if 0.0 <= x * self._fs < len(self._signal):
            b = ev.button()
            self.time_clicked.emit(x, "left" if b == QtCore.Qt.MouseButton.LeftButton
                                   else "right" if b == QtCore.Qt.MouseButton.RightButton
                                   else "other")

    # ── look ────────────────────────────────────────────────
    def apply_palette(self, p: "theme.Palette") -> None:
        self.view.setBackground(p.PANEL)
        for plot in (self.detail, self.overview):
            for ax in ("left", "bottom"):
                plot.getAxis(ax).setPen(pg.mkPen(p.MUTED))
                plot.getAxis(ax).setTextPen(pg.mkPen(p.MUTED))
        self.trace.setPen(pg.mkPen(p.plot_signal, width=1))
        self.over_trace.setPen(pg.mkPen(p.LIGHT, width=1))
        self.accepted.setBrush(pg.mkBrush(p.plot_rpeak))
        self.accepted.setPen(pg.mkPen(None))
        self.rejected.setBrush(pg.mkBrush(p.LIGHT))
        self.rejected.setPen(pg.mkPen(None))
        self.excluded.setBrush(pg.mkBrush(p.RED))
        self.excluded.setPen(pg.mkPen(p.RED))
        self.added.setBrush(pg.mkBrush("#14B8A6"))
        self.added.setPen(pg.mkPen(None))
        self.band.setBrush(pg.mkBrush(_alpha(p.BLUE, 70)))
        for line in self.band.lines:
            line.setPen(pg.mkPen(p.BLUE, width=1))
