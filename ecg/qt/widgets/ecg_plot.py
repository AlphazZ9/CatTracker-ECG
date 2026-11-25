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

# Two seconds: about 16 beats of a mouse at 500 bpm, each QRS readable (ten
# seconds packed some 85 beats into the view).
DEFAULT_WINDOW_S = 2.0
MIN_WINDOW_S = 0.1          # as the Tk views (ecg.app.state.MIN_VIEW_WINDOW_S)
AXIS_WIDTH_PX = 52          # every lane's value axis: their time axes line up
TRACE_MIN_HEIGHT_PX = 170   # the trace keeps this much; the strip gives way first
FIT_DELAY_MS = 40           # vertical fits while panning: at most ~25 per second
COMPACT_HEIGHT_PX = 460     # below this, the strip drops its Quality lane
OVERVIEW_HEIGHT_PX = 120
STRIP_MIN_PX, STRIP_MAX_PX = 56, 90   # each RR / HR / Quality lane
HR_SMOOTH_BEATS = 9                   # the overview's heart rate: a running median


def _alpha(color: str, alpha: int) -> QtGui.QColor:
    """*color* with opacity *alpha* (0-255): shading that shows the trace
    under it, on light and dark backgrounds alike."""
    c = QtGui.QColor(color)
    c.setAlpha(alpha)
    return c


def _running_median(values: np.ndarray, n: int) -> np.ndarray:
    """The median of each run of *n* values centred on each one (the ends
    take the nearest full run): the overview's heart-rate trend, without the
    beat-to-beat scatter that turned it into a solid band. Display only."""
    v = np.asarray(values, dtype=float)
    if len(v) < n or n < 3:
        return v
    from numpy.lib.stride_tricks import sliding_window_view
    med = np.median(sliding_window_view(v, n), axis=1)
    half = n // 2
    return np.concatenate([np.full(half, med[0]), med, np.full(len(v) - len(med) - half, med[-1])])


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
        lay.addWidget(self.view, 1)
        self.detail = self.view.addPlot(row=0, col=0)
        self.view.ci.layout.setRowStretchFactor(0, 5)
        # The whole recording, in its own view (a page may place it apart,
        # e.g. beside the heart rate figures): the heart rate of every beat
        # once there are peaks, else the signal; the band is the part shown
        # above, drag it to navigate.
        self.over_view = pg.GraphicsLayoutWidget()
        self.over_view.setFixedHeight(OVERVIEW_HEIGHT_PX)
        lay.addWidget(self.over_view)
        self.overview = self.over_view.addPlot(row=0, col=0)
        # The RR / HR / Quality strip under the trace (the Tk detail_rrhr
        # panel): each beat's RR and HR at its second peak, and its template
        # correlation, on the trace's time axis.
        # One time axis for the whole stack (the trace's): the lanes under it
        # share it instead of repeating it, which also spares three axes to
        # repaint on every move of the view.
        self.strip = []
        for r, label in ((1, "RR (ms)"), (2, "HR (bpm)"), (3, "Quality")):
            p = self.view.addPlot(row=r, col=0)
            p.setXLink(self.detail)
            p.setMouseEnabled(x=False, y=False)
            p.setMenuEnabled(False)
            p.setLabel("left", label)
            p.hideAxis("bottom")
            p.getAxis("left").setTickDensity(0.6)
            p.hideButtons()
            p.setMinimumHeight(STRIP_MIN_PX)
            p.setMaximumHeight(STRIP_MAX_PX)
            self.view.ci.layout.setRowStretchFactor(r, 1)
            self.strip.append(p)
        self.rr_curve, self.hr_curve, self.q_curve = (
            pg.PlotDataItem(symbol="o", symbolSize=3) for _ in range(3))
        for p, c in zip(self.strip, (self.rr_curve, self.hr_curve, self.q_curve)):
            c.setClipToView(True)
            p.addItem(c)
        self.q_line = pg.InfiniteLine(pos=0.90, angle=0, movable=False)  # the 0.90 corr. line
        self.strip[2].addItem(self.q_line)
        self._rr = None
        # Grid: the time lines of the trace only. A grid is drawn across the
        # whole plot with transparency on every repaint; on all five plots it
        # took more than half of the time of a move of the view.
        self.detail.showGrid(x=True, y=False, alpha=0.15)
        for p in (self.detail, self.overview):
            p.setMenuEnabled(False)
        self.detail.setLabel("bottom", "Time", units="s")
        self.detail.setMouseEnabled(x=True, y=False)
        self.detail.setMinimumHeight(TRACE_MIN_HEIGHT_PX)
        self.overview.setMouseEnabled(x=False, y=False)
        self.overview.setMinimumHeight(48)
        self.overview.getAxis("left").setStyle(showValues=False)
        self.overview.getAxis("left").setPen(pg.mkPen(None))
        for p in [self.detail, self.overview] + self.strip:
            p.getAxis("left").setWidth(AXIS_WIDTH_PX)

        self.trace = pg.PlotDataItem(pen=pg.mkPen(width=1))
        self.trace.setDownsampling(auto=True, method="peak")
        self.trace.setClipToView(True)
        self.detail.addItem(self.trace)
        self.over_trace = pg.PlotDataItem(pen=pg.mkPen(width=1))
        self.over_trace.setDownsampling(auto=True, method="peak")
        self.overview.addItem(self.over_trace)
        self.hr_over = pg.PlotDataItem(pen=pg.mkPen(width=1.5))
        self.hr_over.setDownsampling(auto=True, method="peak")
        self.overview.addItem(self.hr_over)
        self._over_regions: list = []

        self.accepted = pg.ScatterPlotItem(size=7, symbol="o")
        self.rejected = pg.ScatterPlotItem(size=5, symbol="o")
        self.excluded = pg.ScatterPlotItem(size=9, symbol="x")
        self.added = pg.ScatterPlotItem(size=8, symbol="t")
        # Check agreement: accepted beats no other detector found (open ring).
        self.unconfirmed = pg.ScatterPlotItem(size=16, symbol="o")
        for item in (self.rejected, self.accepted, self.unconfirmed, self.excluded, self.added):
            self.detail.addItem(item)

        self.band = pg.LinearRegionItem(values=(0, DEFAULT_WINDOW_S), movable=True)
        self.overview.addItem(self.band)
        self.band.sigRegionChanged.connect(self._band_moved)
        self.detail.sigXRangeChanged.connect(self._detail_moved)
        # The vertical fits (trace and strip) follow the view at most every
        # FIT_DELAY_MS while it moves: each one is a full repaint.
        self._fit_timer = QtCore.QTimer(self, singleShot=True, interval=FIT_DELAY_MS)
        self._fit_timer.timeout.connect(self._fit_now)
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
        self._show_overview_hr(False)
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

    def set_strip_visible(self, on: bool) -> None:
        """Show / hide the RR / HR / Quality lanes."""
        self._strip_on = bool(on)
        self._layout_strip()

    def _layout_strip(self) -> None:
        """The lanes shown (Quality gives way in a short window, so the trace
        keeps its height) and the time axis at the bottom of the stack: the
        trace's, or the last lane's."""
        on = getattr(self, "_strip_on", True)
        compact = self.height() < COMPACT_HEIGHT_PX
        shown = [on, on, on and not compact]
        for p, vis in zip(self.strip, shown):
            p.setVisible(vis)
            # A hidden item keeps its place in a graphics layout: none then.
            p.setMinimumHeight(STRIP_MIN_PX if vis else 0)
            p.setMaximumHeight(STRIP_MAX_PX if vis else 0)
            p.hideAxis("bottom")
        lanes = [p for p, vis in zip(self.strip, shown) if vis]
        if lanes:
            self.detail.hideAxis("bottom")
            lanes[-1].showAxis("bottom")
            lanes[-1].setLabel("bottom", "Time", units="s")
        else:
            self.detail.showAxis("bottom")

    def resizeEvent(self, ev: QtGui.QResizeEvent) -> None:
        super().resizeEvent(ev)
        if (ev.size().height() < COMPACT_HEIGHT_PX) != (ev.oldSize().height() < COMPACT_HEIGHT_PX):
            self._layout_strip()

    def set_beats(self, peaks: "Optional[np.ndarray]", quality_t: "Optional[np.ndarray]" = None,
                  quality: "Optional[np.ndarray]" = None) -> None:
        """The strip: RR and HR of *peaks* (sample indices), and the beat
        quality *quality* at the times *quality_t* (s)."""
        if self._signal is None or peaks is None or len(peaks) < 2:
            for c in (self.rr_curve, self.hr_curve, self.q_curve, self.hr_over):
                c.setData([], [])
            self._rr = None
            self._show_overview_hr(False)
            return
        rp = np.asarray(peaks, dtype=float)
        rr_ms = np.diff(rp) / self._fs * 1000.0
        rr_t = rp[1:] / self._fs
        self._rr = (rr_t, rr_ms, 60000.0 / rr_ms)
        self.rr_curve.setData(rr_t, rr_ms)
        self.hr_curve.setData(rr_t, self._rr[2])
        self.hr_over.setData(rr_t, _running_median(self._rr[2], HR_SMOOTH_BEATS))
        self._show_overview_hr(True)
        if quality_t is not None and quality is not None and len(quality):
            self.q_curve.setData(np.asarray(quality_t, float), np.asarray(quality, float))
        else:
            self.q_curve.setData([], [])
        self._fit_strip()

    def _show_overview_hr(self, on: bool) -> None:
        """The overview shows the heart rate of every beat (on), else the
        signal; its heart-rate scale leaves out the rare extreme beats (1st
        to 99th percentile) so the trend is not flattened by one of them."""
        self.over_trace.setVisible(not on)
        self.hr_over.setVisible(on)
        axis = self.overview.getAxis("left")
        axis.setStyle(showValues=on)
        axis.setTickDensity(0.5)
        if on and self._rr is not None:
            hr = self._rr[2][np.isfinite(self._rr[2])]
            if len(hr):
                lo, hi = (float(v) for v in np.percentile(hr, [1, 99]))
                pad = max(0.08 * (hi - lo), 5.0)
                self.overview.setYRange(lo - pad, hi + pad, padding=0)
        elif self._signal is not None and len(self._signal):
            self.overview.enableAutoRange(axis="y")

    def hr_stats(self) -> "Optional[tuple[float, float, float]]":
        """The heart rate of the beats in view: (median, min, max), bpm."""
        if self._rr is None:
            return None
        t0, t1 = self.window()
        rr_t, _rr, hr = self._rr
        v = hr[(rr_t >= t0) & (rr_t <= t1) & np.isfinite(hr)]
        if not len(v):
            return None
        return float(np.median(v)), float(np.min(v)), float(np.max(v))

    def mean_hr(self) -> "Optional[float]":
        """The mean of the instantaneous heart rate over the recording."""
        if self._rr is None:
            return None
        hr = self._rr[2][np.isfinite(self._rr[2])]
        return float(np.mean(hr)) if len(hr) else None

    def _fit_strip(self) -> None:
        """RR and HR centred on the mean of what is in view (centered_ylim)."""
        if self._rr is None:
            return
        t0, t1 = self.window()
        rr_t, rr_ms, hr = self._rr
        inside = (rr_t >= t0) & (rr_t <= t1)
        for p, vals in ((self.strip[0], rr_ms[inside]), (self.strip[1], hr[inside])):
            v = vals[np.isfinite(vals)]
            if not len(v):
                continue
            c = float(np.mean(v))
            half = float(np.max(np.abs(v - c))) or max(abs(c) * 0.05, 1.0)
            half *= 1.08
            p.setYRange(c - half, c + half, padding=0)
        # Quality (template correlation, up to 1): from a little under the
        # lowest beat in view, at most 0.85, so the 0.90 line and the beats
        # near 1 are both readable.
        q_t, q = self.q_curve.getData()
        lo = 0.85
        if q_t is not None and len(q_t):
            inside = (q_t >= t0) & (q_t <= t1) & np.isfinite(q)
            if inside.any():
                lo = min(lo, float(np.min(q[inside])) - 0.03)
        self.strip[2].setYRange(max(-1.0, lo), 1.01, padding=0)

    def set_unconfirmed(self, peaks: "Optional[np.ndarray]") -> None:
        """Circle the beats Check agreement found unconfirmed (sample indices)."""
        if self._signal is None or peaks is None or len(peaks) == 0:
            self.unconfirmed.setData([], [])
            return
        idx = np.asarray(peaks, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < len(self._signal))]
        self.unconfirmed.setData(idx / self._fs, self._signal[idx])

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
        for item in self._over_regions:
            self.overview.removeItem(item)
        self._over_regions = []
        for a in annotations or []:
            t0, t1 = float(a["t_start"]), float(a["t_end"])
            color = a.get("color") or "#888888"
            over = pg.LinearRegionItem(values=(t0, t1), movable=False,
                                       brush=pg.mkBrush(_alpha(color, 45)),
                                       pen=pg.mkPen(None))
            over.setZValue(-20)
            self.overview.addItem(over)
            self._over_regions.append(over)
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
        self.set_unconfirmed(None)
        self.set_beats(None)
        self.set_noise_regions([])

    # ── window ──────────────────────────────────────────────
    def window(self) -> "tuple[float, float]":
        (t0, t1), _ = self.detail.viewRange()
        return float(t0), float(t1)

    def set_window(self, t0: float, t1: float) -> None:
        """Show *t0*-*t1* (s), fitted at once (a jump, not a drag)."""
        if t1 - t0 < MIN_WINDOW_S:
            t1 = t0 + MIN_WINDOW_S
        self.detail.setXRange(t0, t1, padding=0)
        self.flush_fit()

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
        if not self._fit_timer.isActive():
            self._fit_timer.start()
        self.window_changed.emit(t0, t1)

    def _fit_now(self) -> None:
        """The vertical ranges for what is in view: the trace, then the strip."""
        t0, t1 = self.window()
        self._fit_y(t0, t1)
        self._fit_strip()

    def flush_fit(self) -> None:
        """Fit now what a move of the view left to fit (tests, exports)."""
        if self._fit_timer.isActive():
            self._fit_timer.stop()
        self._fit_now()

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
        self.over_view.setBackground(p.PANEL)
        self.trace.setPen(pg.mkPen(p.plot_signal, width=1.2))
        self.over_trace.setPen(pg.mkPen(p.LIGHT, width=1))
        self.hr_over.setPen(pg.mkPen(p.ACCENT, width=1.5))
        # R peaks in red; a peak excluded by hand is a cross in ink, one added
        # by hand a triangle in blue.
        self.accepted.setBrush(pg.mkBrush(p.plot_rpeak))
        self.accepted.setPen(pg.mkPen(None))
        self.rejected.setBrush(pg.mkBrush(p.LIGHT))
        self.rejected.setPen(pg.mkPen(None))
        self.excluded.setBrush(pg.mkBrush(p.TEXT))
        self.excluded.setPen(pg.mkPen(p.TEXT, width=2))
        self.added.setBrush(pg.mkBrush(p.BLUE))
        self.added.setPen(pg.mkPen(None))
        self.unconfirmed.setBrush(pg.mkBrush(None))
        self.unconfirmed.setPen(pg.mkPen("#FF9800", width=1.8))   # AMBER, as the Tk ring
        for plot in self.strip:
            for ax in ("left", "bottom"):
                plot.getAxis(ax).setPen(pg.mkPen(p.MUTED))
                plot.getAxis(ax).setTextPen(pg.mkPen(p.MUTED))
        for c, col in ((self.rr_curve, p.MUTED), (self.hr_curve, p.ACCENT),
                       (self.q_curve, p.GREEN)):
            c.setPen(pg.mkPen(col, width=1.2))
            c.setSymbolBrush(pg.mkBrush(col))
            c.setSymbolPen(pg.mkPen(None))
        self.q_line.setPen(pg.mkPen(p.MUTED, width=0.8, style=QtCore.Qt.PenStyle.DashLine))
        self.band.setBrush(pg.mkBrush(_alpha(p.TEXT, 28)))
        for line in self.band.lines:
            line.setPen(pg.mkPen(p.TEXT, width=2))
