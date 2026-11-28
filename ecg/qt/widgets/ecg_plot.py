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
the caller, which also places the edit preview (show_edit_preview) from the
pointer position it is given (``hovered``).

Zoom and navigation as the Tk detail view (its Matplotlib toolbar and wheel):
the wheel zooms the time axis around the pointer (0.8 or 1.25 times the
window per notch); under the trace, Home / Back / Forward, Pan (drag: both
axes; right drag: stretch), Zoom (drag a box: zoom in, right button: zoom
out; a flat box zooms time only, a narrow one the height only) and Save,
with the pointer's coordinates. Without a tool, a drag moves the time
window.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ecg.qt import theme
from ecg.qt.widgets.rhythm_charts import few_ticks

# Two seconds: about 16 beats of a mouse at 500 bpm, each QRS readable (ten
# seconds packed some 85 beats into the view).
DEFAULT_WINDOW_S = 2.0
MIN_WINDOW_S = 0.1          # as the Tk views (ecg.app.state.MIN_VIEW_WINDOW_S)
AXIS_WIDTH_PX = 52          # every lane's value axis: their time axes line up
TRACE_MIN_HEIGHT_PX = 170   # the trace keeps this much; the strip gives way first
FIT_DELAY_MS = 40           # vertical fits while panning: at most ~25 per second
COMPACT_HEIGHT_PX = 460     # below this, the strip drops its Quality lane
OVERVIEW_HEIGHT_PX = 96
STRIP_MIN_PX, STRIP_MAX_PX = 56, 90   # each RR / HR / Quality lane
HR_SMOOTH_BEATS = 9                   # the overview's heart rate: a running median
WHEEL_STEP = 0.8            # one wheel notch: a window 0.8 times as long (or 1.25), as Tk
ZOOM_MIN_PX = 5             # a zoom box thinner than this on one axis keeps that axis
HOVER_DELAY_MS = 30         # the edit preview follows the pointer at most ~33 times a second
# Tk's colours for the edit marks: the peaks added by hand (a cyan star, teal
# edge), the preview of a right click (cyan: adds a peak; orange: replaces or
# removes one).
ADD_COLOR, ADD_EDGE = "#00BCD4", "#00695C"
CHANGE_COLOR = "#FF9500"
UNCONFIRMED_COLOR = "#FF9800"
_ALPHA_LEVELS = 13          # confidence shades of the accepted peaks (0.35 to 0.95)


def nav_icon(name: str, dark: bool, color: str = "#F5F5F7") -> QtGui.QIcon:
    """Matplotlib's toolbar icon *name* (the Tk toolbar's), recoloured to
    *color* on a dark theme as Matplotlib's own Qt toolbar does."""
    from pathlib import Path
    import matplotlib
    base = Path(matplotlib.get_data_path()) / "images"
    path = base / f"{name}_large.png"
    if not path.exists():
        path = base / f"{name}.png"
    pm = QtGui.QPixmap(str(path))
    if dark and not pm.isNull():
        mask = pm.createMaskFromColor(QtGui.QColor("black"), QtCore.Qt.MaskMode.MaskOutColor)
        pm.fill(QtGui.QColor(color))
        pm.setMask(mask)
    return QtGui.QIcon(pm)


def _zoom_out(v0: float, v1: float, b0: float, b1: float) -> "tuple[float, float]":
    """Matplotlib's zoom out to a box: the view [v0, v1] shrinks into the
    box [b0, b1]."""
    if b1 <= b0:
        return v0, v1
    k = (v1 - v0) / (b1 - b0)
    return v0 - k * (b0 - v0), v1 + k * (v1 - b1)


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
    hovered = QtCore.Signal(float)                  # the pointer's time on the trace (s); NaN: left it
    nav_mode_changed = QtCore.Signal(str)           # "pan", "zoom" or "" (none)

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
        self._build_navbar()
        lay.addWidget(self.navbar)
        self.view.ci.setContentsMargins(6, 8, 10, 2)
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
        # Short titles, small: a lane is ~60 px high and the full ones ran
        # into each other.
        for r, label in ((1, "RR ms"), (2, "HR bpm"), (3, "Corr.")):
            p = self.view.addPlot(row=r, col=0)
            p.setXLink(self.detail)
            p.setMouseEnabled(x=False, y=False)
            p.setMenuEnabled(False)
            p.setLabel("left", label, **{"font-size": "8pt"})
            p.hideAxis("bottom")
            p.getAxis("left").setTickDensity(0.35)
            p.hideButtons()
            p.setMinimumHeight(STRIP_MIN_PX)
            p.setMaximumHeight(STRIP_MAX_PX)
            self.view.ci.layout.setRowStretchFactor(r, 1)
            self.strip.append(p)
        self.rr_curve, self.hr_curve, self.q_curve = (
            pg.PlotDataItem(symbol="o", symbolSize=3) for _ in range(3))
        for p, c in zip(self.strip, (self.rr_curve, self.hr_curve, self.q_curve)):
            p.addItem(c)
            # On the plot, not the curve: PlotItem.addItem gives each curve
            # the plot's own (default: off) clip and downsampling settings.
            p.setClipToView(True)
        self.q_line = pg.InfiniteLine(pos=0.90, angle=0, movable=False)  # the 0.90 corr. line
        self.strip[2].addItem(self.q_line)
        self._rr = None
        # Grid: the time lines of the trace only. A grid is drawn across the
        # whole plot with transparency on every repaint; on all five plots it
        # took more than half of the time of a move of the view.
        # No grid: drawing it on every move was a third of the time of a
        # drag (the time ticks and the window presets carry the scale).
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

        # Only what is in view, at most ~one point per pixel column (peak
        # keeps the R waves). Set on the plots: PlotItem.addItem overwrites a
        # curve's own clip / downsampling with the plot's (off by default),
        # so the trace used to rebuild every sample of the recording on each
        # move (an hour at 2 kHz: ~1 s per step).
        self.trace = pg.PlotDataItem(pen=pg.mkPen(width=1))
        self.detail.addItem(self.trace)
        self.detail.setClipToView(True)
        self.detail.setDownsampling(auto=True, mode="peak")
        self.over_trace = pg.PlotDataItem(pen=pg.mkPen(width=1))
        self.overview.addItem(self.over_trace)
        self.hr_over = pg.PlotDataItem(pen=pg.mkPen(width=1.5))
        self.overview.addItem(self.hr_over)
        self.overview.setDownsampling(auto=True, mode="peak")
        self._over_regions: list = []

        # The Tk marks: accepted peaks (paler: a lower detection confidence),
        # rejected candidates, a cross on a peak excluded by hand, a star on
        # one added by hand, a ring on a beat no other detector found, and a
        # down triangle on a beat removed by the artifact review (its type's
        # colour).
        self.accepted = pg.ScatterPlotItem(size=8, symbol="o")
        self.rejected = pg.ScatterPlotItem(size=6, symbol="o")
        self.excluded = pg.ScatterPlotItem(size=11, symbol="x")
        self.added = pg.ScatterPlotItem(size=15, symbol="star")
        # Check agreement: accepted beats no other detector found (open ring).
        self.unconfirmed = pg.ScatterPlotItem(size=16, symbol="o")
        self.artifact_marks: "dict[str, pg.ScatterPlotItem]" = {}
        for kind in ("nonphysio", "ectopic", "duplicate"):
            self.artifact_marks[kind] = pg.ScatterPlotItem(size=8, symbol="t")
        self.threshold_line = pg.InfiniteLine(angle=0, movable=False)
        self.threshold_line.setVisible(False)
        self.detail.addItem(self.threshold_line, ignoreBounds=True)
        for item in (self.rejected, *self.artifact_marks.values(), self.accepted,
                     self.unconfirmed, self.excluded, self.added):
            self.detail.addItem(item)
        z = {self.rejected: 4, self.accepted: 5, self.unconfirmed: 5.5, self.excluded: 6,
             self.added: 7}
        for item, zv in z.items():
            item.setZValue(zv)
        for item in self.artifact_marks.values():
            item.setZValue(4.5)
        self._layers: "dict" = {}
        self._acc_brushes: "list" = []
        # Edit preview (Tk's hover marker): where a right click puts a peak.
        self.pv_line = pg.InfiniteLine(angle=90, movable=False)
        self.pv_point = pg.ScatterPlotItem(size=13, symbol="d")
        self.pv_text = pg.TextItem("", anchor=(0.5, 1.9))
        f = QtGui.QFont(self.font())
        f.setBold(True)
        f.setPointSizeF(max(7.0, f.pointSizeF() - 1))
        self.pv_text.setFont(f)
        for item, zv in ((self.pv_line, 8), (self.pv_point, 10), (self.pv_text, 11)):
            item.setZValue(zv)
            item.setVisible(False)
            self.detail.addItem(item, ignoreBounds=True)

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
        self.detail.scene().sigMouseMoved.connect(self._mouse_moved)
        self._hover_t = float("nan")
        self._hover_timer = QtCore.QTimer(self, singleShot=True, interval=HOVER_DELAY_MS)
        self._hover_timer.timeout.connect(lambda: self.hovered.emit(self._hover_t))
        # Span mode (Mark Noise): a left drag draws a span instead of panning.
        self.span_mode = False
        self._span: "Optional[pg.LinearRegionItem]" = None
        # The zoom tools: "" (a drag moves the time window), "pan", "zoom";
        # a height set by hand (box zoom, pan) is kept until the next move
        # of the time window; the views of the tools, for Back / Forward.
        self._nav_mode = ""
        self._y_manual = False
        self._hist: "list" = [((0.0, DEFAULT_WINDOW_S), None)]
        self._hist_i = 0
        vb = self.detail.getViewBox()
        self._pan_drag = vb.mouseDragEvent
        vb.mouseDragEvent = self._drag
        vb.wheelEvent = self._wheel
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
        end = self._end()
        self.overview.setXRange(0, end, padding=0)
        self.set_window(0.0, min(DEFAULT_WINDOW_S, end))

    def _end(self) -> float:
        """The recording's last time (s)."""
        return float(self._time[-1]) if self._time is not None and len(self._time) else 0.0

    def _layer(self, peaks: "Optional[np.ndarray]", alpha: "Optional[np.ndarray]" = None):
        """(times, amplitudes, shade levels or None) of *peaks* (sample
        indices), sorted, inside the signal."""
        if self._signal is None or peaks is None or len(peaks) == 0:
            return (np.empty(0), np.empty(0), None)
        idx = np.asarray(peaks, dtype=np.int64)
        order = np.argsort(idx, kind="stable")
        idx = idx[order]
        keep = (idx >= 0) & (idx < len(self._signal))
        levels = None
        if alpha is not None and len(alpha) == len(order):
            a = np.asarray(alpha, dtype=float)[order][keep]
            levels = np.clip(np.round((np.nan_to_num(a, nan=0.95) - 0.35) / 0.05),
                             0, _ALPHA_LEVELS - 1).astype(int)
        idx = idx[keep]
        return (idx / self._fs, self._signal[idx], levels)

    def set_peaks(self, accepted: "Optional[np.ndarray]" = None,
                  rejected: "Optional[np.ndarray]" = None,
                  excluded: "Optional[np.ndarray]" = None,
                  added: "Optional[np.ndarray]" = None,
                  accepted_alpha: "Optional[np.ndarray]" = None) -> None:
        """Mark the peaks (sample indices) on the detail trace;
        *accepted_alpha*: the opacity of each accepted peak (its detection
        confidence, 0.35 to 0.95, as Tk), else all alike."""
        self._layers[self.accepted] = self._layer(accepted, accepted_alpha)
        for item, peaks in ((self.rejected, rejected), (self.excluded, excluded),
                            (self.added, added)):
            self._layers[item] = self._layer(peaks)
        self._clip_peaks(force=True)

    def set_artifacts(self, by_type: "Optional[dict]" = None) -> None:
        """The beats removed by the artifact review, by type
        ({"nonphysio" / "ectopic" / "duplicate": sample indices})."""
        for kind, item in self.artifact_marks.items():
            self._layers[item] = self._layer((by_type or {}).get(kind))
        self._clip_peaks(force=True)

    def set_threshold(self, amp: "Optional[float]") -> None:
        """The detection threshold (signal units), dashed across the trace;
        None: hidden."""
        ok = amp is not None and np.isfinite(amp)
        self.threshold_line.setVisible(bool(ok))
        if ok:
            self.threshold_line.setPos(float(amp))

    def _clip_peaks(self, force: bool = False) -> None:
        """Give each marker layer only the peaks around the view (one window
        width on each side): a scatter is not clipped to the view, and an
        hour of beats (~36 000) was masked and painted on every move. Redone
        only when the view leaves that span."""
        if not self._layers:
            return
        t0, t1 = self.window()
        lo, hi = getattr(self, "_peak_span", (0.0, -1.0))
        if not force and lo <= t0 and t1 <= hi:
            return
        span = max(t1 - t0, 1e-6)
        a, b = t0 - span, t1 + span
        for item, (x, y, levels) in self._layers.items():
            i0, i1 = np.searchsorted(x, a), np.searchsorted(x, b)
            if levels is not None and self._acc_brushes:
                item.setData(x[i0:i1], y[i0:i1],
                             brush=[self._acc_brushes[k] for k in levels[i0:i1]])
            else:
                item.setData(x[i0:i1], y[i0:i1])
        self._peak_span = (a, b)

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
        # No room between hidden lanes either (each kept its spacing: an empty
        # band under the trace), and only a thin margin under the stack.
        grid = self.view.ci.layout
        for r, vis in enumerate(shown):
            grid.setRowSpacing(r, 6 if vis else 0)
        grid.setRowSpacing(len(shown), 0)
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
                few_ticks(axis, lo - pad, hi + pad, 2)      # a short chart: 2-3 labels
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
            few_ticks(p.getAxis("left"), c - half, c + half, 2)
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
        self.set_artifacts()
        self.set_threshold(None)
        self.show_edit_preview(None)
        self.set_unconfirmed(None)
        self.set_beats(None)
        self.set_noise_regions([])

    # ── window ──────────────────────────────────────────────
    def window(self) -> "tuple[float, float]":
        (t0, t1), _ = self.detail.viewRange()
        return float(t0), float(t1)

    def set_window(self, t0: float, t1: float) -> None:
        """Show *t0*-*t1* (s), fitted at once (a jump, not a drag): a move of
        the time window, which Home then comes back to."""
        if t1 - t0 < MIN_WINDOW_S:
            t1 = t0 + MIN_WINDOW_S
        self._y_manual = False
        self.detail.setXRange(t0, t1, padding=0)
        self.flush_fit()
        self._reset_history()

    def _band_moved(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        t0, t1 = self.band.getRegion()
        self._y_manual = False
        self.detail.setXRange(t0, t1, padding=0)
        self._syncing = False
        self._reset_history()

    # ── views of the zoom tools (Home / Back / Forward) ─────
    def _reset_history(self) -> None:
        """The time window just moved to is the view Home comes back to;
        the earlier tool views are dropped (as Tk, which redrew the axes)."""
        t0, t1 = self.window()
        self._hist = [((t0, t1), None)]
        self._hist_i = 0
        self._sync_nav_actions()

    def _push_view(self) -> None:
        """Keep the view a tool just made, for Back / Forward."""
        (x0, x1), (y0, y1) = self.detail.viewRange()
        del self._hist[self._hist_i + 1:]
        self._hist.append(((x0, x1), (y0, y1) if self._y_manual else None))
        self._hist_i = len(self._hist) - 1
        self._sync_nav_actions()

    def _apply_view(self, view) -> None:
        (x0, x1), y = view
        self._y_manual = y is not None
        if y is not None:
            self.detail.setRange(xRange=(x0, x1), yRange=y, padding=0)
        else:
            self.detail.setXRange(x0, x1, padding=0)
        self.flush_fit()
        self._sync_nav_actions()

    def go_home(self) -> None:
        """Back to the time window as navigated, its height fitted."""
        self._hist_i = 0
        self._apply_view(self._hist[0])

    def go_back(self) -> None:
        if self._hist_i > 0:
            self._hist_i -= 1
            self._apply_view(self._hist[self._hist_i])

    def go_forward(self) -> None:
        if self._hist_i < len(self._hist) - 1:
            self._hist_i += 1
            self._apply_view(self._hist[self._hist_i])

    def _sync_nav_actions(self) -> None:
        if hasattr(self, "act_back"):
            self.act_back.setEnabled(self._hist_i > 0)
            self.act_forward.setEnabled(self._hist_i < len(self._hist) - 1)

    def _detail_moved(self, *_a) -> None:
        self._clip_peaks()
        t0, t1 = self.window()
        if not self._syncing:
            self._syncing = True
            self.band.setRegion((t0, t1))
            self._syncing = False
        if not self._fit_timer.isActive():
            self._fit_timer.start()
        self.window_changed.emit(t0, t1)

    def _fit_now(self) -> None:
        """The vertical ranges for what is in view: the trace (unless its
        height was set by a zoom tool), then the strip."""
        t0, t1 = self.window()
        if not self._y_manual:
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
        left = ev.button() == QtCore.Qt.MouseButton.LeftButton
        right = ev.button() == QtCore.Qt.MouseButton.RightButton
        if self.span_mode and left:
            self._span_drag(ev)
        elif self._nav_mode == "zoom" and (left or right):
            self._zoom_drag(ev, out=right)
        elif self._nav_mode == "pan":
            # Matplotlib's pan: a drag moves both axes, a right drag
            # stretches them; the height is then the one set by hand.
            if ev.isStart():
                self._y_manual = True
            self._pan_drag(ev, axis)
            if ev.isFinish():
                self._push_view()
        elif left:
            # No tool: a drag moves the time window (the height follows).
            if ev.isStart():
                self._y_manual = False
            self._pan_drag(ev, axis)
            if ev.isFinish():
                self._reset_history()
        else:
            ev.accept()                       # a right drag does nothing (a click edits)

    def _zoom_drag(self, ev, out: bool) -> None:
        """Matplotlib's zoom to a box: drawn while dragging; on release, the
        view becomes the box (left button) or shrinks into it (right). A box
        under ZOOM_MIN_PX high zooms the time only, one under it wide the
        height only."""
        ev.accept()
        vb = self.detail.getViewBox()
        p0, p1 = ev.buttonDownPos(), ev.pos()
        if not ev.isFinish():
            vb.updateScaleBox(p0, p1)
            return
        vb.rbScaleBox.hide()
        dx, dy = abs(p1.x() - p0.x()), abs(p1.y() - p0.y())
        if dx < ZOOM_MIN_PX and dy < ZOOM_MIN_PX:
            return
        a, b = vb.mapToView(p0), vb.mapToView(p1)
        bx0, bx1 = sorted((float(a.x()), float(b.x())))
        by0, by1 = sorted((float(a.y()), float(b.y())))
        (vx0, vx1), (vy0, vy1) = vb.viewRange()
        if dx < ZOOM_MIN_PX:
            bx0, bx1 = vx0, vx1
        if dy < ZOOM_MIN_PX:
            by0, by1 = vy0, vy1
        if out:
            bx0, bx1 = _zoom_out(vx0, vx1, bx0, bx1) if dx >= ZOOM_MIN_PX else (vx0, vx1)
            by0, by1 = _zoom_out(vy0, vy1, by0, by1) if dy >= ZOOM_MIN_PX else (vy0, vy1)
        end = self._end()
        width = min(max(bx1 - bx0, MIN_WINDOW_S), max(end, MIN_WINDOW_S))
        bx0 = max(0.0, min(bx0, end - width))
        if dy >= ZOOM_MIN_PX:
            self._y_manual = True
            self.detail.setRange(xRange=(bx0, bx0 + width), yRange=(by0, by1), padding=0)
        else:
            self.detail.setXRange(bx0, bx0 + width, padding=0)
        self._push_view()

    def _wheel(self, ev, axis=None) -> None:
        """Tk's wheel zoom: the time window 0.8 times as long per notch up
        (1.25 times down), the time under the pointer kept in place, within
        the recording; the height is fitted again."""
        ev.accept()
        if self._signal is None or not len(self._signal):
            return
        delta = ev.delta()
        if not delta or ev.orientation() == QtCore.Qt.Orientation.Horizontal:
            return
        t0, t1 = self.window()
        cur = max(t1 - t0, 1e-9)
        end = self._end()
        new = max(MIN_WINDOW_S, min(max(end, MIN_WINDOW_S), cur * WHEEL_STEP ** (delta / 120.0)))
        x = float(self.detail.getViewBox().mapToView(ev.pos()).x())
        frac = min(1.0, max(0.0, (x - t0) / cur))
        start = max(0.0, min(x - frac * new, max(0.0, end - new)))
        self._y_manual = False
        self.detail.setXRange(start, start + new, padding=0)
        self._reset_history()

    def _span_drag(self, ev) -> None:
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

    def _mouse_moved(self, pos) -> None:
        """The pointer's coordinates under the trace, and its time for the
        edit preview (NaN once it leaves the trace)."""
        inside = (self._signal is not None
                  and self.detail.getViewBox().sceneBoundingRect().contains(pos))
        if inside:
            pt = self.detail.getViewBox().mapSceneToView(pos)
            t, y = float(pt.x()), float(pt.y())
            self.lbl_coords.setText(f"t = {t:,.4f} s    y = {y:.3f}")
        else:
            t = float("nan")
            self.lbl_coords.setText("")
        if not (np.isnan(t) and np.isnan(self._hover_t)):
            self._hover_t = t
            if not self._hover_timer.isActive():
                self._hover_timer.start()

    def leaveEvent(self, ev: QtCore.QEvent) -> None:  # noqa: N802 (Qt name)
        super().leaveEvent(ev)
        self._mouse_moved(QtCore.QPointF(-1e9, -1e9))

    def show_edit_preview(self, sample: "Optional[int]", kind: str = "add") -> None:
        """Tk's edit preview: a dashed line and a diamond where a right click
        puts the peak (*sample*), its time above; cyan when it adds one
        ("add", "add_free"), orange when it replaces or removes one. None:
        hidden."""
        items = (self.pv_line, self.pv_point, self.pv_text)
        if sample is None or self._signal is None or not 0 <= int(sample) < len(self._signal):
            for item in items:
                item.setVisible(False)
            return
        t = int(sample) / self._fs
        y = float(self._signal[int(sample)])
        color = ADD_COLOR if kind in ("add", "add_free") else CHANGE_COLOR
        self.pv_line.setPen(pg.mkPen(_alpha(color, 166), width=1,
                                     style=QtCore.Qt.PenStyle.DashLine))
        self.pv_line.setPos(t)
        self.pv_point.setData([t], [y], brush=pg.mkBrush(_alpha(color, 204)),
                              pen=pg.mkPen("#FFFFFF", width=1.4))
        p = theme.current()
        # Its time above it, or under it when the peak is near the top of
        # the view (the label would be cut).
        (_x0, _x1), (y0, y1) = self.detail.viewRange()
        near_top = (y1 - y) < 0.15 * max(y1 - y0, 1e-12)
        self.pv_text.setAnchor((0.5, -0.9) if near_top else (0.5, 1.9))
        self.pv_text.setText(f"{t:.3f} s", color=color)
        self.pv_text.fill = pg.mkBrush(_alpha(p.PANEL, 217))
        self.pv_text.border = pg.mkPen(color, width=0.8)
        self.pv_text.setPos(t, y)
        self.pv_text.update()
        for item in items:
            item.setVisible(True)

    # ── the zoom tools (Tk's Matplotlib toolbar) ────────────
    def _build_navbar(self) -> None:
        bar = self.navbar = QtWidgets.QToolBar()
        bar.setIconSize(QtCore.QSize(18, 18))
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.act_home = bar.addAction("Home", self.go_home)
        self.act_home.setToolTip("Home: back to the time window, its height fitted")
        self.act_back = bar.addAction("Back", self.go_back)
        self.act_back.setToolTip("Back to the previous view")
        self.act_forward = bar.addAction("Forward", self.go_forward)
        self.act_forward.setToolTip("Forward to the next view")
        bar.addSeparator()
        self.act_pan = bar.addAction("Pan")
        self.act_pan.setCheckable(True)
        self.act_pan.setToolTip("Pan: drag to move the view (both axes), right drag to "
                                "stretch it")
        self.act_pan.toggled.connect(lambda on: self._nav_toggled("pan", on))
        self.act_zoom = bar.addAction("Zoom")
        self.act_zoom.setCheckable(True)
        self.act_zoom.setToolTip("Zoom to rectangle: drag a box to zoom in, with the right "
                                 "button to zoom out (a flat box: time only, a narrow one: "
                                 "height only)")
        self.act_zoom.toggled.connect(lambda on: self._nav_toggled("zoom", on))
        bar.addSeparator()
        self.act_save = bar.addAction("Save", self.save_image)
        self.act_save.setToolTip("Save the view as an image (PNG or SVG)")
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                             QtWidgets.QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)
        # A fixed size: its new text on every move of the pointer does not
        # lay the window out again.
        self.lbl_coords = QtWidgets.QLabel("")
        self.lbl_coords.setProperty("muted", True)
        self.lbl_coords.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                                     | QtCore.Qt.AlignmentFlag.AlignVCenter)
        fm = self.lbl_coords.fontMetrics()
        self.lbl_coords.setFixedSize(fm.horizontalAdvance("t = 00,000.0000 s    y = -00.000") + 16,
                                     fm.height() + 4)
        self._coords_action = bar.addWidget(self.lbl_coords)
        self.setToolTip("")
        self.view.setToolTip("Wheel: zoom the time · drag: move the time window · "
                             "Pan / Zoom below for both axes")

    def add_to_navbar(self, widget: QtWidgets.QWidget) -> None:
        """A page's widget on the bar under the trace, before the pointer's
        coordinates (e.g. the legend of the marks, as Tk's in the plot)."""
        self.navbar.insertWidget(self._coords_action, widget)

    def _nav_toggled(self, mode: str, on: bool) -> None:
        if on:
            self.set_nav_mode(mode)
        elif self._nav_mode == mode:
            self.set_nav_mode("")

    def nav_mode(self) -> str:
        return self._nav_mode

    def set_nav_mode(self, mode: str) -> None:
        """The zoom tool in use: "pan", "zoom" or "" (none: a drag moves the
        time window). The page turns its own tools off when one is chosen
        (nav_mode_changed)."""
        mode = mode if mode in ("pan", "zoom") else ""
        changed = mode != self._nav_mode
        self._nav_mode = mode
        for act, m in ((self.act_pan, "pan"), (self.act_zoom, "zoom")):
            act.blockSignals(True)
            act.setChecked(mode == m)
            act.blockSignals(False)
        self.detail.setMouseEnabled(x=True, y=(mode == "pan"))
        shape = {"pan": QtCore.Qt.CursorShape.SizeAllCursor,
                 "zoom": QtCore.Qt.CursorShape.CrossCursor}.get(mode)
        if shape is None:
            self.view.unsetCursor()
        else:
            self.view.setCursor(shape)
        if changed:
            self.nav_mode_changed.emit(mode)

    def save_image(self, path: "Optional[str]" = None) -> "Optional[str]":
        """The trace (and the strip under it) as shown, to a PNG or SVG file
        (asked for when *path* is None); the path written."""
        import pyqtgraph.exporters as exporters
        if path is None:
            t0, t1 = self.window()
            path, _f = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save the view", f"signal_{t0:.1f}-{t1:.1f}s.png",
                "PNG image (*.png);;SVG image (*.svg)")
            if not path:
                return None
        if path.lower().endswith(".svg"):
            exp = exporters.SVGExporter(self.view.scene())
        else:
            exp = exporters.ImageExporter(self.view.scene())
            exp.parameters()["width"] = int(self.view.width() * 2)
        exp.export(path)
        return path

    # ── look ────────────────────────────────────────────────
    def apply_palette(self, p: "theme.Palette") -> None:
        self.view.setBackground(p.PANEL)
        for plot in (self.detail, self.overview):
            for ax in ("left", "bottom"):
                plot.getAxis(ax).setPen(pg.mkPen(p.MUTED))
                plot.getAxis(ax).setTextPen(pg.mkPen(p.MUTED))
        self.over_view.setBackground(p.PANEL)
        # Pens one pixel wide: a wider pen is drawn segment by segment and
        # made every move of the view three to four times slower.
        self.trace.setPen(pg.mkPen(p.plot_signal, width=1))
        self.over_trace.setPen(pg.mkPen(p.LIGHT, width=1))
        self.hr_over.setPen(pg.mkPen(p.ACCENT, width=1))
        # The Tk marks: accepted peaks (the theme's R colour, paler when the
        # detection was less sure), rejected candidates half-transparent, a
        # red cross on a peak excluded by hand, a cyan star with a teal edge
        # on one added by hand, an amber ring on an unconfirmed beat.
        self.accepted.setPen(pg.mkPen(None))
        self._acc_brushes = [pg.mkBrush(_alpha(p.plot_rpeak, int(255 * (0.35 + 0.05 * k))))
                             for k in range(_ALPHA_LEVELS)]
        self.accepted.setBrush(self._acc_brushes[-1])
        self.rejected.setBrush(pg.mkBrush(_alpha(p.LIGHT, 128)))
        self.rejected.setPen(pg.mkPen(None))
        self.excluded.setBrush(pg.mkBrush(p.RED))
        self.excluded.setPen(pg.mkPen(p.RED, width=1))
        self.added.setBrush(pg.mkBrush(ADD_COLOR))
        self.added.setPen(pg.mkPen(ADD_EDGE, width=1.2))
        self.unconfirmed.setBrush(pg.mkBrush(None))
        self.unconfirmed.setPen(pg.mkPen(UNCONFIRMED_COLOR, width=1.8))   # AMBER, as the Tk ring
        from ecg.app.theme import ARTIFACT_TYPE_COLOR
        for kind, item in self.artifact_marks.items():
            col = ARTIFACT_TYPE_COLOR.get(kind, p.ORANGE)
            item.setBrush(pg.mkBrush(_alpha(col, 217)))
            item.setPen(pg.mkPen(col, width=1))
        self.threshold_line.setPen(pg.mkPen(p.ORANGE, width=1.2,
                                            style=QtCore.Qt.PenStyle.DashLine))
        # Matplotlib's zoom box: a thin dashed outline (PyQtGraph's is a
        # yellow block).
        box = self.detail.getViewBox().rbScaleBox
        box.setPen(pg.mkPen(p.TEXT, width=1, style=QtCore.Qt.PenStyle.DashLine))
        box.setBrush(pg.mkBrush(_alpha(p.TEXT, 18)))
        for act, name in ((self.act_home, "home"), (self.act_back, "back"),
                          (self.act_forward, "forward"), (self.act_pan, "move"),
                          (self.act_zoom, "zoom_to_rect"), (self.act_save, "filesave")):
            act.setIcon(nav_icon(name, p.is_dark, p.TEXT))
        self._clip_peaks(force=True)
        for plot in self.strip:
            for ax in ("left", "bottom"):
                plot.getAxis(ax).setPen(pg.mkPen(p.MUTED))
                plot.getAxis(ax).setTextPen(pg.mkPen(p.MUTED))
        for c, col in ((self.rr_curve, p.MUTED), (self.hr_curve, p.ACCENT),
                       (self.q_curve, p.GREEN)):
            c.setPen(pg.mkPen(col, width=1))
            c.setSymbolBrush(pg.mkBrush(col))
            c.setSymbolPen(pg.mkPen(None))
        self.q_line.setPen(pg.mkPen(p.MUTED, width=0.8, style=QtCore.Qt.PenStyle.DashLine))
        self.band.setBrush(pg.mkBrush(_alpha(p.TEXT, 28)))
        for line in self.band.lines:
            line.setPen(pg.mkPen(p.TEXT, width=2))
