# -*- coding: utf-8 -*-
"""
ecg.qt.widgets.rhythm_charts
----------------------------
The Rhythm & HRV charts on screen, drawn with PyQtGraph: heart rate (or RR)
over time, Poincaré with its SD1 / SD2 ellipse and the multiscale entropy,
the power spectrum with its bands, the distribution of the valid RR
intervals.

Display only, from the results the analysis already holds (no number is
computed here that a table or an export shows): they follow the window
size at once, and a theme change recolours them without a redraw. The
Matplotlib figures of the same results stay what the exports (PDF, ZIP)
contain.

Scales leave out the rare extreme values (the 0.5th to 99.5th percentile of
what is drawn), so one missed beat does not flatten the rest; the points
beyond are drawn at the edge as small triangles.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ecg.qt import theme

log = logging.getLogger("ecg")

ROBUST_PCT = (0.5, 99.5)


def robust_range(values: np.ndarray, pad_frac: float = 0.08,
                 min_span: float = 0.0) -> "Optional[tuple[float, float]]":
    """The 0.5th-99.5th percentile of the finite *values*, padded, at least
    *min_span* wide; None when there is nothing to scale."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    lo, hi = (float(x) for x in np.percentile(v, ROBUST_PCT))
    span = max(hi - lo, min_span)
    mid = (lo + hi) / 2
    lo, hi = mid - span / 2, mid + span / 2
    pad = pad_frac * span or 1.0
    return lo - pad, hi + pad


def few_ticks(axis: "pg.AxisItem", lo: float, hi: float, n: int = 3) -> None:
    """At most about *n* labelled ticks on *axis* for the range lo..hi (a
    "nice" step: 1, 2 or 5 × 10^k): pyqtgraph's own spacing crowds the
    labels of a chart only a few tens of pixels high."""
    span = float(hi - lo)
    if not np.isfinite(span) or span <= 0:
        axis.setTickSpacing()                     # back to automatic
        return
    raw = span / max(n, 1)
    mag = 10 ** np.floor(np.log10(raw))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)
    axis.setTickSpacing(step, step)


def _alpha(color: str, a: int) -> QtGui.QColor:
    c = QtGui.QColor(color)
    c.setAlpha(a)
    return c


class _ZoomBox(pg.ViewBox):
    """The view of a zoomable chart: Ctrl + wheel zooms (on the plot the time
    axis only when *time_only*, on an axis that axis), a drag pans."""

    def __init__(self) -> None:
        super().__init__()
        self.time_only = False

    def wheelEvent(self, ev, axis=None):  # noqa: N802 (Qt name)
        if axis is None and self.time_only:
            axis = 0
        super().wheelEvent(ev, axis)

    def mouseDragEvent(self, ev, axis=None):  # noqa: N802 (Qt name)
        if axis is None and self.time_only:
            axis = 0
        super().mouseDragEvent(ev, axis)


class _Chart(pg.PlotWidget):
    """A plot with the app's look: no menu, no buttons, quiet axes. Fixed
    unless enable_zoom() (then Ctrl + wheel zooms, a drag pans, go_home()
    brings back the view the data was drawn with)."""

    def __init__(self, parent: "Optional[QtWidgets.QWidget]" = None) -> None:
        self._box = _ZoomBox()
        super().__init__(parent, viewBox=self._box)
        self._zoom = False
        self._home: "Optional[list]" = None
        self.setMenuEnabled(False)
        self.hideButtons()
        self.getPlotItem().getViewBox().setMouseEnabled(x=False, y=False)
        # At most ~one point per pixel column (peak keeps the extremes): an
        # hour of mouse ECG is ~36 000 beats per tachogram line.
        self.getPlotItem().setDownsampling(auto=True, mode="peak")
        self._placeholder = pg.TextItem("", anchor=(0.5, 0.5))
        self.addItem(self._placeholder, ignoreBounds=True)
        # Room above the plot: the top tick label was cut in half.
        self.getPlotItem().setContentsMargins(2, 8, 6, 2)

    # ── zoom ────────────────────────────────────────────────
    def enable_zoom(self, time_only: bool = False) -> None:
        """Ctrl + wheel zooms (on the plot: time only if *time_only*; on an
        axis: that axis), a drag pans. Plain wheel still scrolls the page."""
        self._zoom = True
        self._box.time_only = time_only
        self._box.setMouseEnabled(x=True, y=True)
        self.setToolTip("Ctrl + wheel: zoom (over the left axis: vertical) · drag: move · "
                        "Reset zoom: the whole chart")

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:  # noqa: N802 (Qt name)
        ctrl = bool(ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
        if self._zoom and ctrl:
            super().wheelEvent(ev)
            ev.accept()
        else:
            ev.ignore()                     # the page scrolls

    def set_home(self) -> None:
        """The view just drawn is the one Reset zoom comes back to."""
        self._home = self._box.viewRange()

    def go_home(self) -> None:
        if self._home is not None:
            (x0, x1), (y0, y1) = self._home
            self._box.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)

    def sizeHint(self) -> QtCore.QSize:
        # A modest wish: pyqtgraph asks for 600 px, which made every page of
        # charts taller than the window. The layouts then give a chart the
        # height there is (RhythmTab._fit_charts).
        return QtCore.QSize(360, max(self.minimumHeight(), 100))

    def say(self, text: str) -> None:
        """A message in the middle of an empty chart."""
        self._placeholder.setText(text)
        self._placeholder.setVisible(bool(text))
        if text:
            vb = self.getPlotItem().getViewBox()
            (x0, x1), (y0, y1) = vb.viewRange()
            self._placeholder.setPos((x0 + x1) / 2, (y0 + y1) / 2)

    def style(self, p: "theme.Palette") -> None:
        self.setBackground(p.PANEL)
        for ax in ("left", "bottom"):
            a = self.getAxis(ax)
            a.setPen(pg.mkPen(p.BORDER2))
            a.setTextPen(pg.mkPen(p.MUTED))
        self._placeholder.setColor(p.MUTED)


class RhythmCharts(QtCore.QObject):
    """The four charts of Rhythm & HRV and how to fill them from results.

    *on_time_clicked(t)*: a click on the heart-rate chart (the beat at *t*
    seconds is shown in Signal)."""

    def __init__(self, on_time_clicked: "Optional[Callable[[float], None]]" = None,
                 parent: "Optional[QtCore.QObject]" = None,
                 on_next_flagged: "Optional[Callable[[], None]]" = None) -> None:
        super().__init__(parent)
        self._on_time = on_time_clicked
        self._on_next = on_next_flagged          # right click on a tachogram (Tk)
        self.unit = "HR"                     # the tachogram shows "HR" or "RR"
        self._results: "Optional[dict]" = None
        self._spikes: "Optional[np.ndarray]" = None
        self._annotations: list = []

        # RR intervals (tacho_rr) above the instantaneous heart rate (tacho),
        # one time axis, as the Tk tachogram.
        self.tacho = _Chart()
        self.tacho.setMinimumHeight(140)
        self.tacho.setLabel("bottom", "Time", units="s")
        self.tacho.setLabel("left", "HR (bpm)")
        self.tacho.showGrid(x=False, y=True, alpha=0.12)
        self.t_line = pg.PlotDataItem(connect="finite")        # every beat, thin
        self.t_trend = pg.PlotDataItem(connect="finite")       # its 9-beat running median
        self.t_excl = pg.ScatterPlotItem(size=6, symbol="o")
        self.t_flag = pg.ScatterPlotItem(size=9, symbol="o")
        self.t_over = pg.ScatterPlotItem(size=8, symbol="t1")     # beyond the top
        self.t_under = pg.ScatterPlotItem(size=8, symbol="t")     # beyond the bottom
        self.tacho_rr = _Chart()
        self.tacho_rr.setMinimumHeight(120)
        self.tacho_rr.setLabel("left", "RR (ms)")
        self.tacho_rr.showGrid(x=False, y=True, alpha=0.12)
        self.tacho_rr.setXLink(self.tacho)
        self.tacho_rr.getAxis("bottom").setStyle(showValues=False)
        self.r_line = pg.PlotDataItem(connect="finite")
        self.r_trend = pg.PlotDataItem(connect="finite")
        self.r_excl = pg.ScatterPlotItem(size=6, symbol="o")
        self.r_flag = pg.ScatterPlotItem(size=9, symbol="o")
        self.r_over = pg.ScatterPlotItem(size=8, symbol="t1")
        self.r_under = pg.ScatterPlotItem(size=8, symbol="t")
        # One lane per chart: (chart, line, trend, excluded, flagged, over,
        # under, the unit drawn, the annotated-period items).
        self._lanes = {
            "HR": (self.tacho, self.t_line, self.t_trend, self.t_excl, self.t_flag,
                   self.t_over, self.t_under, []),
            "RR": (self.tacho_rr, self.r_line, self.r_trend, self.r_excl, self.r_flag,
                   self.r_over, self.r_under, []),
        }
        for chart, *items, _regions in self._lanes.values():
            for item in items:
                chart.addItem(item)
            chart.scene().sigMouseClicked.connect(
                lambda ev, c=chart: self._tacho_clicked(ev, c))

        # Poincaré, square, and the entropy under it.
        self.poincare = _Chart()
        self.poincare.setAspectLocked(True)
        self.poincare.setMinimumHeight(95)
        self.poincare.setLabel("bottom", "RRₙ (ms)")
        self.poincare.setLabel("left", "RRₙ₊₁ (ms)")
        self.p_identity = pg.PlotDataItem()
        self.p_points = pg.ScatterPlotItem(size=4, symbol="o")
        self.p_ellipse = pg.PlotDataItem()
        for item in (self.p_identity, self.p_points, self.p_ellipse):
            self.poincare.addItem(item)
        self.mse = _Chart()
        self.mse.setMinimumHeight(85)
        self.mse.setLabel("bottom", "Scale factor")
        self.mse.setLabel("left", "SampEn")
        self.m_line = pg.PlotDataItem(symbol="o", symbolSize=5)
        self.mse.addItem(self.m_line)

        # Power spectrum (log power), its bands shaded.
        self.psd = _Chart()
        self.psd.setLogMode(x=False, y=True)
        self.psd.setMinimumHeight(150)
        self.psd.setLabel("bottom", "Frequency", units="Hz")
        self.psd.setLabel("left", "PSD (ms²/Hz)")
        self.s_line = pg.PlotDataItem()
        self.psd.addItem(self.s_line)
        self._s_bands: list = []

        # RR distribution of the valid intervals.
        self.hist = _Chart()
        self.hist.setMinimumHeight(150)
        self.hist.setLabel("bottom", "RR (ms)")
        self.hist.setLabel("left", "Beats")
        self.h_bars = pg.BarGraphItem(x=[], height=[], width=1.0)
        self.hist.addItem(self.h_bars)
        # The triangular-index fit (HTI / TINN) over it, as Tk and the export.
        self.h_tri = pg.PlotDataItem()
        self.h_tri_label = pg.TextItem("", anchor=(1, 0))
        self.hist.addItem(self.h_tri)
        self.hist.addItem(self.h_tri_label, ignoreBounds=True)

        for chart in (self.tacho, self.tacho_rr):
            chart.enable_zoom(time_only=True)
        for chart in (self.poincare, self.mse, self.psd, self.hist):
            chart.enable_zoom()

        self.apply_palette(theme.current())

    def reset_zoom(self) -> None:
        """Every chart back to the view it was drawn with."""
        for chart in (self.tacho, self.tacho_rr, self.poincare, self.mse, self.psd,
                      self.hist):
            chart.go_home()

    # ── data ────────────────────────────────────────────────
    def set_results(self, results: "Optional[dict]", spikes: "Optional[np.ndarray]" = None,
                    annotations: "Optional[list]" = None,
                    psd: "Optional[dict]" = None) -> None:
        """Draw *results* (an Analyze result dict; None: empty), the flagged
        beats *spikes* (s), the annotated periods, and *psd* (the spectrum
        the analysis integrated: PlotController.psd_for_display)."""
        self._results = results
        self._spikes = None if spikes is None else np.asarray(spikes, dtype=float)
        self._annotations = list(annotations or [])
        self._psd = psd
        self._draw_tacho()
        self._draw_poincare()
        self._draw_psd()
        self._draw_hist()
        for chart in (self.tacho, self.tacho_rr, self.poincare, self.mse, self.psd,
                      self.hist):
            chart.set_home()                  # what Reset zoom comes back to

    def set_unit(self, unit: str) -> None:
        """Kept for callers of the former HR / RR switch: both are drawn."""
        self.unit = "RR" if unit == "RR" else "HR"

    def _series(self):
        from ecg.core.analysis import rr_timeseries
        ts = rr_timeseries(self._results)
        if len(ts) == 0:
            return None
        return (ts["Time_s"].to_numpy(float), ts["RR_ms"].to_numpy(float),
                ts["HR_bpm"].to_numpy(float), ts["valid"].to_numpy(bool))

    def _draw_tacho(self) -> None:
        """Both lanes: RR above, HR below (the HR one is the "tacho" chart)."""
        s = self._series() if self._results is not None else None
        for unit in ("RR", "HR"):
            self._draw_lane(unit, s)

    def _draw_lane(self, unit: str, s) -> None:
        p = theme.current()
        chart, line_i, trend_i, excl_i, flag_i, over_i, under_i, regions = self._lanes[unit]
        for item in regions:
            chart.removeItem(item)
        regions.clear()
        if s is None:
            for item in (line_i, trend_i, excl_i, flag_i, over_i, under_i):
                item.setData([], [])
            chart.say("Detect the peaks, then click Analyze: every beat appears here."
                      if unit == "HR" else "")
            return
        chart.say("")
        t, rr, hr, valid = s
        y = hr if unit == "HR" else rr
        line = np.where(valid, y, np.nan)          # the line breaks around excluded ones
        line_i.setData(t, line)
        from ecg.qt.widgets.ecg_plot import _running_median
        trend = np.full(len(y), np.nan)
        if valid.sum() >= 3:
            trend[valid] = _running_median(y[valid], 9)
        trend_i.setData(t, trend)
        excl_i.setData(t[~valid], y[~valid])
        rng = robust_range(y[valid], min_span=20.0 if unit == "HR" else 4.0)
        if rng is not None:
            lo, hi = rng
            chart.setYRange(lo, hi, padding=0)
            over, under = y > hi, y < lo
            over_i.setData(t[over], np.full(int(over.sum()), hi - 0.02 * (hi - lo)))
            under_i.setData(t[under], np.full(int(under.sum()), lo + 0.02 * (hi - lo)))
        if len(t):
            chart.setXRange(float(t[0]), float(t[-1]), padding=0.01)
        if self._spikes is not None and len(self._spikes):
            idx = np.clip(np.searchsorted(t, self._spikes), 0, len(t) - 1)
            ys = y[idx]
            if rng is not None:
                ys = np.clip(ys, rng[0], rng[1])
            flag_i.setData(self._spikes, ys)
        else:
            flag_i.setData([], [])
        for a in self._annotations:
            color = a.get("color") or p.MUTED
            region = pg.LinearRegionItem(values=(float(a["t_start"]), float(a["t_end"])),
                                         movable=False, brush=pg.mkBrush(_alpha(color, 40)),
                                         pen=pg.mkPen(None))
            region.setZValue(-20)
            chart.addItem(region)
            regions.append(region)

    def _draw_poincare(self) -> None:
        r = self._results
        if r is None or r.get("rr_ms") is None:
            for item in (self.p_points, self.p_ellipse, self.p_identity, self.m_line):
                item.setData([], [])
            self.poincare.say("Click Analyze to see the Poincaré plot.")
            return
        from ecg.core.analysis import successive_valid_pairs
        s = self._series()
        rr = np.asarray(r["rr_ms"], dtype=float)
        valid = s[3] if s is not None and len(s[3]) == len(rr) else np.isfinite(rr)
        a, b = successive_valid_pairs(rr, valid)
        if len(a) < 1:
            self.poincare.say("Too few valid RR pairs.")
            return
        self.poincare.say("")
        self.p_points.setData(a, b)
        rng = robust_range(np.concatenate([a, b]), pad_frac=0.12, min_span=10.0)
        lo, hi = rng if rng is not None else (float(min(a.min(), b.min())),
                                              float(max(a.max(), b.max())))
        self.p_identity.setData([lo, hi], [lo, hi])
        self.poincare.setXRange(lo, hi, padding=0)
        self.poincare.setYRange(lo, hi, padding=0)
        nl = r.get("hrv_nonlin")
        sd1 = sd2 = float("nan")
        try:
            sd1 = float(nl["HRV_SD1"].values[0])
            sd2 = float(nl["HRV_SD2"].values[0])
        except Exception as e:
            log.debug("SD1 / SD2 not in the non-linear table: %s", e)
        if np.isfinite(sd1) and np.isfinite(sd2):
            c = float(np.mean(rr[valid]))
            th = np.linspace(0, 2 * np.pi, 120)
            # SD2 along the identity line, SD1 across it.
            u, v = sd2 * np.cos(th), sd1 * np.sin(th)
            self.p_ellipse.setData(c + (u - v) / np.sqrt(2), c + (u + v) / np.sqrt(2))
        else:
            self.p_ellipse.setData([], [])
        attrs = getattr(nl, "attrs", {}) if nl is not None else {}
        scales, values = attrs.get("mse_scales") or [], attrs.get("mse_values") or []
        if scales and values:
            vals = np.asarray(values, float)
            self.m_line.setData(np.asarray(scales, float), vals)
            if np.isfinite(vals).any():
                few_ticks(self.mse.getAxis("left"), 0.0, float(np.nanmax(vals)), 2)
            self.mse.say("")
        else:
            self.m_line.setData([], [])
            self.mse.say("No multiscale entropy (see the status above)")

    def _draw_psd(self) -> None:
        p = theme.current()
        for item in self._s_bands:
            self.psd.removeItem(item)
        self._s_bands = []
        disp = getattr(self, "_psd", None)
        if not disp:
            self.s_line.setData([], [])
            self.psd.say("Click Analyze to see the power spectrum.")
            return
        if disp.get("withheld"):
            self.s_line.setData([], [])
            self.psd.say(str(disp["withheld"]))
            return
        self.psd.say("")
        f = np.asarray(disp["freqs"], float)
        pw = np.asarray(disp["psd"], float)
        keep = (f > 0) & np.isfinite(pw) & (pw > 0)
        f, pw = f[keep], pw[keep]
        if not len(f):
            self.s_line.setData([], [])
            return
        self.s_line.setData(f, pw)
        colors = (p.LIGHT, p.ACCENT, p.MUTED)
        names = ("VLF", "LF", "HF")
        for (band, name, color) in zip(disp["bands"], names, colors):
            lo, hi = float(band[0]), float(band[1])
            region = pg.LinearRegionItem(values=(lo, hi), movable=False,
                                         brush=pg.mkBrush(_alpha(color, 38)),
                                         pen=pg.mkPen(None))
            region.setZValue(-20)
            self.psd.addItem(region)
            label = pg.TextItem(name, color=color if name != "VLF" else p.MUTED,
                                anchor=(0, 0))
            label.setPos(lo, float(np.log10(pw.max())))
            self.psd.addItem(label, ignoreBounds=True)
            self._s_bands += [region, label]
        top = float(disp["bands"][-1][1])
        self.psd.setXRange(0.0, top, padding=0.01)

    def _draw_hist(self) -> None:
        s = self._series() if self._results is not None else None
        if s is None or not s[3].any():
            self.h_bars.setOpts(x=[], height=[], width=1.0)
            self.h_tri.setData([], [])
            self.h_tri_label.setText("")
            self.hist.say("Click Analyze to see the RR distribution.")
            return
        self.hist.say("")
        rr = s[1][s[3]]
        rng = robust_range(rr, pad_frac=0.05, min_span=4.0)
        lo, hi = rng if rng is not None else (float(rr.min()), float(rr.max()))
        # Bins a whole number of the sampling step (RR comes in steps of
        # 1 / fs): finer bins leave every other one empty.
        u = np.unique(np.round(rr, 6))
        step = float(np.min(np.diff(u))) if len(u) > 1 else 0.5
        width = step * max(1, int(np.ceil((hi - lo) / 30.0 / step)))
        edges = np.arange(lo, hi + width, width)
        counts, edges = np.histogram(np.clip(rr, lo, hi), bins=edges)
        centres = (edges[:-1] + edges[1:]) / 2
        self.h_bars.setOpts(x=centres, height=counts, width=width * 0.86)
        self.hist.setXRange(lo, hi, padding=0)
        self.hist.setYRange(0, max(1, int(counts.max())) * 1.08, padding=0)
        self._draw_triangle(rr, float(counts.max()), lo, hi)

    def _draw_triangle(self, rr: np.ndarray, y_top: float, lo: float, hi: float) -> None:
        """PlotController.plot_rr's HTI / TINN overlay: a triangle on the
        mode of NeuroKit2's 1/128 s bins, as wide as TINN, as tall as the
        bars; the two values in the corner. Nothing without HTI."""
        self.h_tri.setData([], [])
        self.h_tri_label.setText("")
        r = self._results or {}
        rt = r.get("hrv_time")
        hti = tinn = None
        try:
            hti = float(rt["HRV_HTI"].values[0])
            tinn = float(rt["HRV_TINN"].values[0])
        except Exception:
            pass
        if not (hti and np.isfinite(hti) and hti > 0 and len(rr) > 3):
            return
        binsize = 1000.0 / 128.0
        edges = np.arange(0, np.nanmax(rr) + binsize, binsize)
        c, edges = np.histogram(rr, bins=edges)
        if not len(c) or c.max() <= 0:
            return
        x_mode = float(edges[int(np.argmax(c))]) + binsize / 2       # the bin's centre
        half = tinn / 2.0 if (tinn and np.isfinite(tinn) and tinn > 0) else binsize
        p = theme.current()
        self.h_tri.setData([x_mode - half, x_mode, x_mode + half], [0, y_top, 0])
        self.h_tri.setPen(pg.mkPen(p.TEXT, width=1, style=QtCore.Qt.PenStyle.DashLine))
        text = f"HTI {hti:.2f}" + (f"  ·  TINN {tinn:.1f} ms"
                                    if tinn and np.isfinite(tinn) else "")
        self.h_tri_label.setText(text)
        self.h_tri_label.setColor(p.TEXT)
        self.h_tri_label.setPos(hi, y_top * 1.07)

    # ── interaction, look ───────────────────────────────────
    def _tacho_clicked(self, ev, chart: "Optional[_Chart]" = None) -> None:
        chart = chart or self.tacho
        if self._results is None:
            return
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            if self._on_next is not None:       # Tk: right click, the next flagged beat
                self._on_next()
            return
        if self._on_time is None:
            return
        vb = chart.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        self._on_time(float(vb.mapSceneToView(ev.scenePos()).x()))

    def apply_palette(self, p: "theme.Palette") -> None:
        for chart in (self.tacho, self.tacho_rr, self.poincare, self.mse, self.psd, self.hist):
            chart.style(p)
        self.r_line.setPen(pg.mkPen(_alpha(p.TEXT, 60), width=1))
        self.r_trend.setPen(pg.mkPen(p.TEXT, width=2))
        self.r_excl.setPen(pg.mkPen(p.MUTED, width=1.2))
        self.r_excl.setBrush(pg.mkBrush(None))
        self.r_flag.setPen(pg.mkPen(p.ACCENT, width=1.6))
        self.r_flag.setBrush(pg.mkBrush(None))
        for item in (self.r_over, self.r_under):
            item.setPen(pg.mkPen(None))
            item.setBrush(pg.mkBrush(p.MUTED))
        self.t_line.setPen(pg.mkPen(_alpha(p.ACCENT, 70), width=1))
        self.t_trend.setPen(pg.mkPen(p.ACCENT, width=2))
        self.t_excl.setPen(pg.mkPen(p.MUTED, width=1.2))
        self.t_excl.setBrush(pg.mkBrush(None))
        self.t_flag.setPen(pg.mkPen(p.TEXT, width=1.6))
        self.t_flag.setBrush(pg.mkBrush(None))
        for item in (self.t_over, self.t_under):
            item.setPen(pg.mkPen(None))
            item.setBrush(pg.mkBrush(p.MUTED))
        self.p_points.setPen(pg.mkPen(None))
        self.p_points.setBrush(pg.mkBrush(_alpha(p.ACCENT, 90)))
        self.p_identity.setPen(pg.mkPen(p.BORDER2, width=1, style=QtCore.Qt.PenStyle.DashLine))
        self.p_ellipse.setPen(pg.mkPen(p.TEXT, width=1.6))
        self.m_line.setPen(pg.mkPen(p.TEXT, width=1.4))
        self.m_line.setSymbolBrush(pg.mkBrush(p.TEXT))
        self.m_line.setSymbolPen(pg.mkPen(None))
        self.s_line.setPen(pg.mkPen(p.TEXT, width=1.4))
        self.h_bars.setOpts(brush=pg.mkBrush(_alpha(p.ACCENT, 200)), pen=pg.mkPen(None))
        if self._results is not None:
            self._draw_tacho()
            self._draw_psd()


__all__ = ["RhythmCharts", "robust_range"]
