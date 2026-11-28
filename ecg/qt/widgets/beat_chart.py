# -*- coding: utf-8 -*-
"""
ecg.qt.widgets.beat_chart
-------------------------
Morphology's mean beat on screen (PyQtGraph): a few individual beats in
light grey, the ±1 SD band, the mean beat in red, the R peak, and the wave
landmarks of the template as dotted lines with their names in rows above
the beat, so no two overlap (measured in pixels at the chart's width) (their search windows are shown in the wave
editor). The view spans the landmarks and the beat, a few ms around.

Display only: the Matplotlib figure of the same beat stays what the
exports hold.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui

from ecg.qt import theme
from ecg.qt.widgets.rhythm_charts import _Chart, _alpha

GHOSTS = 30                 # individual beats drawn under the mean
LABEL_PAD_PX = 10           # room left between two landmark names on one row


def label_rows(spans: "list[tuple[float, float]]") -> "list[int]":
    """A row for each landmark name given its extent (left, right) on the
    time axis, left to right: the lowest row where it starts after the
    previous name of that row ends, so no two names overlap."""
    order = sorted(range(len(spans)), key=lambda i: spans[i][0])
    ends: "list[float]" = []
    rows = [0] * len(spans)
    for i in order:
        left, right = spans[i]
        for r, end in enumerate(ends):
            if left >= end:
                rows[i], ends[r] = r, right
                break
        else:
            rows[i] = len(ends)
            ends.append(right)
    return rows


class BeatChart(_Chart):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(300)
        self.setLabel("bottom", "Time from the R peak (ms)")
        self.setLabel("left", "Amplitude (norm.)")
        self._items: list = []
        self._data: Optional[dict] = None
        self.style(theme.current())
        self.say("Click Analyze: the mean beat and its landmarks appear here.")

    def set_beat(self, results: "Optional[dict]", landmarks: "Optional[dict]" = None,
                 labels: "Optional[dict]" = None, colors: "Optional[dict]" = None) -> None:
        """*results*: Analyze's (beat_time, beat_template, beat_sd,
        beat_matrix); *landmarks*: {key: (centre_ms, half_width_ms)}."""
        self._data = None
        if results is not None and results.get("beat_template") is not None \
                and results.get("beat_time") is not None:
            self._data = dict(results=results, landmarks=landmarks or {},
                              labels=labels or {}, colors=colors or {})
        self._draw()

    def _draw(self) -> None:
        p = theme.current()
        for item in self._items:
            self.removeItem(item)
        self._items = []
        if self._data is None:
            self.say("Click Analyze: the mean beat and its landmarks appear here.")
            return
        self.say("")
        r = self._data["results"]
        t = np.asarray(r["beat_time"], float)
        mean = np.asarray(r["beat_template"], float)
        sd = r.get("beat_sd")
        mat = r.get("beat_matrix")

        def add(item, bounds=True):
            self.addItem(item, ignoreBounds=not bounds)
            self._items.append(item)

        if mat is not None and len(mat):
            stride = max(1, len(mat) // GHOSTS)
            for beat in np.asarray(mat)[::stride]:
                add(pg.PlotDataItem(t, beat, pen=pg.mkPen(_alpha(p.LIGHT, 60), width=1)))
        if sd is not None:
            sd = np.asarray(sd, float)
            lo = pg.PlotDataItem(t, mean - sd)
            hi = pg.PlotDataItem(t, mean + sd)
            add(pg.FillBetweenItem(lo, hi, brush=pg.mkBrush(_alpha(p.ACCENT, 40))))
        add(pg.PlotDataItem(t, mean, pen=pg.mkPen(p.ACCENT, width=2.2)))
        add(pg.InfiniteLine(0, angle=90, pen=pg.mkPen(p.TEXT, width=1,
                                                       style=QtCore.Qt.PenStyle.DashLine)))
        lm = self._data["landmarks"]
        centers = [float(v[0]) for v in lm.values()]
        names = [self._data["labels"].get(k, k.replace("_", " ")) for k in lm]
        font = QtGui.QFont()
        font.setBold(True)
        fm = QtGui.QFontMetrics(font)
        widths = [fm.horizontalAdvance(n) + LABEL_PAD_PX for n in names]
        # The beat and the landmarks themselves (their search windows are the
        # wave editor's business), widened so each name fits inside; the
        # names' extent in ms follows the chart's width in pixels.
        xs = [float(t[0]), float(t[-1])] + centers
        x0, x1 = min(xs) - 6, max(xs) + 6
        px = max(float(self.getPlotItem().getViewBox().width()), 200.0)
        spans: "list[tuple[float, float]]" = []
        for _ in range(3):
            ms_px = (x1 - x0) / px
            spans = [(c - w / 2 * ms_px, c + w / 2 * ms_px) for c, w in zip(centers, widths)]
            x0 = min([x0] + [a for a, _b in spans])
            x1 = max([x1] + [b for _a, b in spans])
        rows = label_rows(spans)
        y_lo = float(np.nanmin(mean - (sd if sd is not None else 0)))
        y_hi = float(np.nanmax(mean + (sd if sd is not None else 0)))
        span = (y_hi - y_lo) or 1.0
        n_rows = (max(rows) + 1) if rows else 0
        row_h = 0.11 * span
        for (key, (c, half)), name, row in zip(lm.items(), names, rows):
            col = self._data["colors"].get(key, p.MUTED)
            add(pg.InfiniteLine(c, angle=90, pen=pg.mkPen(_alpha(col, 200), width=1.2,
                                                          style=QtCore.Qt.PenStyle.DotLine)),
                bounds=False)
            label = pg.TextItem(name, color=col, anchor=(0.5, 1))
            label.setFont(font)
            label.setPos(c, y_hi + 0.06 * span + (n_rows - row) * row_h)
            add(label, bounds=False)
        self.setXRange(x0, x1, padding=0)
        self.setYRange(y_lo - 0.08 * span, y_hi + 0.1 * span + (n_rows + 1) * row_h,
                       padding=0)

    def resizeEvent(self, ev) -> None:  # noqa: N802 (Qt name)
        super().resizeEvent(ev)
        # The names' rows depend on the width: placed again once it settles
        # (pyqtgraph also calls this while building the chart).
        if getattr(self, "_data", None) is not None:
            if not hasattr(self, "_relayout"):
                self._relayout = QtCore.QTimer(self, singleShot=True, interval=80)
                self._relayout.timeout.connect(self._draw)
            self._relayout.start()

    def style(self, p) -> None:
        super().style(p)

    def apply_palette(self, p) -> None:
        self.style(p)
        self._draw()
