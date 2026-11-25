# -*- coding: utf-8 -*-
"""
ecg.qt.widgets.beat_chart
-------------------------
Morphology's mean beat on screen (PyQtGraph): a few individual beats in
light grey, the ±1 SD band, the mean beat in red, the R peak, and the wave
landmarks of the template as dotted lines with their names in rows above
the beat, so no two overlap (their search windows are shown in the wave
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
LABEL_GAP_MS = 9.0          # two names closer than this go on different rows


def label_rows(centers: "list[float]", gap: float = LABEL_GAP_MS) -> "list[int]":
    """A row for each landmark name, left to right: the lowest row whose
    last name is at least *gap* ms away."""
    order = sorted(range(len(centers)), key=lambda i: centers[i])
    last: "list[float]" = []
    rows = [0] * len(centers)
    for i in order:
        for r, x in enumerate(last):
            if centers[i] - x >= gap:
                rows[i], last[r] = r, centers[i]
                break
        else:
            rows[i] = len(last)
            last.append(centers[i])
    return rows


class BeatChart(_Chart):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(360)
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
        rows = label_rows(centers)
        y_lo = float(np.nanmin(mean - (sd if sd is not None else 0)))
        y_hi = float(np.nanmax(mean + (sd if sd is not None else 0)))
        span = (y_hi - y_lo) or 1.0
        n_rows = (max(rows) + 1) if rows else 0
        row_h = 0.11 * span
        for (key, (c, half)), row in zip(lm.items(), rows):
            col = self._data["colors"].get(key, p.MUTED)
            add(pg.InfiniteLine(c, angle=90, pen=pg.mkPen(_alpha(col, 200), width=1.2,
                                                          style=QtCore.Qt.PenStyle.DotLine)),
                bounds=False)
            name = self._data["labels"].get(key, key.replace("_", " "))
            label = pg.TextItem(name, color=col, anchor=(0.5, 1))
            font = QtGui.QFont()
            font.setBold(True)
            label.setFont(font)
            label.setPos(c, y_hi + 0.06 * span + (n_rows - row) * row_h)
            add(label, bounds=False)
        # The beat and the landmarks themselves (their search windows are the
        # wave editor's business): no wide empty margins.
        xs = [float(t[0]), float(t[-1])] + centers
        self.setXRange(min(xs) - 6, max(xs) + 6, padding=0)
        self.setYRange(y_lo - 0.08 * span, y_hi + 0.1 * span + (n_rows + 1) * row_h,
                       padding=0)

    def style(self, p) -> None:
        super().style(p)

    def apply_palette(self, p) -> None:
        self.style(p)
        self._draw()
