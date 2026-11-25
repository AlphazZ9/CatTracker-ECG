# -*- coding: utf-8 -*-
"""
ecg.app.plots
-------------
What every figure shares: the matplotlib defaults (font, sizes, spines),
the axes style, Y limits centred on the mean, and text sized to fit a
figure. Moved from the CustomTkinter interface's ecg.ui.plots, without its
Tk canvas (the interface draws a Figure in its own canvas).
"""
from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ecg.app.theme import PLOT, THEME

log = logging.getLogger("ecg")

# Applied once at import time. font.family is the system UI font
# (THEME.font_family), so chart text matches the interface around it.
plt.rcParams.update({
    "font.family":     THEME.font_family,
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.titleweight": "bold",
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "legend.frameon":   False,
    "legend.borderpad": 0.4,
    "legend.labelspacing": 0.4,
    "axes.titlepad":    6,
    "axes.labelpad":    4,
    "xtick.major.pad":  4,
    "ytick.major.pad":  4,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "grid.linewidth":   0.4,
    "lines.antialiased": True,
})


def fig_text_width_in(text: str, fontsize: float) -> float:
    """Width (inches) of one line of *text* at *fontsize* pt in the current
    matplotlib font (measured like ``wrap_fig_text``)."""
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.font_manager import FontProperties

    if not text:
        return 0.0
    renderer = RendererAgg(1, 1, 72)
    w = renderer.get_text_width_height_descent(text, FontProperties(size=fontsize),
                                               ismath=False)[0]
    return w / 72.0 / 0.98


def ellipsize_fig_text(text: str, width_in: float, fontsize: float) -> str:
    """*text* shortened in the middle ("…") so that it fits *width_in*
    inches at *fontsize* pt; unchanged when it already fits. For a file
    path in a one-line figure header (both ends -- the drive / top folder
    and the file name -- stay readable)."""
    if fig_text_width_in(text, fontsize) <= width_in:
        return text
    keep = len(text)
    while keep > 2:
        keep -= 1
        head = (keep + 1) // 2
        cand = f"{text[:head]}…{text[len(text) - (keep - head):]}"
        if fig_text_width_in(cand, fontsize) <= width_in:
            return cand
    return "…"


def wrap_fig_text(text: str, width_in: float, fontsize: float) -> str:
    """*text* with line breaks so that no line is wider than *width_in*
    inches at *fontsize* pt in the current matplotlib font (the one figure
    text is drawn with; measured with an offscreen
    Agg renderer at 72 dpi, i.e. in points; 2 % margin for hinting at other
    resolutions). Long single-line figure footers were otherwise clipped by
    the figure edge."""
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.font_manager import FontProperties

    max_pt = max(1.0, width_in * 72.0 * 0.98)
    renderer = RendererAgg(1, 1, 72)
    prop = FontProperties(size=fontsize)

    def _w(s: str) -> float:
        if not s:
            return 0.0
        return renderer.get_text_width_height_descent(s, prop, ismath=False)[0]

    lines: "list[str]" = []
    cur = ""
    for word in text.split(" "):
        cand = f"{cur} {word}" if cur else word
        if cur and _w(cand) > max_pt:
            lines.append(cur)
            cur = word
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def centered_ylim(values, center: "Optional[float]" = None,
                  pad: float = 0.08) -> "Optional[tuple[float, float]]":
    """Y limits centred on *center* (default: the mean of *values*), wide
    enough for every finite value plus *pad* of the half-range: RR and HR
    plots put their mean on the middle of the axis. None without a finite
    value."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    c = float(np.mean(v)) if center is None or not np.isfinite(center) else float(center)
    half = float(np.max(np.abs(v - c)))
    if half <= 0:
        half = max(abs(c) * 0.05, 1.0)
    half *= 1.0 + pad
    return c - half, c + half


def style_axes(ax) -> None:
    """Apply the active PLOT theme to a matplotlib Axes object."""
    ax.set_facecolor(PLOT["axes"])
    ax.grid(True, color=PLOT["grid"], lw=0.4, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PLOT["border"])
    ax.spines["bottom"].set_color(PLOT["border"])
    ax.tick_params(colors=PLOT["muted"])
    # y=0.05 laissait un bandeau d'air visible au-dessus des pics les plus
    # hauts (ex. seuil de détection R-peak collé au bord du graphe). 0.03
    # garde assez de marge pour ne rien couper (markers, annotations texte)
    # tout en réduisant l'espace mort entre les données et le cadre.
    ax.margins(x=0.01, y=0.03)
    ax.xaxis.label.set_color(PLOT["text"])
    ax.yaxis.label.set_color(PLOT["text"])
    ax.title.set_color(PLOT["text"])

