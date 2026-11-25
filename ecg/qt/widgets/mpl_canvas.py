# -*- coding: utf-8 -*-
"""
ecg.qt.widgets.mpl_canvas
-------------------------
A Matplotlib figure in a Qt widget (backend QtAgg). It takes the same
``draw_fn(fig)`` functions as the Tk interface's CanvasSlot
(ecg/ui/plots.py): the function receives the Figure and fills it, and is
replayed when asked again (theme change). The scientific figures (PSD,
Poincaré, Time course, Group...) therefore look the same in both interfaces
and in the exports.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtGui, QtWidgets

log = logging.getLogger("ecg")


class MplCanvas(QtWidgets.QWidget):
    """One Matplotlib Figure, drawn by a replaceable ``draw_fn(fig)``."""

    def __init__(self, parent: "QtWidgets.QWidget | None" = None,
                 width_in: float = 6.0, height_in: float = 4.0) -> None:
        super().__init__(parent)
        # At the screen's resolution, as the Tk CanvasSlot: text keeps its point
        # size on screen whatever matplotlib's figure.dpi setting (HiDPI scaling
        # is the canvas's device pixel ratio).
        screen = QtGui.QGuiApplication.primaryScreen()
        dpi = screen.logicalDotsPerInch() if screen is not None else 96.0
        self._figure = Figure(figsize=(width_in, height_in), dpi=dpi, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self._figure)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self._draw_fn: "Optional[Callable[[Figure], None]]" = None

    @property
    def figure(self) -> Figure:
        return self._figure

    def update_figure(self, draw_fn: "Callable[[Figure], None]") -> None:
        """Replace the figure's content by what *draw_fn(fig)* draws."""
        self._draw_fn = draw_fn
        self.redraw()

    def redraw(self) -> None:
        """Draw the last draw_fn again (e.g. after a theme change). The
        figure takes the theme's background first: a Figure keeps the colour
        it was created with, and clear() does not reset it (the dark theme
        drew light text on a white figure)."""
        self._figure.clear()
        self._figure.set_facecolor(matplotlib.rcParams["figure.facecolor"])
        if self._draw_fn is not None:
            try:
                self._draw_fn(self._figure)
            except Exception:                      # a figure never breaks the window
                log.exception("figure not drawn")
        self.canvas.draw_idle()

    def clear(self) -> None:
        self._draw_fn = None
        self._figure.clear()
        self.canvas.draw_idle()
