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
from PySide6 import QtCore, QtGui, QtWidgets

log = logging.getLogger("ecg")

RESIZE_SETTLE_MS = 150      # a figure re-lays itself out once a resize has settled


class _Canvas(FigureCanvasQTAgg):
    """The Agg canvas, re-laid out once a resize of the window settles
    rather than on every step of it: a figure with a constrained layout
    takes a tenth of a second or more to draw, and a window being dragged
    to a new size sends dozens of steps. Until then the previous image
    stays."""

    def __init__(self, figure: Figure) -> None:
        super().__init__(figure)
        self._resize_timer = QtCore.QTimer(self, singleShot=True, interval=RESIZE_SETTLE_MS)
        self._resize_timer.timeout.connect(self._apply_resize)
        self._resize_from: "Optional[QtCore.QSize]" = None

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        if not self.isVisible() or self._in_resize_event:
            super().resizeEvent(event)            # first layout, or hidden: at once
            return
        if self._resize_from is None:
            self._resize_from = event.oldSize()
        QtWidgets.QWidget.resizeEvent(self, event)
        self._resize_timer.start()

    def _apply_resize(self) -> None:
        old, self._resize_from = self._resize_from, None
        super().resizeEvent(QtGui.QResizeEvent(self.size(), old or self.size()))

    wheel_zoom = False      # MplCanvas.enable_wheel_yzoom

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802 (Qt name)
        """The wheel scrolls the page the figure is on (matplotlib's handler
        kept every wheel event, so a page did not scroll over a figure);
        Ctrl + wheel goes to the figure when it zooms."""
        ctrl = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
        if self.wheel_zoom and ctrl:
            super().wheelEvent(event)
            event.accept()
        else:
            event.ignore()


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
        self.canvas = _Canvas(self._figure)
        # An empty figure shows what to do instead (set_placeholder): a blank
        # frame told nothing.
        self._placeholder = QtWidgets.QLabel("")
        self._placeholder.setProperty("muted", True)
        self._placeholder.setWordWrap(True)
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setObjectName("Placeholder")
        # The canvas (and its toolbar, add_toolbar) or the placeholder.
        self._page = QtWidgets.QWidget()
        self._page_lay = QtWidgets.QVBoxLayout(self._page)
        self._page_lay.setContentsMargins(0, 0, 0, 0)
        self._page_lay.setSpacing(0)
        self._page_lay.addWidget(self.canvas, 1)
        self.toolbar = None
        self._stack = QtWidgets.QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._page)
        self._stack.addWidget(self._placeholder)
        self._draw_fn: "Optional[Callable[[Figure], None]]" = None

    def add_toolbar(self) -> None:
        """Matplotlib's toolbar above the figure, as Tk had on some figures:
        back to the whole view, previous / next view, move, zoom to a box,
        save the image."""
        if self.toolbar is not None:
            return
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
        tb = NavigationToolbar2QT(self.canvas, self._page, coordinates=True)
        tb.setIconSize(QtCore.QSize(16, 16))
        for act in list(tb.actions()):
            if act.text() in ("Subplots", "Customize"):     # layout dialogs: not needed
                tb.removeAction(act)
        self._page_lay.insertWidget(0, tb)
        self.toolbar = tb

    def enable_wheel_yzoom(self) -> None:
        """Ctrl + wheel over a panel zooms its vertical axis around the
        pointer (Tk's wheel zoom; the toolbar's Home brings the drawn view
        back). A plain wheel still scrolls the page."""
        self.canvas.wheel_zoom = True

        def on_scroll(ev) -> None:
            ax = ev.inaxes
            if ax is None or ev.ydata is None:
                return
            lo, hi = ax.get_ylim()
            f = 0.8 if ev.button == "up" else 1 / 0.8
            y = float(ev.ydata)
            ax.set_ylim(y - (y - lo) * f, y + (hi - y) * f)
            self.canvas.draw_idle()

        self.canvas.mpl_connect("scroll_event", on_scroll)

    @property
    def figure(self) -> Figure:
        return self._figure

    def set_placeholder(self, text: str) -> None:
        """What an empty figure says instead (e.g. which button fills it)."""
        self._placeholder.setText(text)
        self._sync_placeholder()

    def _sync_placeholder(self) -> None:
        empty = self._draw_fn is None and bool(self._placeholder.text())
        self._stack.setCurrentWidget(self._placeholder if empty else self._page)

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
        self._sync_placeholder()
        if self._draw_fn is not None:
            try:
                self._draw_fn(self._figure)
            except Exception:                      # a figure never breaks the window
                log.exception("figure not drawn")
        self.canvas.draw_idle()

    def clear(self) -> None:
        self._draw_fn = None
        self._figure.clear()
        self._figure.set_facecolor(matplotlib.rcParams["figure.facecolor"])
        self._sync_placeholder()
        self.canvas.draw_idle()
