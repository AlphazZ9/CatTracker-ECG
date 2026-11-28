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
    wheel_always = False    # the figure takes the plain wheel (a page that does not scroll)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802 (Qt name)
        """The wheel scrolls the page the figure is on (matplotlib's handler
        kept every wheel event, so a page did not scroll over a figure);
        Ctrl + wheel goes to the figure when it zooms, the plain wheel too
        when it takes it (wheel_always)."""
        ctrl = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
        if self.wheel_always or (self.wheel_zoom and ctrl):
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
        # An empty figure shows what to do instead (set_placeholder): a faint
        # chart, the sentence, and the button that fills it when the page
        # gives one (set_placeholder_action); a blank frame told nothing.
        self._empty = QtWidgets.QWidget(objectName="EmptyState")
        el = QtWidgets.QVBoxLayout(self._empty)
        el.setContentsMargins(24, 16, 24, 16)
        el.setSpacing(10)
        el.addStretch(1)
        self._picto = QtWidgets.QLabel()
        self._picto.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        el.addWidget(self._picto)
        self._placeholder = QtWidgets.QLabel("")
        self._placeholder.setProperty("muted", True)
        self._placeholder.setWordWrap(True)
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setObjectName("Placeholder")
        # The whole width (centred text): a wrapped label given an alignment
        # in the layout lost its height and showed half a line.
        el.addWidget(self._placeholder)
        self._empty_btn = QtWidgets.QPushButton("")
        self._empty_btn.setProperty("action", True)
        self._empty_btn.hide()
        el.addWidget(self._empty_btn, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)
        el.addStretch(1)
        self._empty_action: "Optional[Callable[[], None]]" = None
        self._empty_btn.clicked.connect(lambda: self._empty_action and self._empty_action())
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
        self._stack.addWidget(self._empty)
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

    def set_placeholder_action(self, label: str, action: "Optional[Callable[[], None]]",
                               source: "Optional[QtWidgets.QAbstractButton]" = None) -> None:
        """The button under the empty figure's sentence (*label*, running
        *action*: the page's own command that fills the figure); None: none.
        *source*: the page's button for that command, whose enabled state it
        follows (no button offered that would do nothing)."""
        self._empty_action = action
        self._empty_btn.setText(label)
        self._empty_btn.setVisible(action is not None)
        if source is not None:
            self._empty_source = source
            source.installEventFilter(self)
            self._empty_btn.setEnabled(source.isEnabled())

    def eventFilter(self, obj: QtCore.QObject, ev: QtCore.QEvent) -> bool:  # noqa: N802
        if ev.type() == QtCore.QEvent.Type.EnabledChange \
                and obj is getattr(self, "_empty_source", None):
            self._empty_btn.setEnabled(obj.isEnabled())
        return False

    def _sync_placeholder(self) -> None:
        empty = self._draw_fn is None and bool(self._placeholder.text())
        if empty:
            from ecg.qt import theme
            from ecg.qt.widgets.shell import chart_pictogram
            self._picto.setPixmap(chart_pictogram(theme.current().BORDER2))
        self._stack.setCurrentWidget(self._empty if empty else self._page)

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
