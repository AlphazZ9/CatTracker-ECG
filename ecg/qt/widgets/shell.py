# -*- coding: utf-8 -*-
"""
ecg.qt.widgets.shell
--------------------
The pieces of the window's frame: line icons drawn from SVG paths in the
theme's colours, the navigation column (the analysis steps, comparisons and
the report, each with its state dot), a clickable chip, the app's mark and
the short notices that rise at the bottom of the window.
"""
from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtGui, QtSvg, QtWidgets

# 24 x 24 line icons (stroke only).
ICONS = {
    "signal": "M2 12h4l2-5 4 10 2-5h8",
    "heart": "M20.8 5.6a5.5 5.5 0 0 0-7.8 0L12 6.6l-1-1a5.5 5.5 0 0 0-7.8 7.8L12 22l8.8-8.6"
             "a5.5 5.5 0 0 0 0-7.8z",
    "trend": "M3 17l6-6 4 4 8-8M15 7h6v6",
    "beat": "M3 14h5l1.5-3 2 7 2.5-13 2 9h5",
    "alert": "M12 3l10 18H2zM12 10v4M12 17h.01",
    "split": "M4 5h7v14H4zM13 5h7v14h-7z",
    "group": "M8 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM16 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM2 20c0-3 "
             "2.7-5 6-5s6 2 6 5M14 15c3.3 0 6 2 6 5",
    "report": "M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8zM14 3v5h5M9 13h6M9 17h6",
    "move": "M5 9l-3 3 3 3M9 5l3-3 3 3M15 19l-3 3-3-3M19 9l3 3-3 3M2 12h20M12 2v20",
    "edit": "M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z",
    "noise": "M3 5h18v14H3z",
    "undo": "M9 14L4 9l5-5M4 9h11a5 5 0 0 1 0 10h-3",
    "redo": "M15 14l5-5-5-5M20 9H9a5 5 0 0 0 0 10h3",
    "sliders": "M4 7h10M18 7h2M4 17h4M12 17h8M16 5v4M10 15v4",
    "more": "M5 12h.01M12 12h.01M19 12h.01",
    "open": "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z",
    "chevron_left": "M15 6l-6 6 6 6",
    "chevron_right": "M9 6l6 6-6 6",
    "chevron_down": "M6 9l6 6 6-6",
    "sun": "M12 4V2M12 22v-2M4 12H2M22 12h-2M5.6 5.6 4.2 4.2M19.8 19.8l-1.4-1.4M5.6 18.4 4.2 "
           "19.8M19.8 4.2l-1.4 1.4M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8z",
    "moon": "M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z",
}


def icon_pixmap(name: str, color: str, size: int = 18, width: float = 2.0) -> QtGui.QPixmap:
    """The line icon *name* in *color*, *size* px (device pixel ratio 2)."""
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
           f'stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
           f'stroke-linejoin="round"><path d="{ICONS[name]}"/></svg>')
    ratio = 2
    pix = QtGui.QPixmap(size * ratio, size * ratio)
    pix.fill(QtCore.Qt.GlobalColor.transparent)
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode()))
    painter = QtGui.QPainter(pix)
    renderer.render(painter)
    painter.end()
    pix.setDevicePixelRatio(ratio)
    return pix


def icon(name: str, color: str, checked_color: "Optional[str]" = None,
         size: int = 18) -> QtGui.QIcon:
    """A QIcon of *name*: *color*, and *checked_color* when its button is on."""
    ic = QtGui.QIcon()
    ic.addPixmap(icon_pixmap(name, color, size), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
    ic.addPixmap(icon_pixmap(name, checked_color or color, size), QtGui.QIcon.Mode.Normal,
                 QtGui.QIcon.State.On)
    return ic


def logo_pixmap(color: str, size: int = 26) -> QtGui.QPixmap:
    """The app's mark: an ECG stroke on a rounded square of *color*."""
    ratio = 2
    pix = QtGui.QPixmap(size * ratio, size * ratio)
    pix.fill(QtCore.Qt.GlobalColor.transparent)
    pix.setDevicePixelRatio(ratio)
    painter = QtGui.QPainter(pix)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setBrush(QtGui.QColor(color))
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.drawRoundedRect(QtCore.QRectF(0, 0, size, size), size * 0.27, size * 0.27)
    inner = icon_pixmap("signal", "#FFFFFF", int(size * 0.66), 2.6)
    off = (size - size * 0.66) / 2
    painter.drawPixmap(QtCore.QPointF(off, off), inner)
    painter.end()
    return pix


def dot_pixmap(color: "Optional[str]", ring: str, size: int = 8) -> QtGui.QPixmap:
    """A state dot: filled with *color*, or an empty ring of *ring*."""
    ratio = 2
    pix = QtGui.QPixmap(size * ratio, size * ratio)
    pix.fill(QtCore.Qt.GlobalColor.transparent)
    pix.setDevicePixelRatio(ratio)
    painter = QtGui.QPainter(pix)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    if color:
        painter.setBrush(QtGui.QColor(color))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawEllipse(QtCore.QRectF(0.5, 0.5, size - 1, size - 1))
    else:
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor(ring), 1.4))
        painter.drawEllipse(QtCore.QRectF(1, 1, size - 2, size - 2))
    painter.end()
    return pix


class ClickableFrame(QtWidgets.QFrame):
    """A frame that acts as a button (the file chip)."""
    clicked = QtCore.Signal()

    def __init__(self, parent: "Optional[QtWidgets.QWidget]" = None, **kw) -> None:
        super().__init__(parent, **kw)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton and self.rect().contains(
                ev.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(ev)

    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:
        if ev.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter,
                        QtCore.Qt.Key.Key_Space):
            self.clicked.emit()
            return
        super().keyPressEvent(ev)


class Segmented(QtWidgets.QFrame):
    """A segmented control: exclusive (or not) checkable buttons in a trough."""

    def __init__(self, labels: "list[str]", exclusive: bool = True, tool: bool = False,
                 parent: "Optional[QtWidgets.QWidget]" = None) -> None:
        super().__init__(parent, objectName="Segmented")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(exclusive)
        self.buttons: "list[QtWidgets.QPushButton]" = []
        for text in labels:
            b = QtWidgets.QPushButton(text)
            b.setCheckable(True)
            if tool:
                b.setProperty("tool", True)
            lay.addWidget(b)
            self.group.addButton(b)
            self.buttons.append(b)

    def add_button(self, button: QtWidgets.QPushButton, tool: bool = False) -> None:
        """Put an existing checkable button in the trough."""
        if tool:
            button.setProperty("tool", True)
        self.layout().addWidget(button)
        self.group.addButton(button)
        self.buttons.append(button)


class NavItem(QtWidgets.QPushButton):
    """One entry of the navigation column: icon, name, state dot."""

    def __init__(self, text: str, icon_name: str, parent=None) -> None:
        super().__init__(text.replace("&", "&&"), parent)      # "&" alone: a shortcut mark
        self.icon_name = icon_name
        self.setProperty("nav", True)
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setIconSize(QtCore.QSize(17, 17))
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 10, 0)
        lay.addStretch(1)
        self.dot = QtWidgets.QLabel()
        self.dot.setVisible(False)
        lay.addWidget(self.dot, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)


class Sidebar(QtWidgets.QFrame):
    """The navigation column: groups of entries, each bound to a page."""
    selected = QtCore.Signal(str)              # the page key

    def __init__(self, groups: "list[tuple[str, list[tuple[str, str, str]]]]",
                 parent=None) -> None:
        """*groups*: [(group title, [(page key, label, icon name), ...]), ...]."""
        super().__init__(parent, objectName="Sidebar")
        self.setFixedWidth(220)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(2)
        self.items: "dict[str, NavItem]" = {}
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        for title, entries in groups:
            head = QtWidgets.QLabel(title)
            head.setProperty("role", "navgroup")
            lay.addWidget(head)
            for key, label, icon_name in entries:
                item = NavItem(label, icon_name)
                item.clicked.connect(lambda _c=False, k=key: self.selected.emit(k))
                self._group.addButton(item)
                lay.addWidget(item)
                self.items[key] = item
        lay.addStretch(1)
        self.footer = QtWidgets.QVBoxLayout()
        self.footer.setSpacing(6)
        lay.addLayout(self.footer)

    def select(self, key: str) -> None:
        item = self.items.get(key)
        if item is not None and not item.isChecked():
            item.setChecked(True)

    def set_state(self, key: str, color: "Optional[str]", ring: str, tip: str) -> None:
        item = self.items.get(key)
        if item is None:
            return
        item.dot.setPixmap(dot_pixmap(color, ring))
        item.dot.setVisible(True)
        item.setToolTip(tip)

    def apply_palette(self, p) -> None:
        for item in self.items.values():
            item.setIcon(icon(item.icon_name, p.TEXT, p.ACCENT_TEXT, 17))


class Toast(QtWidgets.QFrame):
    """A short notice at the bottom of the window (a warning, an error, a
    result discarded): readable, and gone by itself after a few seconds."""

    def __init__(self, window: QtWidgets.QWidget) -> None:
        super().__init__(window, objectName="Toast")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(16, 10, 10, 10)
        self.label = QtWidgets.QLabel("")
        self.label.setWordWrap(True)
        lay.addWidget(self.label, 1)
        close = QtWidgets.QPushButton("✕")
        close.setProperty("flat", True)
        close.setFixedSize(24, 24)
        close.setStyleSheet("color: inherit; padding: 0;")
        close.clicked.connect(self.hide)
        lay.addWidget(close)
        self._timer = QtCore.QTimer(self, singleShot=True)
        self._timer.timeout.connect(self.hide)
        self.hide()

    def show_text(self, text: str, msecs: int = 6000) -> None:
        self.label.setText(text)
        self._place()
        self.show()
        self.raise_()
        self._timer.start(msecs)

    def _place(self) -> None:
        win = self.parentWidget()
        width = min(560, max(280, win.width() - 80))
        self.setFixedWidth(width)
        self.adjustSize()
        self.move((win.width() - width) // 2, win.height() - self.height() - 44)

    def text(self) -> str:
        return self.label.text() if self.isVisible() else ""



def page_header(title: str, subtitle: "Optional[QtWidgets.QLabel]" = None
                ) -> QtWidgets.QHBoxLayout:
    """A page's title (large) and the line under it; widgets added to the
    returned row land on its right."""
    row = QtWidgets.QHBoxLayout()
    row.setSpacing(12)
    col = QtWidgets.QVBoxLayout()
    col.setSpacing(0)
    lbl = QtWidgets.QLabel(title)
    lbl.setProperty("role", "title")
    col.addWidget(lbl)
    if subtitle is not None:
        subtitle.setProperty("muted", True)
        subtitle.setWordWrap(True)
        col.addWidget(subtitle)
    row.addLayout(col, 1)
    return row


def card(title: "Optional[str]" = None, widget: "Optional[QtWidgets.QWidget]" = None,
         min_height: int = 0, extra: "Optional[QtWidgets.QWidget]" = None) -> QtWidgets.QFrame:
    """A card: a heading (and *extra* on its right), then *widget*."""
    frame = QtWidgets.QFrame(objectName="Card")
    lay = QtWidgets.QVBoxLayout(frame)
    lay.setContentsMargins(16, 12, 16, 12)
    lay.setSpacing(8)
    if title or extra is not None:
        head = QtWidgets.QHBoxLayout()
        if title:
            lbl = QtWidgets.QLabel(title)
            lbl.setProperty("role", "heading")
            head.addWidget(lbl)
        head.addStretch(1)
        if extra is not None:
            head.addWidget(extra)
        lay.addLayout(head)
    if widget is not None:
        if min_height:
            widget.setMinimumHeight(min_height)
        lay.addWidget(widget, 1)
    return frame


class CommandPalette(QtWidgets.QDialog):
    """Search every command and page by name (Ctrl+K / ⌘K): type, then
    Enter (or a click) runs the first match."""

    def __init__(self, parent: QtWidgets.QWidget,
                 commands: "callable") -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Popup)
        self.setObjectName("Palette")
        self._commands = commands            # () -> [(name, hint, fn), ...]
        self.setFixedWidth(520)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        self.edit = QtWidgets.QLineEdit()
        self.edit.setPlaceholderText("Search commands and pages")
        self.edit.textChanged.connect(self._filter)
        self.edit.returnPressed.connect(self._run_current)
        lay.addWidget(self.edit)
        self.list = QtWidgets.QListWidget()
        self.list.setMinimumHeight(300)
        self.list.itemActivated.connect(lambda _i: self._run_current())
        self.list.itemClicked.connect(lambda _i: self._run_current())
        lay.addWidget(self.list)
        self.edit.installEventFilter(self)

    def open_over(self, anchor: QtWidgets.QWidget) -> None:
        self.edit.clear()
        self._filter("")
        pos = anchor.mapToGlobal(QtCore.QPoint(anchor.width() - self.width(), anchor.height() + 6))
        self.move(pos)
        self.show()
        self.edit.setFocus()

    def _filter(self, text: str) -> None:
        words = text.lower().split()
        self.list.clear()
        for name, hint, fn in self._commands():
            hay = f"{name} {hint}".lower()
            if all(w in hay for w in words):
                item = QtWidgets.QListWidgetItem(f"{name}" + (f"    {hint}" if hint else ""))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, fn)
                self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)

    def eventFilter(self, obj, ev) -> bool:
        if obj is self.edit and ev.type() == QtCore.QEvent.Type.KeyPress:
            key = ev.key()
            if key in (QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_Up):
                row = self.list.currentRow() + (1 if key == QtCore.Qt.Key.Key_Down else -1)
                self.list.setCurrentRow(max(0, min(self.list.count() - 1, row)))
                return True
        return super().eventFilter(obj, ev)

    def _run_current(self) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        fn = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self.hide()
        if callable(fn):
            fn()
