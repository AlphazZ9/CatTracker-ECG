# -*- coding: utf-8 -*-
"""
ecg.qt.widgets.emoji
--------------------
An emoji beside the label of the app's buttons (📂 Open, 🔍 Detect Peaks,
⚡ Analyze all, 💾 Save Session...), as an icon: drawn once with the
system's colour emoji font (Segoe UI Emoji on Windows, Apple Color Emoji on
macOS, Noto Color Emoji on Linux), so every button lines its emoji up with
its text the same way, whatever the font of the text. Without an emoji font
the buttons keep their plain labels.

The emojis are chosen by the button's label (EMOJI below); a button is
decorated when Qt first polishes it (EmojiDecorator, installed once on the
application), so the buttons of the dialogs and the rows built later get
theirs too. A button that already has an icon (the toolbars) or no known
label is left alone; ``button.setProperty("emoji", False)`` opts one out.
"""
from __future__ import annotations

import re
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

# The label (as normalise() reads it) -> its emoji. Labels starting with one
# of the PREFIXES take that prefix's emoji when they have none of their own.
EMOJI: "dict[str, str]" = {
    # header
    "Open": "📂",
    "Detect Peaks": "🔍",
    "Detect peaks": "🔍",
    "Analyze all": "⚡",
    "Save Session": "💾",
    "Export": "📤",
    "Undo": "↩️",
    "Redo": "↪️",
    "Settings": "⚙️",
    "Dark mode": "🌙",
    "Light mode": "☀️",
    # Signal & Detection
    "Detection": "🎛️",
    "Move": "✋",
    "Edit peaks": "✏️",
    "Edit peaks here": "✏️",
    "Mark noise": "🖍️",
    "Clear edits": "🧹",
    "RR · HR · Quality": "📈",
    "Annotations": "🏷️",
    "Noisy regions": "〰️",
    "Review artifacts": "🩺",
    "Review": "🩺",
    "Check agreement": "🤝",
    "Channels of the file": "📡",
    "ML detector": "🤖",
    "Apply": "✅",
    "Full": "↔️",
    "Whole signal": "↔️",
    "Go": "🎯",
    "Reset zoom": "🔄",
    "Refresh": "🔄",
    "Recompute": "🔁",
    "Recompute non-linear": "🔁",
    "Recompute frequency": "🔁",
    "Load": "📥",
    "View": "👁️",
    # Morphology & Intervals
    "Measure intervals": "📏",
    "Landmarks": "📌",
    "Calibrate ks / kp": "🎯",
    # Events
    "Classify events": "🔎",
    "Previous": "⬅️",
    "Next": "➡️",
    "Copy for Excel": "📋",
    # Time course, Compare segments
    "Compute": "🧮",
    "Compute epochs": "🧮",
    "Compute sliding window": "🧮",
    "Compute event-locked": "🧮",
    "Compare": "⚖️",
    "Compare segments": "⚖️",
    "Figure": "🖼️",
    "Figures": "🖼️",
    "Save figure": "🖼️",
    "Save statistics": "💾",
    "Save as": "💾",
    "Save notes": "📝",
    "Save .txt": "📝",
    # Group, Report, exports
    "Open the Group tab": "👥",
    "Cohort Trends": "📈",
    "Save to cohort registry": "🗄️",
    "Recording": "📝",
    "Search": "🔎",
    "Keep all": "✅",
    "Edit selected": "✏️",
    "Copy table": "📋",
    "Copy results": "📋",
    "Show the text tables": "📄",
    "Export Excel": "📊",
    "Export Excel / CSV": "📊",
    "Export CSV": "📊",
    "Export cohort CSV": "📊",
    "GraphPad Prism": "📈",
    "Prism": "📈",
    "Prism (new file / template)": "📈",
    "PDF report": "📄",
    "Train model": "🧠",
    "Suggest from channels": "💡",
    "Copy text": "📋",
    "Events (PDF)": "📄",
    "Figures (ZIP)": "🗜️",
    "RR series (CSV)": "📊",
    "Prism… (new file / template)": "📈",
    "Open Excel": "📂",
    "From saved sessions": "🗂️",
    "Reload": "🔄",
    "Save as Excel": "💾",
    "Reset to mouse ECG defaults": "🐭",
}
PREFIXES: "tuple[tuple[str, str], ...]" = (
    ("Add ", "➕"),
    ("Remove", "🗑️"),
    ("Delete", "🗑️"),
)
# Arrows some labels carry in place of an icon: dropped once the emoji is there.
_ARROWS = re.compile(r"^[↶↷↻→]\s*")
_EMOJI_FONTS = ("Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji",
                "Twemoji Mozilla", "EmojiOne Color", "JoyPixels")
ICON_PX = 16                # on the button
_DRAWN_PX = 64              # drawn at, then scaled (sharp on high-DPI screens)


def normalise(text: str) -> str:
    """The label without its arrows, its keyboard mnemonic, a trailing
    ellipsis or arrow, and spaces at the ends."""
    t = _ARROWS.sub("", (text or "").replace("&&", "\0").replace("&", "").replace("\0", "&"))
    t = t.strip()
    for tail in ("…", "...", "→"):
        if t.endswith(tail):
            t = t[: -len(tail)].strip()
    return t


def emoji_for(text: str) -> "Optional[str]":
    """The emoji of a button labelled *text*, or None."""
    t = normalise(text)
    if not t:
        return None
    if t in EMOJI:
        return EMOJI[t]
    for prefix, emoji in PREFIXES:
        if t.startswith(prefix):
            return emoji
    return None


_FONT: "dict[str, Optional[str]]" = {}
_ICONS: "dict[str, QtGui.QIcon]" = {}


def emoji_font() -> "Optional[str]":
    """The system's colour emoji font family, or None."""
    if "family" not in _FONT:
        families = set(QtGui.QFontDatabase.families())
        _FONT["family"] = next((f for f in _EMOJI_FONTS if f in families), None)
    return _FONT["family"]


def emoji_icon(emoji: str) -> "Optional[QtGui.QIcon]":
    """*emoji* as an icon (cached), or None without an emoji font."""
    if emoji in _ICONS:
        return _ICONS[emoji]
    family = emoji_font()
    if family is None:
        return None
    pm = QtGui.QPixmap(_DRAWN_PX, _DRAWN_PX)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pm)
    painter.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing
                           | QtGui.QPainter.RenderHint.TextAntialiasing
                           | QtGui.QPainter.RenderHint.SmoothPixmapTransform)
    font = QtGui.QFont(family)
    font.setPixelSize(int(_DRAWN_PX * 0.86))
    painter.setFont(font)
    painter.drawText(pm.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, emoji)
    painter.end()
    icon = QtGui.QIcon(pm)
    _ICONS[emoji] = icon
    return icon


def decorate(button: QtWidgets.QAbstractButton) -> bool:
    """Give *button* the emoji of its label (True if it got one); again
    after its label changed (e.g. Dark mode / Light mode). Buttons with an
    icon of their own, check boxes and radio buttons are left alone."""
    if isinstance(button, (QtWidgets.QCheckBox, QtWidgets.QRadioButton)):
        return False
    ours = bool(button.property("emoji_set"))
    if button.property("emoji") is False or (not ours and not button.icon().isNull()):
        return False
    emoji = emoji_for(button.text())
    icon = emoji_icon(emoji) if emoji else None
    if icon is None:
        if ours:                                # its new label has none
            button.setIcon(QtGui.QIcon())
            button.setProperty("emoji_set", False)
        return False
    text = button.text()
    stripped = _ARROWS.sub("", text)
    if stripped != text:
        button.setText(stripped)
    button.setIcon(icon)
    button.setIconSize(QtCore.QSize(ICON_PX, ICON_PX))
    if isinstance(button, QtWidgets.QToolButton):
        # A tool button shows its icon alone by default: the label stays.
        button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    button.setProperty("emoji_set", True)
    return True


def decorate_all(root: QtWidgets.QWidget) -> int:
    """Decorate every button under *root* (the number decorated)."""
    n = 0
    for b in root.findChildren(QtWidgets.QAbstractButton):
        n += decorate(b)
    return n


class EmojiDecorator(QtCore.QObject):
    """On the application: each button gets its emoji when Qt first polishes
    it (before it is shown), wherever it was built."""

    _POLISH = QtCore.QEvent.Type.Polish

    def eventFilter(self, obj: QtCore.QObject, ev: QtCore.QEvent) -> bool:  # noqa: N802
        if ev.type() == self._POLISH and isinstance(obj, QtWidgets.QAbstractButton):
            decorate(obj)
        return False


_DECORATOR: "dict[str, EmojiDecorator]" = {}


def install(app: "Optional[QtWidgets.QApplication]" = None) -> bool:
    """Decorate the buttons of the application from now on (once); False
    without an emoji font."""
    app = app or QtWidgets.QApplication.instance()
    if app is None or emoji_font() is None:
        return False
    if "app" not in _DECORATOR:
        _DECORATOR["app"] = EmojiDecorator(app)
        app.installEventFilter(_DECORATOR["app"])
    return True
