# -*- coding: utf-8 -*-
"""
ecg.qt.theme
------------
Light and dark palettes of the Qt interface -- the colour values of the Tk
interface's presets (ecg/ui/theme.py, ThemeConfig.PRESETS), so both look the
same -- and the application style sheet built from them.

A theme change restyles the running window (no rebuild, unlike the Tk
interface). Plot colours are read from the palette by the widgets.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

from PySide6 import QtGui, QtWidgets


@dataclass(frozen=True)
class Palette:
    name: str
    is_dark: bool
    BG: str
    PANEL: str
    CARD: str
    BORDER: str
    BORDER2: str
    TEXT: str
    MUTED: str
    LIGHT: str
    RED: str
    BLUE: str
    GREEN: str
    ORANGE: str
    plot_signal: str
    plot_rpeak: str
    plot_filtered: str
    STRIP_BG: str
    HAIRLINE: str
    ACCENT_TEXT: str
    WARN_TEXT: str


LIGHT = Palette(
    name="Light", is_dark=False,
    BG="#F4F5F7", PANEL="#FFFFFF", CARD="#FFFFFF",
    BORDER="#DCE0E5", BORDER2="#C7CDD4",
    TEXT="#1B222C", MUTED="#5B6472", LIGHT="#939DA8",
    RED="#D64545", BLUE="#2D6CDF", GREEN="#2E9E5B", ORANGE="#E08A2A",
    plot_signal="#2D6CDF", plot_rpeak="#2E9E5B", plot_filtered="#6A1B9A",
    STRIP_BG="#EEF3FC", HAIRLINE="#EEF0F3", ACCENT_TEXT="#1344A8", WARN_TEXT="#9A5A12",
)

DARK = Palette(
    name="Dark", is_dark=True,
    BG="#14171C", PANEL="#1B1F26", CARD="#1B1F26",
    BORDER="#2B3138", BORDER2="#3A4149",
    TEXT="#E7EAEE", MUTED="#9BA5B2", LIGHT="#656E79",
    RED="#E5695F", BLUE="#5B8DF5", GREEN="#4CBA7C", ORANGE="#E8A552",
    plot_signal="#5B8DF5", plot_rpeak="#4CBA7C", plot_filtered="#9125D2",
    STRIP_BG="#1C2635", HAIRLINE="#232831", ACCENT_TEXT="#8FB2F8", WARN_TEXT="#E8B36A",
)

PALETTES = {LIGHT.name: LIGHT, DARK.name: DARK}
_current: Palette = LIGHT


def current() -> Palette:
    """The palette in use."""
    return _current


def system_prefers_dark() -> bool:
    """True when the operating system is in dark mode (Qt >= 6.5)."""
    app = QtGui.QGuiApplication.instance()
    if app is None:
        return False
    try:
        from PySide6.QtCore import Qt
        return app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except AttributeError:
        return False


def style_sheet(p: Palette) -> str:
    """The application style sheet for palette *p*."""
    return f"""
    QWidget {{ background: {p.BG}; color: {p.TEXT}; font-size: 10pt; }}
    QMainWindow, QDialog {{ background: {p.BG}; }}
    QFrame#Header, QFrame#Card {{ background: {p.CARD}; border: 1px solid {p.BORDER};
        border-radius: 6px; }}
    QLabel {{ background: transparent; }}
    QLabel[muted="true"] {{ color: {p.MUTED}; }}
    QLabel[warn="true"] {{ color: {p.WARN_TEXT}; }}
    QPushButton {{ background: {p.CARD}; border: 1px solid {p.BORDER2}; border-radius: 5px;
        padding: 4px 12px; }}
    QPushButton:hover {{ border-color: {p.BLUE}; }}
    QPushButton:disabled {{ color: {p.LIGHT}; border-color: {p.BORDER}; }}
    QPushButton[primary="true"] {{ background: {p.BLUE}; color: white; border-color: {p.BLUE}; }}
    QPushButton[primary="true"]:disabled {{ background: {p.BORDER2}; border-color: {p.BORDER2}; }}
    QTabWidget::pane {{ border: 1px solid {p.BORDER}; background: {p.PANEL}; top: -1px; }}
    QTabBar::tab {{ background: {p.BG}; color: {p.MUTED}; padding: 6px 14px;
        border: 1px solid {p.BORDER}; border-bottom: none; margin-right: 2px;
        border-top-left-radius: 5px; border-top-right-radius: 5px; }}
    QTabBar::tab:selected {{ background: {p.PANEL}; color: {p.ACCENT_TEXT}; }}
    QComboBox, QLineEdit, QDoubleSpinBox, QSpinBox {{ background: {p.CARD};
        border: 1px solid {p.BORDER2}; border-radius: 4px; padding: 2px 6px; }}
    QCheckBox {{ background: transparent; spacing: 6px; }}
    QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {p.BORDER2};
        border-radius: 3px; background: {p.CARD}; }}
    QCheckBox::indicator:checked {{ background: {p.BLUE}; border-color: {p.BLUE}; }}
    QProgressBar {{ background: {p.HAIRLINE}; border: 1px solid {p.BORDER}; border-radius: 4px;
        text-align: center; height: 14px; }}
    QProgressBar::chunk {{ background: {p.BLUE}; border-radius: 3px; }}
    QStatusBar {{ background: {p.CARD}; color: {p.MUTED}; }}
    """


def apply(app: QtWidgets.QApplication, name: "str | None" = None) -> Palette:
    """Apply the palette *name* ("Light" / "Dark"; None: the system's) to
    *app* and return it."""
    global _current
    if name is None:
        name = DARK.name if system_prefers_dark() else LIGHT.name
    _current = PALETTES[name]
    app.setStyleSheet(style_sheet(_current))
    return _current


def palette_fields() -> "list[str]":
    """The colour tokens of a palette (for tests and tools)."""
    return [f.name for f in fields(Palette) if f.name not in ("name", "is_dark")]
