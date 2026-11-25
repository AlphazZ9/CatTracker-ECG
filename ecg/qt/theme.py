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

from pathlib import Path
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


# ── preferences: the palette chosen and the text size, kept between runs ──
ASSETS = Path(__file__).resolve().parents[1] / "app" / "assets"
PREFS_PATH = Path.home() / ".ecg_qt_prefs.json"
LEGACY_THEME_PATH = Path.home() / ".ecg_theme.json"     # the Tk interface's
FONT_SCALE_RANGE = (0.8, 1.6)
_font_scale = 1.0


def load_prefs() -> dict:
    """{"theme": "Light" / "Dark" / None (the system's), "font_scale"}: this
    interface's file, else the Tk interface's theme file (its light / dark
    and text size), else the defaults."""
    import json
    for path, legacy in ((PREFS_PATH, False), (LEGACY_THEME_PATH, True)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        if legacy:
            data = {"theme": DARK.name if data.get("is_dark") else LIGHT.name,
                    "font_scale": data.get("font_scale", 1.0)}
        name = data.get("theme")
        try:
            scale = float(data.get("font_scale", 1.0))
        except (TypeError, ValueError):
            scale = 1.0
        lo, hi = FONT_SCALE_RANGE
        return {"theme": name if name in PALETTES else None,
                "font_scale": min(hi, max(lo, scale))}
    return {"theme": None, "font_scale": 1.0}


def save_prefs() -> None:
    """Keep the palette in use and the text size for the next run."""
    import json
    try:
        PREFS_PATH.write_text(json.dumps({"theme": _current.name, "font_scale": _font_scale},
                                         indent=2), encoding="utf-8")
    except OSError as exc:
        import logging
        logging.getLogger("ecg").warning("preferences not saved: %s", exc)


def font_scale() -> float:
    return _font_scale


def style_sheet(p: Palette, scale: float = 1.0) -> str:
    """The application style sheet for palette *p* (text at *scale* x 10 pt)."""
    # The spin box and combo box arrows: drawn from the app's own images,
    # since a styled box without them shows broken arrows.
    up = (ASSETS / "chevron_up.svg").as_posix()
    down = (ASSETS / "chevron_down.svg").as_posix()
    return f"""
    QWidget {{ background: {p.BG}; color: {p.TEXT}; font-size: {10 * scale:.1f}pt; }}
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
    QAbstractSpinBox {{ padding-right: 20px; }}
    QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ subcontrol-origin: border;
        width: 18px; border: none; border-left: 1px solid {p.BORDER}; background: transparent; }}
    QAbstractSpinBox::up-button {{ subcontrol-position: top right;
        border-top-right-radius: 4px; }}
    QAbstractSpinBox::down-button {{ subcontrol-position: bottom right;
        border-bottom-right-radius: 4px; }}
    QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
        background: {p.HAIRLINE}; }}
    QAbstractSpinBox::up-arrow {{ image: url("{up}"); width: 9px; height: 9px; }}
    QAbstractSpinBox::down-arrow {{ image: url("{down}"); width: 9px; height: 9px; }}
    QComboBox {{ padding-right: 20px; }}
    QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right;
        width: 20px; border: none; }}
    QComboBox::down-arrow {{ image: url("{down}"); width: 9px; height: 9px; }}
    QCheckBox {{ background: transparent; spacing: 6px; }}
    QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {p.BORDER2};
        border-radius: 3px; background: {p.CARD}; }}
    QCheckBox::indicator:checked {{ background: {p.BLUE}; border-color: {p.BLUE}; }}
    QProgressBar {{ background: {p.HAIRLINE}; border: 1px solid {p.BORDER}; border-radius: 4px;
        text-align: center; height: 14px; }}
    QProgressBar::chunk {{ background: {p.BLUE}; border-radius: 3px; }}
    QStatusBar {{ background: {p.CARD}; color: {p.MUTED}; }}
    """


def apply(app: QtWidgets.QApplication, name: "str | None" = None,
          scale: "float | None" = None) -> Palette:
    """Apply the palette *name* ("Light" / "Dark"; None: the system's) and
    the text size *scale* (None: kept) to *app* and return the palette."""
    global _current, _font_scale
    if name is None:
        name = DARK.name if system_prefers_dark() else LIGHT.name
    if scale is not None:
        lo, hi = FONT_SCALE_RANGE
        _font_scale = min(hi, max(lo, float(scale)))
    _current = PALETTES[name]
    app.setStyleSheet(style_sheet(_current, _font_scale))
    return _current


def palette_fields() -> "list[str]":
    """The colour tokens of a palette (for tests and tools)."""
    return [f.name for f in fields(Palette) if f.name not in ("name", "is_dark")]
