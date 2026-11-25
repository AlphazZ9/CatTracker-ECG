# -*- coding: utf-8 -*-
"""
ecg.qt.theme
------------
Light and dark palettes of the interface (red and white), its typeface
(Inter, shipped with the app, figures of equal width) and the application
style sheet built from them.

A theme change restyles the running window. Plot colours are read from the
palette by the widgets; the Matplotlib figures keep their own colours (those
of the exports), only their background follows the theme.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, fields

from PySide6 import QtGui, QtWidgets


@dataclass(frozen=True)
class Palette:
    name: str
    is_dark: bool
    BG: str            # the window
    PANEL: str         # pages
    CARD: str          # cards, fields, buttons
    SIDEBAR: str       # the navigation column
    BORDER: str
    BORDER2: str
    TEXT: str
    MUTED: str
    LIGHT: str
    ACCENT: str        # the one action colour: primary button, R peaks, heart rate
    ACCENT_SOFT: str   # its tint: selection, the dobutamine-like bands
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
    SEGMENT_BG: str    # the trough of a segmented control
    WARN_BG: str       # an out-of-date banner


# Red and white, after the platform's own light / dark system colours: the
# ECG trace in near-black ink, the R peaks and the heart rate in red, the rest
# in neutral greys. Green / orange only mark a state (up to date / out of date).
LIGHT = Palette(
    name="Light", is_dark=False,
    BG="#FFFFFF", PANEL="#FFFFFF", CARD="#FFFFFF", SIDEBAR="#F5F5F7",
    BORDER="#E5E5EA", BORDER2="#D1D1D6",
    TEXT="#1D1D1F", MUTED="#636366", LIGHT="#AEAEB2",
    ACCENT="#D70015", ACCENT_SOFT="#FDECEE",
    RED="#D70015", BLUE="#0A66D6", GREEN="#248A3D", ORANGE="#E08600",
    plot_signal="#1D1D1F", plot_rpeak="#D70015", plot_filtered="#6A1B9A",
    STRIP_BG="#FDECEE", HAIRLINE="#F2F2F7", ACCENT_TEXT="#B0000F", WARN_TEXT="#9A5000",
    SEGMENT_BG="#EFEFF4", WARN_BG="#FFF4E5",
)

DARK = Palette(
    name="Dark", is_dark=True,
    BG="#1C1C1E", PANEL="#1C1C1E", CARD="#1C1C1E", SIDEBAR="#252527",
    BORDER="#38383A", BORDER2="#48484A",
    TEXT="#F5F5F7", MUTED="#98989D", LIGHT="#636366",
    ACCENT="#FF453A", ACCENT_SOFT="#3D1F1E",
    RED="#FF453A", BLUE="#409CFF", GREEN="#30D158", ORANGE="#FF9F0A",
    plot_signal="#F2F2F7", plot_rpeak="#FF453A", plot_filtered="#BF5AF2",
    STRIP_BG="#3D1F1E", HAIRLINE="#2C2C2E", ACCENT_TEXT="#FF6961", WARN_TEXT="#FFB340",
    SEGMENT_BG="#3A3A3C", WARN_BG="#3A2B12",
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
    # The spin box and combo box arrows and the check mark: drawn from the
    # app's own images, since a styled box without them shows broken arrows.
    up = (ASSETS / "chevron_up.svg").as_posix()
    down = (ASSETS / "chevron_down.svg").as_posix()
    check = (ASSETS / "check.svg").as_posix()
    pt = 10 * scale
    return f"""
    QWidget {{ background: {p.BG}; color: {p.TEXT}; font-size: {pt:.1f}pt; }}
    QMainWindow, QDialog {{ background: {p.BG}; }}
    QToolTip {{ background: {p.TEXT}; color: {p.BG}; border: none; padding: 6px 8px;
        border-radius: 6px; }}

    QFrame#Header {{ background: {p.BG}; border: none; border-bottom: 1px solid {p.BORDER}; }}
    QFrame#Sidebar {{ background: {p.SIDEBAR}; border: none;
        border-right: 1px solid {p.BORDER}; }}
    QFrame#Sidebar QLabel {{ background: transparent; }}
    QFrame#Card {{ background: {p.CARD}; border: 1px solid {p.BORDER}; border-radius: 12px; }}
    QFrame#Card QLabel, QFrame#Card QCheckBox {{ background: transparent; }}
    QFrame#Hero {{ background: {p.ACCENT}; border: none; border-radius: 12px; }}
    QFrame#Hero QLabel {{ background: transparent; color: white; }}
    QFrame#Segmented {{ background: {p.SEGMENT_BG}; border: none; border-radius: 8px; }}
    QFrame#Toast {{ background: {p.TEXT}; border: none; border-radius: 10px; }}
    QFrame#Toast QLabel {{ background: transparent; color: {p.BG}; }}
    QFrame#Banner {{ background: {p.ACCENT_SOFT}; border: none; border-radius: 10px; }}
    QFrame#Warning {{ background: {p.WARN_BG}; border: none; border-radius: 10px; }}
    QFrame#Warning QLabel {{ background: transparent; color: {p.WARN_TEXT}; }}
    QFrame#Chip {{ background: {p.SIDEBAR}; border: none; border-radius: 8px; }}
    QFrame#Chip:hover {{ background: {p.BORDER}; }}
    QFrame#Chip QLabel {{ background: transparent; }}

    QLabel {{ background: transparent; }}
    QLabel[muted="true"] {{ color: {p.MUTED}; }}
    QLabel[warn="true"] {{ color: {p.WARN_TEXT}; }}
    QLabel[role="title"] {{ font-size: {pt * 2.0:.1f}pt; font-weight: 700; }}
    QLabel[role="heading"] {{ font-size: {pt * 1.15:.1f}pt; font-weight: 600; }}
    QLabel[role="big"] {{ font-size: {pt * 3.6:.1f}pt; font-weight: 700; color: {p.ACCENT}; }}
    QLabel[role="unit"] {{ font-size: {pt * 1.3:.1f}pt; font-weight: 600; color: {p.ACCENT}; }}
    QLabel[role="value"] {{ font-size: {pt * 1.9:.1f}pt; font-weight: 700; }}
    QLabel#Placeholder {{ color: {p.MUTED}; padding: 24px; }}

    QPushButton, QToolButton {{ background: {p.CARD}; color: {p.TEXT};
        border: 1px solid {p.BORDER2}; border-radius: 8px; padding: 5px 12px; }}
    QPushButton:hover, QToolButton:hover {{ background: {p.HAIRLINE}; }}
    QPushButton:pressed, QToolButton:pressed {{ background: {p.BORDER}; }}
    QPushButton:checked {{ background: {p.ACCENT_SOFT}; color: {p.ACCENT_TEXT};
        border-color: {p.ACCENT}; }}
    QPushButton:disabled, QToolButton:disabled {{ color: {p.LIGHT}; border-color: {p.BORDER}; }}
    QPushButton[primary="true"] {{ background: {p.ACCENT}; color: white; border: none;
        font-weight: 600; padding: 6px 16px; }}
    QPushButton[primary="true"]:hover {{ background: {p.ACCENT_TEXT}; }}
    QPushButton[primary="true"]:disabled {{ background: {p.BORDER2}; color: {p.BG}; }}
    QPushButton[action="true"] {{ background: {p.ACCENT_SOFT}; color: {p.ACCENT_TEXT};
        border: none; font-weight: 600; }}
    QPushButton[action="true"]:hover {{ background: {p.BORDER}; }}
    QPushButton[action="true"]:disabled {{ background: {p.HAIRLINE}; color: {p.LIGHT}; }}
    QPushButton[flat="true"], QToolButton[flat="true"] {{ background: transparent;
        border: none; }}
    QPushButton[flat="true"]:hover, QToolButton[flat="true"]:hover {{
        background: {p.HAIRLINE}; }}
    QPushButton[chip="true"] {{ background: {p.SIDEBAR}; border: none; }}
    QPushButton[chip="true"]:hover {{ background: {p.BORDER}; }}
    QToolButton::menu-indicator {{ image: url("{down}"); width: 9px; height: 9px;
        subcontrol-position: right center; right: 6px; }}
    QToolButton[popupMode="2"] {{ padding-right: 22px; }}

    QPushButton[nav="true"] {{ background: transparent; border: none; border-radius: 8px;
        text-align: left; padding: 7px 10px; color: {p.TEXT}; }}
    QPushButton[nav="true"]:hover {{ background: {p.BORDER}; }}
    QPushButton[nav="true"]:checked {{ background: {p.CARD}; color: {p.ACCENT_TEXT};
        font-weight: 600; border: 1px solid {p.BORDER}; }}
    QLabel[role="navgroup"] {{ color: {p.MUTED}; font-size: {pt * 0.85:.1f}pt;
        font-weight: 600; padding: 12px 10px 4px 10px; }}

    QFrame#Segmented QPushButton {{ background: transparent; border: none;
        border-radius: 6px; padding: 4px 11px; color: {p.MUTED}; }}
    QFrame#Segmented QPushButton:hover {{ color: {p.TEXT}; }}
    QFrame#Segmented QPushButton:checked {{ background: {p.CARD}; color: {p.TEXT};
        font-weight: 600; border: 1px solid {p.BORDER}; }}
    QFrame#Segmented QPushButton[tool="true"]:checked {{ background: {p.ACCENT};
        color: white; border: none; }}

    QTabWidget::pane {{ border: none; background: {p.PANEL}; }}
    QTabBar {{ background: transparent; }}
    QTabBar::tab {{ background: {p.SEGMENT_BG}; color: {p.MUTED}; padding: 5px 14px;
        border: none; margin: 0; }}
    QTabBar::tab:first {{ border-top-left-radius: 8px; border-bottom-left-radius: 8px; }}
    QTabBar::tab:last {{ border-top-right-radius: 8px; border-bottom-right-radius: 8px; }}
    QTabBar::tab:selected {{ background: {p.CARD}; color: {p.TEXT}; font-weight: 600;
        border: 1px solid {p.BORDER}; border-radius: 7px; }}
    QTabWidget#TimeCourse > QTabBar::tab {{ margin-top: 22px; margin-bottom: 6px; }}

    QComboBox, QLineEdit, QDoubleSpinBox, QSpinBox, QPlainTextEdit, QTextEdit {{
        background: {p.CARD}; border: 1px solid {p.BORDER2}; border-radius: 7px;
        padding: 3px 8px; selection-background-color: {p.ACCENT_SOFT};
        selection-color: {p.TEXT}; }}
    QComboBox:focus, QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
        border-color: {p.ACCENT}; }}
    QPlainTextEdit, QTextEdit {{ border-color: {p.BORDER}; }}
    QAbstractSpinBox {{ padding-right: 20px; }}
    QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ subcontrol-origin: border;
        width: 18px; border: none; border-left: 1px solid {p.BORDER}; background: transparent; }}
    QAbstractSpinBox::up-button {{ subcontrol-position: top right;
        border-top-right-radius: 7px; }}
    QAbstractSpinBox::down-button {{ subcontrol-position: bottom right;
        border-bottom-right-radius: 7px; }}
    QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
        background: {p.HAIRLINE}; }}
    QAbstractSpinBox::up-arrow {{ image: url("{up}"); width: 9px; height: 9px; }}
    QAbstractSpinBox::down-arrow {{ image: url("{down}"); width: 9px; height: 9px; }}
    QComboBox {{ padding-right: 22px; }}
    QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right;
        width: 22px; border: none; }}
    QComboBox::down-arrow {{ image: url("{down}"); width: 9px; height: 9px; }}
    QComboBox QAbstractItemView {{ background: {p.CARD}; border: 1px solid {p.BORDER};
        selection-background-color: {p.ACCENT_SOFT}; selection-color: {p.TEXT};
        outline: none; }}

    QCheckBox, QRadioButton {{ background: transparent; spacing: 7px; }}
    QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid {p.BORDER2};
        border-radius: 4px; background: {p.CARD}; }}
    QCheckBox::indicator:checked {{ background: {p.ACCENT}; border-color: {p.ACCENT};
        image: url("{check}"); }}
    QCheckBox::indicator:disabled {{ background: {p.HAIRLINE}; }}
    QRadioButton::indicator {{ width: 15px; height: 15px; border: 1px solid {p.BORDER2};
        border-radius: 8px; background: {p.CARD}; }}
    QRadioButton::indicator:checked {{ border: 5px solid {p.ACCENT}; }}

    QSlider::groove:horizontal {{ height: 4px; background: {p.BORDER}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {p.ACCENT}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ width: 16px; height: 16px; margin: -7px 0;
        border-radius: 8px; background: white; border: 1px solid {p.BORDER2}; }}

    QProgressBar {{ background: {p.BORDER}; border: none; border-radius: 3px;
        max-height: 6px; text-align: center; color: transparent; }}
    QProgressBar::chunk {{ background: {p.ACCENT}; border-radius: 3px; }}

    QTableWidget, QTableView, QTreeView, QListView {{ background: {p.CARD};
        border: 1px solid {p.BORDER}; border-radius: 8px; gridline-color: {p.HAIRLINE};
        selection-background-color: {p.ACCENT_SOFT}; selection-color: {p.TEXT};
        alternate-background-color: {p.SIDEBAR}; }}
    QHeaderView {{ background: transparent; }}
    QHeaderView::section {{ background: {p.CARD}; color: {p.MUTED}; border: none;
        border-bottom: 1px solid {p.BORDER}; padding: 5px 8px; font-weight: 600; }}
    QTableCornerButton::section {{ background: {p.CARD}; border: none; }}

    QScrollArea {{ border: none; background: {p.PANEL}; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {p.BORDER2}; border-radius: 3px;
        min-height: 30px; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: {p.BORDER2}; border-radius: 3px;
        min-width: 30px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    QMenuBar {{ background: {p.BG}; }}
    QMenuBar::item:selected {{ background: {p.HAIRLINE}; border-radius: 4px; }}
    QMenu {{ background: {p.CARD}; border: 1px solid {p.BORDER}; padding: 4px;
        border-radius: 8px; }}
    QMenu::item {{ padding: 6px 22px 6px 12px; border-radius: 5px; }}
    QMenu::item:selected {{ background: {p.ACCENT_SOFT}; color: {p.TEXT}; }}
    QMenu::separator {{ height: 1px; background: {p.BORDER}; margin: 4px 6px; }}

    QSplitter::handle {{ background: {p.BORDER}; }}
    QStatusBar {{ background: {p.BG}; color: {p.MUTED}; border-top: 1px solid {p.BORDER}; }}
    QStatusBar::item {{ border: none; }}
    """


FONT_FILES = ("Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf", "Inter-Bold.ttf")
_font_family: "str | None" = None


def load_font() -> "str | None":
    """Register the shipped Inter faces once; their family name, or None
    when they could not be loaded (the system's font is then used)."""
    global _font_family
    if _font_family is None:
        families: "list[str]" = []
        for name in FONT_FILES:
            fid = QtGui.QFontDatabase.addApplicationFont(str(ASSETS / "fonts" / name))
            if fid >= 0:
                families += QtGui.QFontDatabase.applicationFontFamilies(fid)
        _font_family = families[0] if families else ""
    return _font_family or None


def app_font() -> QtGui.QFont:
    """Inter (else the system's sans serif), figures of equal width so the
    numbers do not jump as they change."""
    family = load_font()
    font = QtGui.QFont(family) if family else QtGui.QFont()
    font.setStyleHint(QtGui.QFont.StyleHint.SansSerif)
    try:
        font.setFeature(QtGui.QFont.Tag("tnum"), 1)
    except (AttributeError, TypeError):           # Qt < 6.7: proportional figures
        pass
    return font


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
    font = app_font()
    font.setPointSizeF(10 * _font_scale)
    app.setFont(font)
    app.setStyleSheet(style_sheet(_current, _font_scale))
    return _current


def palette_fields() -> "list[str]":
    """The colour tokens of a palette (for tests and tools)."""
    return [f.name for f in fields(Palette) if f.name not in ("name", "is_dark")]
