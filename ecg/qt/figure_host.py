# -*- coding: utf-8 -*-
"""
ecg.qt.figure_host
------------------
Lets the Qt interface draw with the Tk interface's own figure code
(ecg.ui.plot_controller.PlotController), so a figure is the same in both
interfaces and in the exports, without a second copy of 2,700 lines of
drawing code.

PlotController reads a handful of attributes of its "app": the recording's
state, figure slots (``_slots[name].update(draw_fn)``), result text boxes,
two status labels, the detail window width and Tk timers. ``FigureHost``
provides them on top of Qt widgets: a slot is an MplCanvas, a text box a
QPlainTextEdit, a timer a QTimer. No Tk window is ever created (the Tk-side
modules are imported, not run).

Transitional: before the CustomTkinter interface is removed (phase 4), the
figure code moves out of ecg.ui into a UI-free package; this host then
points at it.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from ecg.qt.widgets.mpl_canvas import MplCanvas

log = logging.getLogger("ecg")


class FigureSlot(MplCanvas):
    """A CanvasSlot (ecg.ui.plots) as PlotController uses it."""

    def update(self, draw_fn: Callable) -> None:          # noqa: A003 (CanvasSlot API)
        self.update_figure(draw_fn)

    def _show_placeholder(self, force: bool = False) -> None:
        self.clear()

    def enable_wheel_yzoom(self) -> None:                 # Qt: the toolbar-free canvas
        pass

    def flush(self) -> None:
        pass


class TextBox(QtWidgets.QPlainTextEdit):
    """A result text box (CTkTextbox): read-only, monospace, with a
    "Copy as TSV" menu entry when the text has a table behind it."""

    def __init__(self, parent: "QtWidgets.QWidget | None" = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.tsv: "Optional[str]" = None

    def contextMenuEvent(self, ev: QtGui.QContextMenuEvent) -> None:
        menu = self.createStandardContextMenu()
        if self.tsv:
            act = menu.addAction("Copy as table (paste into Excel)")
            act.triggered.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.tsv))
        menu.exec(ev.globalPos())


class StatusLabel(QtWidgets.QLabel):
    """A CTkLabel's configure(text=, text_color=)."""

    def configure(self, text: "Optional[str]" = None, text_color: "Optional[str]" = None,
                  **_kw: Any) -> None:
        if text is not None:
            self.setText(text)
        if text_color is not None:
            self.setStyleSheet(f"color: {text_color};")


class Value:
    """An entry or slider read with .get()."""

    def __init__(self, getter: Callable[[], Any]) -> None:
        self._getter = getter

    def get(self) -> Any:
        return self._getter()


class _Null:
    """Any Tk-only widget PlotController may touch (KPI tiles, the Detection
    bar...): every attribute is itself, every call returns None."""

    def __getattr__(self, name: str) -> "_Null":
        if name.startswith("__"):
            raise AttributeError(name)
        return self

    def __call__(self, *a: Any, **k: Any) -> None:
        return None

    def __getitem__(self, key: Any) -> "_Null":
        return self

    def __iter__(self):
        return iter(())


_NULL = _Null()


class _Slots(dict):
    """Figure slots by name; a figure whose tab is not built yet still gets
    a (hidden) slot, so drawing it never fails."""

    def __missing__(self, name: str) -> FigureSlot:
        slot = self[name] = FigureSlot(None)
        return slot


class FigureHost:
    """What PlotController / AnalysisController read on their "app"."""

    def __init__(self, window: Any) -> None:
        from ecg.ui.app_helpers import HelpersMixin
        from ecg.ui.state import UIState
        self._window = window
        self.ui = UIState()
        self._slots: "dict[str, FigureSlot]" = _Slots()
        self._kpi: dict = {}
        self._generation = 0
        self._async_busy = False
        self._df_to_tsv = HelpersMixin._df_to_tsv
        self._df_to_text = HelpersMixin._df_to_text
        self._NL_DISPLAY = HelpersMixin._NL_DISPLAY
        self._safe_float = lambda w, d: _safe_float(w, d)
        from ecg.ui.analysis_controller import AnalysisController
        from ecg.ui.plot_controller import PlotController

        class _QtPlotController(PlotController):
            """The Rhythm & HRV metric table is a Qt table: the shared table
            (ecg.core.metric_table) is handed to the host's callback instead of
            being written into a Tk text box."""

            def refresh_rhythm_table(self) -> None:
                t = getattr(self, "_rhythm_table", None)
                cb = self.app.__dict__.get("on_rhythm_table")
                if t is not None and cb is not None:
                    cb(t)

            def fit_rhythm_table(self) -> None:
                pass

        self.analysis_ctrl = AnalysisController(self)
        self.plot_ctrl = _QtPlotController(self)
        self.on_rhythm_table: "Optional[Callable]" = None
        self.txt_rhythm_metrics = None          # no Tk text box (see above)

    # The recording's state, always the window's current one.
    @property
    def signal(self):
        return self._window.recording.signal

    @property
    def detection(self):
        return self._window.recording.detection

    @property
    def analysis(self):
        return self._window.recording.analysis

    def slot(self, name: str, parent: "QtWidgets.QWidget | None" = None,
             width_in: float = 6.0, height_in: float = 3.5) -> FigureSlot:
        if name not in self._slots:
            dict.__setitem__(self._slots, name, FigureSlot(parent, width_in, height_in))
        elif parent is not None and self._slots[name].parent() is None:
            self._slots[name].setParent(parent)
        return self._slots[name]

    # ── Tk app methods PlotController / AnalysisController call ──
    def _set_textbox(self, widget: Any, text: str, tsv: "Optional[str]" = None) -> None:
        if isinstance(widget, TextBox):
            widget.setPlainText(text)
            widget.tsv = tsv

    def _windowed_peaks(self):
        d, s, a = self.detection, self.signal, self.analysis
        if d.rpeaks_ok is None or s.fs is None:
            return None
        rp = d.rpeaks_ok.copy()
        if a.t_start <= 0 and a.t_end <= 0:
            return rp
        mask = rp / s.fs >= a.t_start
        if a.t_end > 0:
            mask &= rp / s.fs <= a.t_end
        return rp[mask]

    def _set_status(self, text: str, color: "Optional[str]" = None) -> None:
        self._window.set_status(text.strip())

    def _set_progress(self, pct: int, msg: str) -> None:
        self._window._progress(int(pct), msg)

    def after(self, ms: int, fn: "Optional[Callable]" = None) -> None:
        if fn is not None:
            QtCore.QTimer.singleShot(int(ms), fn)

    def after_idle(self, fn: "Optional[Callable]" = None) -> None:
        if fn is not None:
            QtCore.QTimer.singleShot(0, fn)

    def after_cancel(self, _id: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        # Tk-only widgets (KPI tiles, buttons, the Detection bar...): no-ops.
        if name.startswith("__"):
            raise AttributeError(name)
        return _NULL


def _safe_float(widget: Any, default: float) -> float:
    try:
        return float(widget.get())
    except (TypeError, ValueError, AttributeError):
        return default


def sync_figure_theme(is_dark: bool) -> None:
    """Give the shared figure code the Qt theme (Light / Dark preset of the
    Tk ThemeConfig: the same colours)."""
    from ecg.ui import theme as tk_theme
    # THEME itself (in memory only, never saved from here): some figures read
    # THEME.is_dark (e.g. the PSD band transparency).
    tk_theme.THEME.apply_preset("Dark" if is_dark else "Light")
    tk_theme.apply_theme_config(tk_theme.THEME)
