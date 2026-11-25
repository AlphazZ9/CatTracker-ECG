# -*- coding: utf-8 -*-
"""
ecg.qt.figure_host
------------------
Lets the Qt interface draw with the Tk interface's own figure code
(ecg.app.plot_controller.PlotController), so a figure is the same in both
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
    """A CanvasSlot (ecg.app.plots) as PlotController uses it."""

    def update(self, draw_fn: Callable) -> None:          # noqa: A003 (CanvasSlot API)
        self.update_figure(draw_fn)

    @property
    def fig(self):                                        # CanvasSlot.fig (exports)
        return self.figure

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


class Field(Value):
    """A Qt input read and written with the Tk calls the controllers use
    (get, set, delete(0, "end") / insert(0, v), select / deselect,
    configure). *data*: a combobox's get() / set() use the item data
    instead of the text; *as_int*: a check box reads 1 / 0 (a Tk IntVar)."""

    def __init__(self, widget: QtWidgets.QWidget, data: bool = False,
                 as_int: bool = False, text: bool = False) -> None:
        self.widget = widget
        self._data = data
        self._as_int = as_int
        self._text = text           # a spin box read as its Tk entry's text ("20")
        super().__init__(self._read)

    def _read(self) -> Any:
        w = self.widget
        if isinstance(w, QtWidgets.QLineEdit):
            return w.text()
        if isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            return f"{w.value():g}" if self._text else w.value()
        if isinstance(w, QtWidgets.QAbstractButton):
            on = w.isChecked()
            return int(on) if self._as_int else on
        if isinstance(w, QtWidgets.QComboBox):
            return w.currentData() if self._data else w.currentText()
        raise TypeError(type(w).__name__)

    def set(self, value: Any) -> None:
        w = self.widget
        if isinstance(w, QtWidgets.QLineEdit):
            w.setText("" if value is None else str(value))
        elif isinstance(w, QtWidgets.QSpinBox):
            w.setValue(int(round(float(value))))
        elif isinstance(w, QtWidgets.QDoubleSpinBox):
            w.setValue(float(value))
        elif isinstance(w, QtWidgets.QAbstractButton):
            w.setChecked(bool(value))
        elif isinstance(w, QtWidgets.QComboBox):
            i = w.findData(value) if self._data else -1
            if i < 0:
                i = w.findText(str(value))
            if i < 0 and not self._data:
                i = w.findData(value)
            if i >= 0:
                w.setCurrentIndex(i)
        else:
            raise TypeError(type(w).__name__)

    def delete(self, *_a: Any) -> None:
        if isinstance(self.widget, QtWidgets.QLineEdit):
            self.widget.clear()

    def insert(self, _index: Any, value: Any) -> None:
        if isinstance(self.widget, QtWidgets.QLineEdit):
            self.widget.setText(str(value) + self.widget.text())
        else:
            try:
                self.set(value)
            except (TypeError, ValueError):
                pass

    def select(self) -> None:
        self.set(True)

    def deselect(self) -> None:
        self.set(False)

    def configure(self, **kw: Any) -> None:
        """Only a combobox's values= (its choices) changes anything."""
        values = kw.get("values")
        if values is not None and isinstance(self.widget, QtWidgets.QComboBox):
            cur = self.widget.currentText()
            self.widget.blockSignals(True)
            self.widget.clear()
            self.widget.addItems([str(v) for v in values])
            i = self.widget.findText(cur)
            self.widget.setCurrentIndex(max(i, 0))
            self.widget.blockSignals(False)


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


class _MainThreadCaller(QtCore.QObject):
    """Runs a callable on the main thread, whichever thread asks (the Tk
    controllers' workers report progress through app.after(0, fn))."""
    call = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.call.connect(lambda fn: fn(), QtCore.Qt.ConnectionType.QueuedConnection)


class QtFileDialog:
    """tkinter.filedialog for the reused Tk controllers: Qt file dialogs."""

    @staticmethod
    def _filter(filetypes: Any) -> str:
        parts = []
        for name, pattern in filetypes or ():
            pats = pattern if isinstance(pattern, str) else " ".join(pattern)
            parts.append(f"{name} ({pats})")
        return ";;".join(parts) or "All files (*)"

    @classmethod
    def asksaveasfilename(cls, title: str = "Save", initialfile: str = "",
                          initialdir: str = "", defaultextension: str = "",
                          filetypes: Any = None, **_k: Any) -> str:
        import os
        start = os.path.join(initialdir or "", initialfile or "")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            QtWidgets.QApplication.activeWindow(), title, start, cls._filter(filetypes))
        if path and defaultextension and not os.path.splitext(path)[1]:
            path += defaultextension
        return path

    @classmethod
    def askopenfilename(cls, title: str = "Open", initialdir: str = "",
                        filetypes: Any = None, **_k: Any) -> str:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            QtWidgets.QApplication.activeWindow(), title, initialdir or "", cls._filter(filetypes))
        return path

    @staticmethod
    def askdirectory(title: str = "Choose a folder", initialdir: str = "", **_k: Any) -> str:
        return QtWidgets.QFileDialog.getExistingDirectory(
            QtWidgets.QApplication.activeWindow(), title, initialdir or "")


class QtMessageBox:
    """tkinter.messagebox for the Tk controllers running under Qt: the same
    calls, Qt boxes (no Tk window is ever opened)."""

    @staticmethod
    def _box(icon, title: str, message: str) -> None:
        box = QtWidgets.QMessageBox(icon, str(title), str(message),
                                    parent=QtWidgets.QApplication.activeWindow())
        box.open()

    @classmethod
    def showwarning(cls, title: str = "", message: str = "", **_k: Any) -> str:
        cls._box(QtWidgets.QMessageBox.Icon.Warning, title, message)
        return "ok"

    @classmethod
    def showerror(cls, title: str = "", message: str = "", **_k: Any) -> str:
        cls._box(QtWidgets.QMessageBox.Icon.Critical, title, message)
        return "ok"

    @classmethod
    def showinfo(cls, title: str = "", message: str = "", **_k: Any) -> str:
        cls._box(QtWidgets.QMessageBox.Icon.Information, title, message)
        return "ok"

    @staticmethod
    def askyesno(title: str = "", message: str = "", **_k: Any) -> bool:
        return QtWidgets.QMessageBox.question(
            QtWidgets.QApplication.activeWindow(), str(title), str(message)) \
            == QtWidgets.QMessageBox.StandardButton.Yes


def _use_qt_message_boxes() -> None:
    """The controllers' message boxes and file choosers (ecg.app.dialogs)
    are Qt ones."""
    from ecg.app import dialogs
    dialogs.install(QtMessageBox, QtFileDialog)


class FigureHost:
    """What PlotController / AnalysisController read on their "app"."""

    def __init__(self, window: Any) -> None:
        from ecg.app.state import UIState
        from ecg.app.text_tables import TextTables
        self._window = window
        self.ui = UIState()
        self._slots: "dict[str, FigureSlot]" = _Slots()
        self._kpi: dict = {}
        self._generation = 0
        self._async_busy = False
        self._df_to_tsv = TextTables._df_to_tsv
        self._df_to_text = TextTables._df_to_text
        self._NL_DISPLAY = TextTables._NL_DISPLAY
        self._safe_float = lambda w, d: _safe_float(w, d)
        from ecg.app.analysis_controller import AnalysisController
        from ecg.app.plot_controller import PlotController
        _use_qt_message_boxes()
        self._main_caller = _MainThreadCaller()

        # The metric table and the events list go to the tabs' callbacks
        # (on_rhythm_table / on_events_list), set below.
        self.analysis_ctrl = AnalysisController(self)
        self.plot_ctrl = PlotController(self)
        from ecg.app.navigation_controller import NavigationController
        self.nav_ctrl = NavigationController(self)
        from ecg.app.export_controller import ExportController
        self.export_ctrl = ExportController(self)
        # What each module was computed on (up to date / out of date), as Tk.
        from ecg.app.freshness import FreshnessService
        self.freshness = FreshnessService(self)
                                     # (Qt drops them on every peak change)
        self._arr_card_widgets: list = []
        self.on_events_list: "Optional[Callable]" = None
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
    def _qtc_formula(self) -> str:
        return self.analysis.qtc_formula

    def clipboard_clear(self) -> None:
        QtWidgets.QApplication.clipboard().clear()

    def clipboard_append(self, text: str) -> None:
        cb = QtWidgets.QApplication.clipboard()
        cb.setText(cb.text() + str(text))

    @property
    def _arrhythmia_tsv(self) -> str:
        """ECGApp._arrhythmia_tsv: the events as TSV (Copy for Excel)."""
        return self.analysis.arrhythmia_tsv

    def _select_arrhythmia_event(self, idx: int) -> None:
        self.nav_ctrl.select_arrhythmia_event(idx)

    def _draw_arr_detail(self) -> None:
        self.plot_ctrl.draw_arr_detail()

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
        self._window.set_progress(int(pct), msg)

    def after(self, ms: int, fn: "Optional[Callable]" = None) -> None:
        if fn is None:
            return
        app = QtCore.QCoreApplication.instance()
        if app is not None and QtCore.QThread.currentThread() is not app.thread():
            self._main_caller.call.emit(fn)        # from a worker thread: queued
        else:
            QtCore.QTimer.singleShot(int(ms), fn)

    def _start_async_result(self, button: Any, busy_label: str, worker: Callable[[], Any],
                            on_done: Callable[[Any], None]) -> None:
        """AsyncMixin._start_async_result: *worker()* in a background job,
        *on_done(result)* on the main thread."""
        self._window.run_job(busy_label.rstrip("…").strip() or "Computing", worker,
                             progress=False, peaks_bound=True, on_done=on_done)

    @property
    def _peak_distance_ms(self) -> float:
        """The recording's peak distance (the Tk app's property of the same name)."""
        return self._window.recording.signal.peak_distance_ms

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
    from ecg.app import theme as tk_theme
    # THEME itself (in memory only, never saved from here): some figures read
    # THEME.is_dark (e.g. the PSD band transparency).
    tk_theme.THEME.apply_preset("Dark" if is_dark else "Light")
    tk_theme.apply_theme_config(tk_theme.THEME)
