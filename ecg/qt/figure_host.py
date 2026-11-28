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


# What an empty figure says, by slot: which step fills it.
PLACEHOLDERS = {
    "rr": "Detect the peaks, then click Analyze: the RR interval and heart rate of every "
          "beat appear here.",
    "poincare": "Click Analyze to see the Poincaré plot.",
    "psd": "Click Analyze to see the power spectrum.",
    "rr_hist": "Click Analyze to see the RR distribution.",
    "asymmetry": "Click Analyze to see the RR asymmetry.",
    "beat": "Click Analyze: the mean beat and its landmarks appear here.",
    "intervals": "Click Measure intervals: PR, QRS and the other intervals of every beat "
                 "appear here.",
    "qtc_rr": "Click Measure intervals to see QTc against RR.",
    "beat_dist": "Click Measure intervals to see the beat distributions.",
    "epochs": "Click Compute epochs: the metrics ticked above, window by window, through "
              "the recording.",
    "rolling_hrv": "Click Compute: heart rate variability in a window sliding along the "
                   "recording.",
    "autonomic": "Click Compute: the sympathetic (CSI) and parasympathetic (CPI) indices "
                 "through the recording.",
    "event_locked": "Label a period in Signal & Detection (Annotations), then click Compute: "
                    "the indices before and after each event.",
    "arr_ribbon": "Click Classify events: the episodes appear here along the recording.",
    "arr_detail": "Select an event in the list to see the ECG around it.",
    "compare": "Set the segments above, then click Compare.",
    "quality_time": "Click Analyze: each beat's correlation with the template, over time.",
    "quality_hist": "Click Analyze to see the distribution.",
}


class _DrawQueue:
    """Figures waiting to be drawn, one per turn of the event loop: a tab
    of seven figures stays responsive (and paints) while they come in,
    instead of freezing until the last one is drawn."""

    def __init__(self) -> None:
        self._queue: "list[FigureSlot]" = []

    def add(self, slot: "FigureSlot") -> None:
        if slot not in self._queue:
            self._queue.append(slot)
            if len(self._queue) == 1:
                QtCore.QTimer.singleShot(0, self._next)

    def _next(self) -> None:
        if not self._queue:
            return
        slot = self._queue.pop(0)
        try:
            if slot.isVisible():
                slot.flush()
        except RuntimeError:                    # the widget was deleted meanwhile
            pass
        if self._queue:
            QtCore.QTimer.singleShot(0, self._next)


_DRAWS = _DrawQueue()


class FigureSlot(MplCanvas):
    """A CanvasSlot (ecg.app.plots) as PlotController uses it.

    Drawn when seen: a figure whose tab is hidden keeps its draw function
    and draws it when the tab is shown, or as soon as the figure is read
    (an export, a test). Analyze and a theme change therefore only draw
    the figures on screen, instead of every figure of every tab, and those
    one per turn of the event loop (_DrawQueue)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pending: "Optional[Callable]" = None
        # What else shows this figure's data on screen and empties with it
        # (the quality panel's PyQtGraph charts).
        self.on_clear: "Optional[Callable[[], None]]" = None

    def update(self, draw_fn: Callable) -> None:          # noqa: A003 (CanvasSlot API)
        self._draw_fn = draw_fn
        self._sync_placeholder()
        self._pending = draw_fn
        if self.isVisible():
            _DRAWS.add(self)

    @property
    def figure(self):
        self.flush()
        return self._figure

    @property
    def fig(self):                                        # CanvasSlot.fig (exports)
        return self.figure

    def flush(self) -> None:
        """Draw now what is waiting to be drawn."""
        if self._pending is not None:
            self._pending = None
            self.redraw()

    def retheme(self) -> None:
        """The figure again in the current theme: now if it is on screen,
        else when it is next shown."""
        if self._draw_fn is None:                         # empty: its background only
            import matplotlib
            self._figure.set_facecolor(matplotlib.rcParams["figure.facecolor"])
            self.canvas.draw_idle()
            return
        self._pending = self._draw_fn
        if self.isVisible():
            _DRAWS.add(self)

    def clear(self) -> None:
        self._pending = None
        super().clear()
        if self.on_clear is not None:
            self.on_clear()

    def showEvent(self, ev: QtGui.QShowEvent) -> None:
        super().showEvent(ev)
        if self._pending is not None:
            # After the tab itself has painted: it appears at once, its
            # figures one after the other.
            _DRAWS.add(self)

    def _show_placeholder(self, force: bool = False) -> None:
        self.clear()



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
        slot.set_placeholder(PLACEHOLDERS.get(name, ""))
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


class _NoButton:
    """A run button the Qt window does not need: anything done to it
    (configure, state…) does nothing."""

    def __getattr__(self, name: str) -> Callable[..., None]:
        return lambda *_a, **_k: None


_NO_BUTTON = _NoButton()


class FigureHost:
    """What PlotController / AnalysisController read on their "app"."""

    def __init__(self, window: Any) -> None:
        from ecg.app.state import UIState
        from ecg.app.text_tables import TextTables
        self._window = window
        self.ui = UIState()
        self._slots: "dict[str, FigureSlot]" = _Slots()
        self._screen_charts: "dict[str, Callable[[Any], None]]" = {}
        # cb_freq_band: no band-preset selector (Tk had none either), so
        # get_freq_bands() keeps the Mouse Thireau bands.
        self.cb_freq_band = None
        # The run buttons the controllers hand to _start_async_result (which
        # ignores them: Qt's jobs disable the actions). Not None: a None
        # button means "no tab built" and stops the run.
        self.btn_run_arrhythmia = _NO_BUTTON
        self.btn_compute_epochs = _NO_BUTTON
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

    def _refresh_morphology(self) -> None:
        """PlotController.plot_intervals: the Morphology steps and table."""
        morph = getattr(self._window, "morphology", None)
        if morph is not None:
            morph.refresh_banner()

    def _sync_arr_edit_btn(self) -> None:
        """The Events "Edit peaks here" button follows the mode (a new
        classification or a restored session)."""
        ev = getattr(self._window, "events", None)
        btn = getattr(ev, "btn_edit", None)
        if btn is not None:
            btn.blockSignals(True)
            on = bool(self.analysis.arr_edit_mode)
            btn.setChecked(on)
            btn.setText("Edit Mode ON" if on else "Edit peaks here")
            btn.blockSignals(False)

    def show_chart_data(self, name: str, data: Any) -> None:
        """The data of a figure to its on-screen PyQtGraph chart, when a tab
        registered one in _screen_charts (PlotController._on_screen)."""
        handler = self._screen_charts.get(name)
        if handler is not None:
            handler(data)

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
            self._slots[name].set_placeholder(PLACEHOLDERS.get(name, ""))
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
            table = getattr(widget, "table", None)
            if table is not None:
                fill_table(table, tsv)
                widget.setVisible(not tsv)
                table.setVisible(bool(tsv))

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

    def after(self, ms: int, fn: "Optional[Callable]" = None) -> "Optional[QtCore.QTimer]":
        """Tk's after: *fn* in *ms* ms on the main thread; the id returned
        cancels it (after_cancel), so the controllers' debounces (a burst of
        wheel ticks: one redraw) work as in Tk."""
        if fn is None:
            return None
        app = QtCore.QCoreApplication.instance()
        if app is not None and QtCore.QThread.currentThread() is not app.thread():
            self._main_caller.call.emit(fn)        # from a worker thread: queued
            return None
        parent = self._window if isinstance(self._window, QtCore.QObject) else None
        timer = QtCore.QTimer(parent)
        timer.setSingleShot(True)
        pending = self.__dict__.setdefault("_pending_timers", set())
        pending.add(timer)                          # alive until run or cancelled

        def run() -> None:
            pending.discard(timer)
            timer.deleteLater()
            fn()
        timer.timeout.connect(run)
        timer.start(int(ms))
        return timer

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
        if isinstance(_id, QtCore.QTimer):
            self.__dict__.setdefault("_pending_timers", set()).discard(_id)
            try:
                _id.stop()
                _id.deleteLater()
            except RuntimeError:                   # already run and gone
                pass

    def __getattr__(self, name: str) -> Any:
        # Tk-only widgets (KPI tiles, buttons, the Detection bar...): no-ops.
        # Each name is logged once, with where it was read: a Tk-era call that
        # does nothing in Qt shows in the log instead of passing unseen.
        if name.startswith("__"):
            raise AttributeError(name)
        seen = self.__dict__.setdefault("_missing_seen", set())
        if name not in seen:
            seen.add(name)
            import sys
            f = sys._getframe(1)
            log.info("FigureHost: '%s' is not provided in Qt (read at %s:%d); no-op",
                     name, f.f_code.co_filename.rsplit("/", 1)[-1], f.f_lineno)
        return _NULL


def pretty_column(name: str) -> str:
    """A table column's name for the eye: 'Epoch_start_s' -> 'Epoch start (s)',
    'N_beats' -> 'Beats', 'HR_mean' -> 'HR mean' (the exports keep theirs)."""
    import re
    n = str(name)
    unit = ""
    for suffix, u in (("_s", "s"), ("_ms", "ms"), ("_bpm", "bpm"), ("_pct", "%")):
        if n.endswith(suffix):
            n, unit = n[: -len(suffix)], u
            break
    if n.startswith("N_"):
        n = n[2:]
    n = re.sub(r"^HRV_", "", n).replace("_", " ")
    n = n[:1].upper() + n[1:]
    return f"{n} ({unit})" if unit else n


def fill_table(table: QtWidgets.QTableWidget, tsv: "Optional[str]") -> None:
    """A table from tab-separated text (header line first): sortable, numbers
    right-aligned; 'Copy' gives the TSV back."""
    rows = [r.split("\t") for r in (tsv or "").splitlines() if r.strip()]
    table.setSortingEnabled(False)
    if not rows:
        table.setRowCount(0)
        table.setColumnCount(0)
        return
    head, body = rows[0], rows[1:]
    table.setColumnCount(len(head))
    table.setHorizontalHeaderLabels([pretty_column(h) for h in head])
    table.setRowCount(len(body))
    for r, row in enumerate(body):
        for c, text in enumerate(row[:len(head)]):
            item = QtWidgets.QTableWidgetItem(text)
            try:
                item.setData(QtCore.Qt.ItemDataRole.DisplayRole, float(text))
                item.setText(text)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                                      | QtCore.Qt.AlignmentFlag.AlignVCenter)
            except ValueError:
                pass
            table.setItem(r, c, item)
    table.resizeColumnsToContents()
    table.setSortingEnabled(True)


def data_table(box: "TextBox") -> QtWidgets.QTableWidget:
    """A table twin of the text box *box*: shown instead of it when what
    fills it comes with a table (FigureHost._set_textbox)."""
    table = QtWidgets.QTableWidget(0, 0)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setAlternatingRowColors(True)
    table.setShowGrid(False)
    table.setVisible(False)
    table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.ActionsContextMenu)
    act = QtGui.QAction("Copy as table (paste into Excel)", table)
    act.triggered.connect(lambda: QtWidgets.QApplication.clipboard().setText(box.tsv or ""))
    table.addAction(act)
    box.table = table
    return table


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
    # The surfaces and the text of the Qt palette: the Tk dark preset is
    # bluish (#1B1F26), the Qt one neutral (#1C1C1E), and the figures sat on
    # the cards as darker blue blocks. The data colours stay the preset's.
    # Dark only: the light preset's surfaces are already the cards' white.
    if is_dark:
        from ecg.qt import theme as qt_theme
        for key in ("BG", "PANEL", "CARD", "BORDER", "BORDER2", "TEXT", "MUTED", "LIGHT",
                    "HAIRLINE"):
            tk_theme.THEME._colors[key] = getattr(qt_theme.DARK, key)
    tk_theme.apply_theme_config(tk_theme.THEME)
    _figures_in_app_font()


_APP_FONT_IN_FIGURES: "dict[str, bool]" = {}


def _figures_in_app_font() -> None:
    """The Matplotlib figures in the app's own font (the shipped Inter), as
    the PyQtGraph charts beside them: they were in DejaVu Sans. Glyphs Inter
    lacks fall back to Matplotlib's defaults. Once."""
    if _APP_FONT_IN_FIGURES:
        return
    _APP_FONT_IN_FIGURES["done"] = True
    try:
        import matplotlib
        from matplotlib import font_manager
        from ecg.qt import theme as qt_theme
        family = qt_theme.load_font()
        if not family:
            return
        for name in qt_theme.FONT_FILES:
            path = qt_theme.ASSETS / "fonts" / name
            if path.exists():
                font_manager.fontManager.addfont(str(path))
        names = {f.name for f in font_manager.fontManager.ttflist}
        if family not in names:
            return
        rest = [f for f in matplotlib.rcParams["font.sans-serif"] if f != family]
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [family] + rest
    except Exception:                       # the default font then: never an error
        log.debug("figures keep Matplotlib's font", exc_info=True)
