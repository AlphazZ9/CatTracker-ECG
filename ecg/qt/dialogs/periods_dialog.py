# -*- coding: utf-8 -*-
"""
ecg.qt.dialogs.periods_dialog
-----------------------------
The annotation and noisy-region managers: a table of the recording's
periods with Add / Edit / Delete (ecg/ui/dialogs.py AnnotationManagerDialog
and NoiseRegionManagerDialog).

The checks and the stored dicts are those of the Tk dialogs: start < end,
inside the recording (0 to its end + 1 s), times rounded to 0.1 ms;
annotations are {t_start, t_end, label, color}, noisy regions
{t_start, t_end, note}. A new period starts as the visible detail window.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from PySide6 import QtGui, QtWidgets

from ecg.io.export import ANNOTATION_COLORS

if TYPE_CHECKING:
    from ecg.qt.main_window import MainWindow


def validate_period(t0_text: str, t1_text: str, t_max: float) -> "tuple[Optional[tuple], str]":
    """AnnotationDialog._ok / NoiseRegionDialog._ok's checks: ((t0, t1), "")
    or (None, the message)."""
    try:
        t0, t1 = float(t0_text), float(t1_text)
    except ValueError:
        return None, "Start and end must be numbers (seconds)."
    if t0 >= t1:
        return None, "Start must be less than end."
    if t0 < 0 or t1 > t_max + 1:
        return None, f"Times are outside the recording (0 – {t_max:.1f} s)."
    return (round(t0, 4), round(t1, 4)), ""


class PeriodDialog(QtWidgets.QDialog):
    """Add or edit one period."""

    def __init__(self, parent: QtWidgets.QWidget, kind: str, t_max: float,
                 existing: "Optional[dict]" = None,
                 default: "Optional[tuple[float, float]]" = None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.t_max = t_max
        self.result_period: "Optional[dict]" = None
        noun = "annotation" if kind == "annotation" else "noisy region"
        self.setWindowTitle(f"{'Add' if existing is None else 'Edit'} {noun}")
        form = QtWidgets.QFormLayout(self)
        start = (existing or {}).get("t_start", default[0] if default else None)
        end = (existing or {}).get("t_end", default[1] if default else None)
        self.ed_start = QtWidgets.QLineEdit("" if start is None else f"{start:.3f}")
        self.ed_end = QtWidgets.QLineEdit("" if end is None else f"{end:.3f}")
        form.addRow("Start (s):", self.ed_start)
        form.addRow("End (s):", self.ed_end)
        text_key = "label" if kind == "annotation" else "note"
        self.ed_text = QtWidgets.QLineEdit((existing or {}).get(text_key, ""))
        form.addRow("Label:" if kind == "annotation" else "Note:", self.ed_text)
        self.cb_color = QtWidgets.QComboBox()
        if kind == "annotation":
            for hex_col, name in ANNOTATION_COLORS:
                pix = QtGui.QPixmap(14, 14)
                pix.fill(QtGui.QColor(hex_col))
                self.cb_color.addItem(QtGui.QIcon(pix), name, hex_col)
            current = (existing or {}).get("color", ANNOTATION_COLORS[0][0])
            idx = self.cb_color.findData(current)
            self.cb_color.setCurrentIndex(max(idx, 0))
            form.addRow("Colour:", self.cb_color)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._ok)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _ok(self) -> None:
        span, message = validate_period(self.ed_start.text(), self.ed_end.text(), self.t_max)
        if span is None:
            QtWidgets.QMessageBox.warning(self, "Invalid period", message)
            return
        period = {"t_start": span[0], "t_end": span[1]}
        if self.kind == "annotation":
            period["label"] = self.ed_text.text().strip()
            period["color"] = self.cb_color.currentData()
        else:
            period["note"] = self.ed_text.text().strip()
        self.result_period = period
        self.accept()


class PeriodsManager(QtWidgets.QDialog):
    """The list of annotations (kind "annotation") or noisy regions
    (kind "noise") of the current recording."""

    def __init__(self, window: "MainWindow", kind: str,
                 on_changed: "Callable[[], None]") -> None:
        super().__init__(window)
        self.win = window
        self.kind = kind
        self._on_changed = on_changed
        noun = "annotation" if kind == "annotation" else "noisy region"
        self.setWindowTitle("Annotations" if kind == "annotation" else "Noisy regions")
        self.resize(720 if kind == "annotation" else 560, 400)
        lay = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton(f"Add {noun}")
        self.btn_add.setProperty("primary", True)
        self.btn_edit = QtWidgets.QPushButton("Edit selected")
        self.btn_delete = QtWidgets.QPushButton("Delete selected")
        for b in (self.btn_add, self.btn_edit, self.btn_delete):
            bar.addWidget(b)
        bar.addStretch(1)
        self.suggest_dialog = None
        if kind == "annotation":
            self.btn_suggest = QtWidgets.QPushButton("Suggest from channels…")
            self.btn_suggest.setToolTip("Annotations from the file's marker and keyboard "
                                        "channels and from the stimulus pulses of its other "
                                        "channels (Spike2 .smr or MATLAB export): you pick "
                                        "which to add")
            self.btn_suggest.clicked.connect(self.suggest)
            bar.addWidget(self.btn_suggest)
        lay.addLayout(bar)
        cols = ["Start (s)", "End (s)", "Duration", "Label" if kind == "annotation" else "Note"]
        self.table = QtWidgets.QTableWidget(0, len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.doubleClicked.connect(lambda *_a: self.edit())
        lay.addWidget(self.table, 1)
        self.btn_add.clicked.connect(self.add)
        self.btn_edit.clicked.connect(self.edit)
        self.btn_delete.clicked.connect(self.delete)
        self.refresh()

    # ── the recording's list ────────────────────────────────
    def periods(self) -> "list[dict]":
        a = self.win.recording.analysis
        return a.annotations if self.kind == "annotation" else a.noise_regions

    def _t_max(self) -> float:
        t = self.win.recording.signal.time
        return float(t[-1]) if t is not None and len(t) else 1e6

    def refresh(self) -> None:
        rows = self.periods()
        self.table.setRowCount(len(rows))
        key = "label" if self.kind == "annotation" else "note"
        for i, p in enumerate(rows):
            texts = (f"{p['t_start']:.3f}", f"{p['t_end']:.3f}",
                     f"{p['t_end'] - p['t_start']:.3f} s", p.get(key, ""))
            for c, text in enumerate(texts):
                item = QtWidgets.QTableWidgetItem(text)
                if c == 3 and self.kind == "annotation" and p.get("color"):
                    pix = QtGui.QPixmap(12, 12)
                    pix.fill(QtGui.QColor(p["color"]))
                    item.setIcon(QtGui.QIcon(pix))
                self.table.setItem(i, c, item)
        self.table.resizeColumnsToContents()

    def selected(self) -> "list[int]":
        return sorted({i.row() for i in self.table.selectionModel().selectedRows()})

    def _changed(self) -> None:
        self.win.session_dirty = True
        self._on_changed()
        self.refresh()

    # ── Add / Edit / Delete ─────────────────────────────────
    def add(self) -> None:
        default = None
        if self.win.recording.signal.time is not None:
            t0, t1 = self.win.detection.plot.window()
            default = (t0, min(self._t_max(), t1))
        dlg = PeriodDialog(self, self.kind, self._t_max(), default=default)
        if dlg.exec() and dlg.result_period is not None:
            self.periods().append(dlg.result_period)
            self._changed()

    def edit(self) -> None:
        sel = self.selected()
        if len(sel) != 1:
            QtWidgets.QMessageBox.information(self, "Selection",
                                              "Select exactly one row to edit.")
            return
        idx = sel[0]
        dlg = PeriodDialog(self, self.kind, self._t_max(), existing=dict(self.periods()[idx]))
        if dlg.exec() and dlg.result_period is not None:
            self.periods()[idx] = dlg.result_period
            self._changed()

    def suggest(self) -> None:
        from ecg.qt.dialogs.suggest_dialog import open_suggestions

        def _add(anns: "list[dict]") -> None:
            periods = self.periods()
            periods.extend(anns)
            periods.sort(key=lambda a: float(a["t_start"]))
            self._changed()
        self.suggest_dialog = open_suggestions(self, self.win, _add)

    def delete(self) -> None:
        sel = self.selected()
        if not sel:
            QtWidgets.QMessageBox.information(self, "Selection", "Select the rows to delete.")
            return
        answer = QtWidgets.QMessageBox.question(self, "Confirm delete",
                                                f"Delete {len(sel)} row(s)?")
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        for i in reversed(sel):
            del self.periods()[i]
        self._changed()
