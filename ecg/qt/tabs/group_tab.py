# -*- coding: utf-8 -*-
"""
ecg.qt.tabs.group_tab
---------------------
Group: the animals of one experiment, from a group table (.xlsx,
ecg.io.group_data), compared without groups, by group or by sex, on their
Pre/Post segments, their time course aligned on an annotation, their whole
recording or their event-locked results (ecg.io.group_compare); the figure
(ecg.ui.group_tab_figure), the results table and the statistics workbook.

The analyses, the table's reading / writing and Add open recording are
those of the Tk tab (ecg/ui/tab_group.py), with its defaults. Not yet
here: From saved sessions (a table in memory, from the project's saved
sessions) and the Prism exports (a new file or a template filled).
"""
from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING, Any, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from ecg.io import group_compare as gc
from ecg.io import group_data as gd
from ecg.io.subject_name import normalise_sex
from ecg.qt.figure_host import StatusLabel

if TYPE_CHECKING:
    from ecg.qt.main_window import MainWindow

ALL_EVENTS = "All events"
FIG_SIZE = (12.0, 4.6)


log = logging.getLogger("ecg")

class _Text:
    """A QLineEdit read as the Tk entry the tab's helpers take."""

    def __init__(self, edit: QtWidgets.QLineEdit) -> None:
        self.edit = edit

    def get(self) -> str:
        return self.edit.text()


class GroupTab(QtWidgets.QScrollArea):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.win = window
        self.setWidgetResizable(True)
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        self.setWidget(page)
        # Add open recording is the Tk tab's (TabGroupMixin), on the window's host.
        from ecg.ui.tab_group import TabGroupMixin
        host = window.figures
        host._group_lines_of_open_recording = (
            lambda comparison=None: TabGroupMixin._group_lines_of_open_recording(host, comparison))
        self.path = ""
        self.name = ""                  # a table from saved sessions: its project
        self.sheets: "Optional[dict]" = None
        self.animals = None
        self.res: "Optional[gc.GroupResult]" = None
        self._filling = False

        def card(title: str) -> QtWidgets.QVBoxLayout:
            f = QtWidgets.QFrame(objectName="Card")
            fl = QtWidgets.QVBoxLayout(f)
            fl.addWidget(QtWidgets.QLabel(f"<b>{title}</b>"))
            lay.addWidget(f)
            return fl

        # ── Group table ──
        tl = card("Group table")
        row = QtWidgets.QHBoxLayout()
        self.buttons = {}
        for key, text, fn, tip in (
                ("open", "Open Excel…", self.open_dialog,
                 "Open a group table (.xlsx): one animal per recording"),
                ("sessions", "From saved sessions…", lambda: self.from_sessions(),
                 "Every saved session of a project (Recording sheet → Project) makes the group, "
                 "in memory (Save as Excel… keeps it as a file). Segments are not in a "
                 "session: Compare segments → Add to group table for them."),
                ("add", "Add open recording…", self.add_open,
                 "Write the recording open to the table: its whole-recording metrics and time "
                 "course (after Analyze), its annotations, its event-locked results and its "
                 "Compare segments result. A recording already in it is replaced; its group "
                 "stays."),
                ("reload", "Reload", self.reload, "Read the Excel file again"),
                ("save", "Save as Excel…", self.save_as, "Write the table to a new .xlsx file"),
                ("close", "Close", self.close_table, "Close the table (the file is kept)")):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
            self.buttons[key] = b
        self.lbl_file = QtWidgets.QLabel("No group table open")
        self.lbl_file.setProperty("muted", True)
        row.addWidget(self.lbl_file, 1)
        tl.addLayout(row)
        self.lbl_msg = StatusLabel("")
        self.lbl_msg.setWordWrap(True)
        tl.addWidget(self.lbl_msg)

        # ── Animals ──
        al = card("Animals")
        self.tbl_animals = QtWidgets.QTableWidget(0, 7)
        self.tbl_animals.setHorizontalHeaderLabels(["Use", "Subject", "Sex", "Group", "File",
                                                    "Data · QC", ""])
        self.tbl_animals.verticalHeader().setVisible(False)
        self.tbl_animals.horizontalHeader().setStretchLastSection(False)
        self.tbl_animals.setMinimumHeight(160)
        al.addWidget(self.tbl_animals)
        hint = QtWidgets.QLabel("Group: type a name (Control, CNO…) or pick one; Sex comes "
                                "from the file name. Unticked animals are left out of every "
                                "analysis. Saved in the table.")
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        al.addWidget(hint)

        # ── Compare ──
        cl = card("Compare")
        form = QtWidgets.QHBoxLayout()
        form.addWidget(QtWidgets.QLabel("By"))
        self.cb_by = QtWidgets.QComboBox()
        self.cb_by.addItems(list(gc.BY_CHOICES))
        self.cb_by.setCurrentText(gc.BY_GROUP)
        form.addWidget(self.cb_by)
        form.addWidget(QtWidgets.QLabel("Analysis"))
        self.cb_an = QtWidgets.QComboBox()
        self.cb_an.addItems(list(gc.ANALYSES))
        form.addWidget(self.cb_an)
        form.addStretch(1)
        cl.addLayout(form)
        self.opts = QtWidgets.QStackedWidget()
        cl.addWidget(self.opts)

        def hbox() -> "tuple[QtWidgets.QWidget, QtWidgets.QHBoxLayout]":
            w = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            return w, h

        w, h = hbox()                                           # Pre/Post segments
        self.cb_pairs = QtWidgets.QComboBox()
        self.cb_pairs.addItems([gc.PAIRS_ONE, gc.PAIRS_ALL])
        h.addWidget(self.cb_pairs)
        h.addWidget(QtWidgets.QLabel("Compare"))
        self.cb_a = QtWidgets.QComboBox()
        self.cb_b = QtWidgets.QComboBox()
        h.addWidget(self.cb_a)
        h.addWidget(QtWidgets.QLabel("vs"))
        h.addWidget(self.cb_b)
        self.chk_leave = QtWidgets.QCheckBox("Leave out rows to check")
        self.chk_leave.setChecked(True)
        h.addWidget(self.chk_leave)
        self.chk_all_seg = QtWidgets.QCheckBox("All metrics")
        h.addWidget(self.chk_all_seg)
        h.addStretch(1)
        self.opts.addWidget(w)
        w, h = hbox()                                           # Time course
        h.addWidget(QtWidgets.QLabel("Time 0"))
        self.cb_stim = QtWidgets.QComboBox()
        self.cb_stim.setMinimumWidth(160)
        h.addWidget(self.cb_stim)
        h.addWidget(QtWidgets.QLabel("Metric"))
        self.cb_tcm = QtWidgets.QComboBox()
        self.cb_tcm.addItems(list(gd.TC_METRICS))
        self.cb_tcm.setCurrentText("HR")
        h.addWidget(self.cb_tcm)
        self.ed = {}
        for key, label, val in (("bin", "bins (s)", "60"), ("from", "from", "-10"),
                                ("to", "to", "30"), ("w0", "min · test window", "0"),
                                ("w1", "–", "5")):
            h.addWidget(QtWidgets.QLabel(label))
            e = QtWidgets.QLineEdit(val)
            e.setMaximumWidth(60)
            e.editingFinished.connect(self.changed)
            h.addWidget(e)
            self.ed[key] = e
        h.addWidget(QtWidgets.QLabel("min vs pre"))
        self.chk_delta = QtWidgets.QCheckBox("Δ from baseline")
        self.chk_delta.setChecked(True)
        h.addWidget(self.chk_delta)
        self.chk_lines = QtWidgets.QCheckBox("Each animal")
        h.addWidget(self.chk_lines)
        h.addStretch(1)
        self.opts.addWidget(w)
        w, h = hbox()                                           # Whole recording
        self.chk_all_wh = QtWidgets.QCheckBox("All metrics")
        h.addWidget(self.chk_all_wh)
        h.addStretch(1)
        self.opts.addWidget(w)
        w, h = hbox()                                           # Event-locked
        h.addWidget(QtWidgets.QLabel("Events"))
        self.cb_ev = QtWidgets.QComboBox()
        self.cb_ev.setMinimumWidth(160)
        h.addWidget(self.cb_ev)
        h.addWidget(QtWidgets.QLabel("Configuration"))
        self.cb_cfg = QtWidgets.QComboBox()
        self.cb_cfg.setMinimumWidth(160)
        h.addWidget(self.cb_cfg)
        self.chk_all_ev = QtWidgets.QCheckBox("All metrics")
        h.addWidget(self.chk_all_ev)
        h.addStretch(1)
        self.opts.addWidget(w)
        self._all_checks = (self.chk_all_seg, self.chk_all_wh, self.chk_all_ev)

        # ── Figure ──
        fl = card("Figure")
        frow = QtWidgets.QHBoxLayout()
        frow.addStretch(1)
        self.cb_fm = QtWidgets.QComboBox()
        self.cb_fm.setMinimumWidth(240)
        self.cb_fm.setVisible(False)
        frow.addWidget(self.cb_fm)
        fl.addLayout(frow)
        self.slot = window.figures.slot("group_tab", page, *FIG_SIZE)
        self.slot.setMinimumHeight(460)
        fl.addWidget(self.slot)

        # ── Results ──
        rl = card("Results")
        self.tbl_res = QtWidgets.QTableWidget(0, 0)
        self.tbl_res.verticalHeader().setVisible(False)
        self.tbl_res.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_res.setMinimumHeight(240)
        rl.addWidget(self.tbl_res)
        self.lbl_notes = QtWidgets.QLabel("")
        self.lbl_notes.setWordWrap(True)
        rl.addWidget(self.lbl_notes)
        brow = QtWidgets.QHBoxLayout()
        self.btn_stats = QtWidgets.QPushButton("Save statistics…")
        self.btn_stats.clicked.connect(self.save_stats)
        self.btn_fig = QtWidgets.QPushButton("Save figure…")
        self.btn_fig.clicked.connect(self.save_figure)
        self.btn_prism = QtWidgets.QPushButton("Prism… (new file / template)")
        self.btn_prism.setToolTip("GraphPad Prism: per group one table per metric (one row per "
                                  "animal), the changes by group, the time course as XY tables; "
                                  "new file or poured into your template")
        menu = QtWidgets.QMenu(self.btn_prism)
        menu.addAction("New Prism file…").triggered.connect(lambda: self.prism_new())
        menu.addAction("Fill a Prism template…").triggered.connect(lambda: self.prism_fill())
        self.btn_prism.setMenu(menu)
        self.btn_stats.setToolTip("Excel: the results table, every number behind it (means, "
                                  "SEM, n, tests, Holm), the ANOVA, the per-animal values and "
                                  "the settings")
        brow.addWidget(self.btn_stats)
        brow.addWidget(self.btn_fig)
        brow.addWidget(self.btn_prism)
        self.lbl_out = StatusLabel("")
        brow.addWidget(self.lbl_out, 1)
        rl.addLayout(brow)
        lay.addStretch(1)

        for cb in (self.cb_by, self.cb_pairs, self.cb_a, self.cb_b, self.cb_stim, self.cb_tcm,
                   self.cb_ev, self.cb_cfg):
            cb.currentIndexChanged.connect(lambda *_a: self.changed())
        for chk in (self.chk_leave, self.chk_delta, self.chk_lines) + self._all_checks:
            chk.toggled.connect(lambda *_a: self.changed())
        self.cb_an.currentIndexChanged.connect(self._analysis_changed)
        self.cb_fm.currentIndexChanged.connect(lambda *_a: self.draw())
        self.show_sheets(None)

    # ── the table ───────────────────────────────────────────
    def say(self, text: str, ok: bool = True) -> None:
        self.lbl_msg.configure(text=text, text_color=None if ok else "#E65100")

    def table_name(self) -> str:
        if self.path:
            return os.path.basename(self.path)
        return f"{self.name} (not saved)" if self.name else "group (not saved)"

    def load(self, path: str) -> bool:
        try:
            sheets = gd.read_sheets(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Group table", f"{os.path.basename(path)}: {exc}")
            return False
        self.path = path
        self.name = ""
        self.show_sheets(sheets)
        return True

    def from_sessions(self, project: "Optional[str]" = None) -> None:
        """Tk _from_sessions: the group of every saved session of a project
        (TabGroupMixin's project_sheets), held in memory."""
        from ecg.io import db
        from ecg.ui.tab_group import project_sheets
        if project is None:
            names = db.project_names()
            if not names:
                QtWidgets.QMessageBox.information(
                    self, "Group from saved sessions",
                    "No saved session has a project name yet: type it in the Recording sheet "
                    "(Recording… → Project) and Save Session for each recording.")
                return
            project, ok = QtWidgets.QInputDialog.getItem(
                self, "Group from saved sessions",
                "Every saved session of the project (Recording sheet → Project) makes the "
                "group (Save as Excel… keeps it as a file):", names, 0, False)
            if not ok or not project:
                return
        try:
            sheets, added, skipped = project_sheets(project)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Group from saved sessions", str(exc))
            return
        if not added:
            text = f"⚠  {project}: no saved session could be used"
            if skipped:
                text += "  ·  " + "; ".join(skipped)
            self.say(text, ok=False)
            return
        self.path, self.name = "", project
        self.show_sheets(sheets)
        text = (f"✓  {project}: {len(added)} recording{'s' if len(added) != 1 else ''} from "
                "the saved sessions (in memory: Save as Excel… to keep it)")
        if skipped:
            text += "  ·  left out: " + "; ".join(skipped)
        self.say(text, ok=not skipped)

    def open_dialog(self) -> None:
        path, _f = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open a group table", os.path.dirname(self.path),
            "Excel workbook (*.xlsx);;All files (*)")
        if path:
            self.load(path)

    def reload(self) -> None:
        if self.path:
            self.load(self.path)

    def show_sheets(self, sheets: "Optional[dict]") -> None:
        self.sheets = sheets
        self.animals = None if sheets is None else gd.animals_frame(sheets)
        self.res = None
        self._summary()
        self._refresh_animals()
        self._fill_choices()
        have = sheets is not None
        for key, on in (("reload", bool(self.path)), ("save", have), ("close", have)):
            self.buttons[key].setEnabled(on)
        self.changed()

    def _summary(self) -> None:
        an = self.animals
        if an is None:
            self.lbl_file.setText("No group table open")
            return
        nm, nf = int((an["sex"] == "M").sum()), int((an["sex"] == "F").sum())
        groups = an.loc[an["group"].astype(str) != "", "group"].value_counts()
        parts = [self.table_name(), f"{len(an)} animals", f"{nf} F / {nm} M"]
        if len(groups):
            parts.append(", ".join(f"{g} {n}" for g, n in groups.items()))
        sheets = self.sheets or {}
        kinds = [n for n, k in (("segments", gd.SHEET_SEGMENTS), ("whole", gd.SHEET_WHOLE),
                                ("time course", gd.SHEET_TIME), ("events", gd.SHEET_EVENTS))
                 if k in sheets and len(sheets[k])]
        parts.append(", ".join(kinds) or "no data yet")
        self.lbl_file.setText("  ·  ".join(parts))

    def _refresh_animals(self) -> None:
        t = self.tbl_animals
        t.setRowCount(0)
        an = self.animals
        if an is None or an.empty:
            return
        groups = sorted({str(g) for g in an["group"] if str(g)}, key=str.casefold)
        for i, (_, r) in enumerate(an.iterrows()):
            key = (r["subject"], r["file"])
            t.insertRow(i)
            chk = QtWidgets.QCheckBox()
            chk.setChecked(bool(r["use"]))
            chk.toggled.connect(lambda on, k=key: self.set_animal(k, "use", on))
            t.setCellWidget(i, 0, chk)
            t.setItem(i, 1, QtWidgets.QTableWidgetItem(str(r["subject"])))
            sex = QtWidgets.QComboBox()
            sex.addItems(["M", "F", ""])
            sex.setCurrentText(str(r["sex"]))
            sex.currentTextChanged.connect(lambda v, k=key: self.set_animal(k, "sex", v))
            t.setCellWidget(i, 2, sex)
            grp = QtWidgets.QComboBox()
            grp.setEditable(True)
            grp.addItems(groups or [""])
            grp.setCurrentText(str(r["group"]))
            grp.lineEdit().editingFinished.connect(
                lambda k=key, c=grp: self.set_animal(k, "group", c.currentText()))
            grp.activated.connect(lambda _i, k=key, c=grp: self.set_animal(k, "group",
                                                                          c.currentText()))
            t.setCellWidget(i, 3, grp)
            t.setItem(i, 4, QtWidgets.QTableWidgetItem(str(r["file"])))
            t.setItem(i, 5, QtWidgets.QTableWidgetItem(gd.data_text(r)))
            rm = QtWidgets.QPushButton("✕")
            rm.setToolTip("Remove this recording from the group table (every sheet)")
            rm.clicked.connect(lambda _c=False, k=key: self.remove_animal(k))
            t.setCellWidget(i, 6, rm)
        t.resizeColumnsToContents()

    def persist(self, what: str) -> bool:
        if not self.path:
            self.say(f"✓  {what} (in memory: Save as Excel… to keep it)")
            return True
        try:
            gd.write_sheets(self.path, self.sheets)
        except Exception as exc:
            self.say(f"⚠  Not saved in {os.path.basename(self.path)}: {exc} "
                     "(is it open in Excel?)", ok=False)
            return False
        self.say(f"✓  {what} saved")
        return True

    def set_animal(self, key: "tuple[str, str]", field_: str, value: Any) -> None:
        an = self.animals
        if an is None:
            return
        if field_ == "sex":
            value = normalise_sex(value)
        elif field_ == "group":
            value = str(value).strip()
        m = (an["subject"] == key[0]) & (an["file"] == key[1])
        if not m.any() or an.loc[m, field_].iloc[0] == value:
            return
        an.loc[m, field_] = value
        self.sheets[gd.SHEET_ANIMALS] = an[["subject", "sex", "group", "use", "file"]].copy()
        self.persist(f"{key[0]}: {field_}")
        self._summary()
        if field_ == "group":
            QtCore.QTimer.singleShot(0, self._refresh_animals)
        self.changed(keep_msg=True)

    def remove_animal(self, key: "tuple[str, str]") -> None:
        if self.sheets is None:
            return
        if QtWidgets.QMessageBox.question(
                self, "Group table", f"Remove {key[0]} ({key[1]}) from {self.table_name()}?"
        ) != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        gd.remove_recording(self.sheets, key[0], key[1])
        self.animals = gd.animals_frame(self.sheets)
        self.persist(f"{key[0]} removed")
        self._summary()
        QtCore.QTimer.singleShot(0, self._refresh_animals)
        self._fill_choices()
        self.changed()

    def add_open(self) -> None:
        """Tk _add_open: the recording open written to the table (created if new)."""
        from ecg.ui.tab_group import TabGroupMixin
        host = self.win.figures
        comparison = self.win.recording.analysis.last_comparison
        if self.sheets is not None and not self.path:
            try:
                n, replaced, what = TabGroupMixin._add_open_recording_to_group_table(
                    host, "", comparison=comparison, sheets=self.sheets)
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Group table", str(exc))
                return
            self.show_sheets(self.sheets)
            self.say(f"✓  {what} — {'updated in' if replaced else 'added to'} "
                     f"{self.table_name()} ({n} animals)")
            return
        path = self.path if self.sheets is not None else ""
        if not path:
            path, _f = QtWidgets.QFileDialog.getSaveFileName(
                self, "Group table: add the recording open (created if new)",
                "group_table.xlsx", "Excel workbook (*.xlsx);;All files (*)",
                options=QtWidgets.QFileDialog.Option.DontConfirmOverwrite)
            if not path:
                return
        try:
            n, replaced, what = TabGroupMixin._add_open_recording_to_group_table(
                host, path, comparison=comparison)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Group table", str(exc))
            return
        if self.load(path):
            self.say(f"✓  {what} — {'updated in' if replaced else 'added to'} "
                     f"{os.path.basename(path)} ({n} animals)")

    def save_as(self) -> None:
        if self.sheets is None:
            return
        path, _f = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save the group table as",
            self.path or "group_table.xlsx", "Excel workbook (*.xlsx);;All files (*)")
        if not path:
            return
        try:
            gd.write_sheets(path, self.sheets)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Group table", str(exc))
            return
        self.path = path
        self._summary()
        self.buttons["reload"].setEnabled(True)
        self.say(f"✓  Saved as {os.path.basename(path)}")

    def close_table(self) -> None:
        if self.sheets is None:
            return
        if not self.path and QtWidgets.QMessageBox.question(
                self, "Group table", f"Close {self.table_name()}? It was never saved as Excel."
        ) != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.path = ""
        self.show_sheets(None)
        self.say("")

    # ── analyses ────────────────────────────────────────────
    def _set_items(self, cb: QtWidgets.QComboBox, items: "list[str]", current: str) -> None:
        cb.blockSignals(True)
        cb.clear()
        cb.addItems(items)
        if current in items:
            cb.setCurrentText(current)
        cb.blockSignals(False)

    def _fill_choices(self) -> None:
        sheets = self.sheets or {}
        seg = sheets.get(gd.SHEET_SEGMENTS)
        tags = gc.segment_tags(seg) if seg is not None and len(seg) else []
        a = self.cb_a.currentText() if self.cb_a.currentText() in tags else (tags[0] if tags
                                                                              else "")
        b = self.cb_b.currentText()
        if b not in tags or b == a:
            b = next((t for t in tags if t != a), a)
        self._set_items(self.cb_a, tags, a)
        self._set_items(self.cb_b, tags, b)
        labels = gd.annotation_labels(sheets.get(gd.SHEET_ANNOT))
        stim = self.cb_stim.currentText()
        self._set_items(self.cb_stim, labels, stim if stim in labels else
                        (labels[0] if labels else ""))
        evs = [ALL_EVENTS] + gc.event_labels(sheets)
        self._set_items(self.cb_ev, evs, self.cb_ev.currentText() or ALL_EVENTS)
        cfgs = gc.event_configs(sheets)
        self._set_items(self.cb_cfg, cfgs, self.cb_cfg.currentText() if
                        self.cb_cfg.currentText() in cfgs else (cfgs[0] if cfgs else ""))

    def _analysis_changed(self, index: int) -> None:
        # One "All metrics" for every analysis (Tk's st["all"]).
        on = self._all_value()
        self.opts.setCurrentIndex(index)
        for c in self._all_checks:
            c.blockSignals(True)
            c.setChecked(on)
            c.blockSignals(False)
        self.changed()

    def _all_value(self) -> bool:
        return getattr(self, "_all", False)

    def _read_all(self) -> None:
        src = {gc.AN_SEGMENTS: self.chk_all_seg, gc.AN_WHOLE: self.chk_all_wh,
               gc.AN_EVENTS: self.chk_all_ev}.get(self.cb_an.currentText())
        if src is not None:
            self._all = src.isChecked()

    def run(self) -> "Optional[gc.GroupResult]":
        """Tk _run."""
        from ecg.ui.tab_group import _bin_size, _float
        sheets, an = self.sheets, self.animals
        if sheets is None or an is None:
            return None
        mode, by = self.cb_an.currentText(), self.cb_by.currentText()
        metrics = "all" if self._all_value() else None
        if mode == gc.AN_SEGMENTS:
            return gc.analyse_segments(sheets, an, by, self.cb_a.currentText(),
                                       self.cb_b.currentText(), metrics,
                                       pairs=self.cb_pairs.currentText(),
                                       leave_out=self.chk_leave.isChecked())
        if mode == gc.AN_WHOLE:
            return gc.analyse_whole(sheets, an, by, metrics)
        if mode == gc.AN_EVENTS:
            ev = "" if self.cb_ev.currentText() == ALL_EVENTS else self.cb_ev.currentText()
            return gc.analyse_events(sheets, an, by, ev, metrics,
                                     config=self.cb_cfg.currentText() or None)
        stim = self.cb_stim.currentText()
        if not stim:
            annot = sheets.get(gd.SHEET_ANNOT)
            if gd.annotation_labels(annot):
                raise ValueError("Choose the Time 0 annotation (the stimulation) to align "
                                 "the time course on.")
            if annot is not None and len(annot):
                raise ValueError("Choose the Time 0 annotation: those of the table have no "
                                 "name. Name the stimulation in each recording, then Add "
                                 "open recording.")
            raise ValueError("No annotation in the table to align on: annotate the "
                             "stimulation in each recording, then Add open recording.")
        e = {k: _Text(v) for k, v in self.ed.items()}
        return gc.analyse_time(
            sheets, an, by, stim, self.cb_tcm.currentText(),
            bin_s=_bin_size(e["bin"]), start_min=_float(e["from"], -10.0),
            end_min=_float(e["to"], 30.0),
            window_min=(_float(e["w0"], 0.0), _float(e["w1"], 5.0)),
            delta=self.chk_delta.isChecked())

    def changed(self, keep_msg: bool = False) -> None:
        self._read_all()
        pairs_all = self.cb_pairs.currentText() == gc.PAIRS_ALL
        self.cb_a.setEnabled(not pairs_all)
        self.cb_b.setEnabled(not pairs_all)
        if not keep_msg:
            self.say("")
        self.lbl_out.configure(text="")
        if self.sheets is None:
            self.res = None
            self.show_result(None, "Open a group table")
            return
        try:
            res = self.run()
        except Exception as exc:
            self.res = None
            self.show_result(None, str(exc))
            return
        self.res = res
        self.show_result(res)

    def show_result(self, res: "Optional[gc.GroupResult]", error: str = "") -> None:
        for b in (self.btn_stats, self.btn_fig, self.btn_prism):
            b.setEnabled(res is not None)
        t = self.tbl_res
        t.clear()
        if res is None:
            t.setRowCount(0)
            t.setColumnCount(1)
            t.setHorizontalHeaderLabels([error or "No result"])
            self.lbl_notes.setText("")
            self.cb_fm.setVisible(False)
            msg = error or "No result"
            self.slot.update(lambda fig: fig.text(0.5, 0.5, msg, ha="center", va="center",
                                                  color="#888"))
            return
        d = res.display
        cols = [c for c in d.columns if not str(c).startswith("_")]
        t.setColumnCount(len(cols))
        t.setHorizontalHeaderLabels([str(c) for c in cols])
        t.setRowCount(0)
        bold = QtGui.QFont()
        bold.setBold(True)
        sig_color = QtGui.QColor("#2E7D32")
        for head, rows in gc.display_by_family(d):
            if head:
                r = t.rowCount()
                t.insertRow(r)
                item = QtWidgets.QTableWidgetItem(head.upper())
                item.setFont(bold)
                t.setItem(r, 0, item)
                t.setSpan(r, 0, 1, len(cols))
            for _, row in rows.iterrows():
                r = t.rowCount()
                t.insertRow(r)
                p = row.get("_p")
                sig = isinstance(p, float) and math.isfinite(p) and p < 0.05
                for c, col in enumerate(cols):
                    item = QtWidgets.QTableWidgetItem(str(row[col]))
                    if sig:
                        item.setForeground(sig_color)
                    t.setItem(r, c, item)
        t.resizeColumnsToContents()
        self.lbl_notes.setText("\n".join(f"⚠  {n}" for n in res.notes))
        fm: "list[str]" = []
        if res.figure.get("kind") in ("paired", "dots"):
            fm = list(res.figure.get("values", {})) + list(res.figure.get("poincare") or {})
        self.cb_fm.blockSignals(True)
        current = self.cb_fm.currentText()
        self.cb_fm.clear()
        self.cb_fm.addItems(fm)
        if fm:
            self.cb_fm.setCurrentText(current if current in fm else self._first_metric(fm))
        self.cb_fm.blockSignals(False)
        self.cb_fm.setVisible(bool(fm))
        self.draw()

    @staticmethod
    def _first_metric(names: "list[str]") -> str:
        """FamilyCombo.first: the first metric of the first family."""
        from ecg.core.metric_families import by_family, family_of
        groups = by_family(names, family_of)
        return next((ns[0] for _f, ns in groups if ns), names[0])

    def _fig_metric(self) -> "Optional[str]":
        return (self.cb_fm.currentText() or None) if self.cb_fm.count() else None

    def draw(self) -> None:
        res = self.res
        if res is None:
            return
        from ecg.ui.group_tab_figure import draw_group_tab_figure
        metric, lines = self._fig_metric(), self.chk_lines.isChecked()
        self.slot.update(lambda fig: draw_group_tab_figure(fig, res, metric=metric,
                                                           show_animals=lines))

    # ── exports ─────────────────────────────────────────────
    def _base(self) -> str:
        return os.path.splitext(os.path.basename(self.path or "group"))[0]

    def save_stats(self) -> None:
        from ecg.ui.tab_group import _file_tag
        res = self.res
        if res is None:
            return
        path, _f = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save the group statistics",
            os.path.join(os.path.dirname(self.path),
                         f"{self._base()}_{_file_tag(res.analysis)}_stats.xlsx"),
            "Excel workbook (*.xlsx);;All files (*)")
        if not path:
            return
        try:
            gc.write_results(path, res, os.path.basename(self.path))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Group statistics", str(exc))
            return
        self.lbl_out.configure(text=f"✓  statistics → {os.path.basename(path)}")

    def save_figure(self) -> None:
        from matplotlib.figure import Figure
        from ecg.ui.group_tab_figure import draw_group_tab_figure
        from ecg.ui.tab_group import _file_tag
        from ecg.ui.theme import light_plot_theme
        res = self.res
        if res is None:
            return
        path, _f = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save the group figure",
            os.path.join(os.path.dirname(self.path),
                         f"{self._base()}_{_file_tag(res.analysis)}.png"),
            "PNG image (300 dpi) (*.png);;PDF (vector) (*.pdf);;SVG (vector) (*.svg)")
        if not path:
            return
        fig = Figure(figsize=FIG_SIZE, dpi=300, facecolor="white", layout="constrained")
        try:
            with light_plot_theme():          # on white whatever the theme on screen
                draw_group_tab_figure(fig, res, metric=self._fig_metric(),
                                      show_animals=self.chk_lines.isChecked())
                fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Group figure", str(exc))
            return
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)
        self.lbl_out.configure(text=f"✓  figure → {os.path.basename(path)}")

    def _notes(self, res: "gc.GroupResult") -> "list[tuple[str, Any]]":
        return ([("Group table", os.path.basename(self.path))] + list(res.settings)
                + [("Groups", ", ".join(f"{g}" for g in res.groups)),
                   ("Animals", res.n_animals)] + [("Note", n) for n in res.notes])

    def prism_new(self, path: "Optional[str]" = None) -> None:
        from ecg.io.prism_template import write_metric_pzfx
        from ecg.ui.tab_group import _file_tag
        res = self.res
        if res is None:
            return
        if path is None:
            path, _f = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export to GraphPad Prism",
                os.path.join(os.path.dirname(self.path),
                             f"{self._base()}_{_file_tag(res.analysis)}.pzfx"),
                "GraphPad Prism (*.pzfx);;All files (*)")
        if not path:
            return
        try:
            n = write_metric_pzfx(res.prism, path, notes=self._notes(res))
        except Exception as exc:
            log.exception("group tab Prism export")
            QtWidgets.QMessageBox.critical(self, "Prism export", str(exc))
            return
        self.lbl_out.configure(text=f"✓  Prism: {n} tables → {os.path.basename(path)}")

    def prism_fill(self, template: "Optional[str]" = None, out: "Optional[str]" = None) -> None:
        from ecg.io.prism_template import PrismTemplateError, fill_template
        res = self.res
        if res is None:
            return
        cmp_tab = self.win.compare                  # the template asked as there
        tpl = template or cmp_tab.ask_prism_template()
        if not tpl:
            return
        out = out or cmp_tab.ask_filled_path(tpl, self._base())
        if not out:
            return
        try:
            report = fill_template(tpl, res.prism, out,
                                   notes=self._notes(res) + [("Prism template",
                                                              os.path.basename(tpl))])
        except Exception as exc:
            if not isinstance(exc, PrismTemplateError):
                log.exception("group tab Prism template")
            QtWidgets.QMessageBox.critical(self, "Prism template", str(exc))
            return
        self.lbl_out.configure(
            text=(("⚠  " if report.warn else "✓  ") + f"Prism template {os.path.basename(tpl)}: "
                  f"{report.summary()} → {os.path.basename(out)}"),
            text_color="#E65100" if report.warn else None)
