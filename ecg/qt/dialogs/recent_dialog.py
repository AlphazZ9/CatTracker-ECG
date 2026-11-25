# -*- coding: utf-8 -*-
"""
ecg.qt.dialogs.recent_dialog
----------------------------
File > Recent recordings (the Tk Recent recordings window): the recordings
of the registry, newest first, with their HR, SDNN, duration and notes;
double-click (or Open) opens one. Cohort Trends plots HR / SDNN / RMSSD of
the whole registry (ecg.figures.cohort_trends).
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from PySide6 import QtCore, QtWidgets

from ecg.figures.cohort_trends import draw_cohort_trends, usable_rows
from ecg.qt.widgets.mpl_canvas import MplCanvas

if TYPE_CHECKING:
    from ecg.qt.main_window import MainWindow

RECENT_LIMIT = 20         # the Tk window's
COHORT_LIMIT = 500        # Cohort Trends: the whole registry, not the 20 newest


def _registry_rows(limit: int) -> "list[dict]":
    from ecg.io.db import _DB_AVAILABLE, recent_recordings
    return recent_recordings(limit=limit) if _DB_AVAILABLE else []


def recent_rows(extra_paths: "list[str]") -> "list[dict]":
    """The registry's recent recordings that still exist, then the files
    opened in this run that the registry does not hold (no stats)."""
    rows = [r for r in _registry_rows(RECENT_LIMIT) if os.path.exists(r["filepath"])]
    known = {r["filepath"] for r in rows}
    rows += [{"filepath": p} for p in extra_paths if p not in known and os.path.exists(p)]
    return rows


def row_texts(r: dict) -> "tuple[str, str, str, str, str]":
    """(file, HR, SDNN, duration, notes) as the Tk cards show them."""
    return (os.path.basename(r["filepath"]),
            f"{r['hr_mean']:.0f} bpm" if r.get("hr_mean") else "—",
            f"{r['sdnn']:.1f} ms" if r.get("sdnn") else "—",
            f"{r['duration_s']:.0f} s" if r.get("duration_s") else "",
            (r.get("notes") or "")[:90])


class RecentDialog(QtWidgets.QDialog):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.win = window
        self.setWindowTitle("Recent recordings")
        self.resize(760, 480)
        lay = QtWidgets.QVBoxLayout(self)
        self.rows = recent_rows(list(window.figures.session.recent_files))
        self.table = QtWidgets.QTableWidget(len(self.rows), 5)
        self.table.setHorizontalHeaderLabels(["File", "HR", "SDNN", "Duration", "Notes"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        for i, r in enumerate(self.rows):
            for c, text in enumerate(row_texts(r)):
                item = QtWidgets.QTableWidgetItem(text)
                if c == 0:
                    item.setToolTip(r["filepath"])
                if c == 2 and text == "—":
                    item.setToolTip("SDNN not cached for this recording — open and save it "
                                    "again to backfill")
                self.table.setItem(i, c, item)
        self.table.resizeColumnsToContents()
        self.table.doubleClicked.connect(lambda idx: self.open_row(idx.row()))
        if not self.rows:
            lay.addWidget(QtWidgets.QLabel("No recent recordings yet."))
        lay.addWidget(self.table, 1)
        bar = QtWidgets.QHBoxLayout()
        b_open = QtWidgets.QPushButton("Open")
        b_open.setProperty("primary", True)
        b_open.clicked.connect(lambda: self.open_row(self.table.currentRow()))
        bar.addWidget(b_open)
        b_trends = QtWidgets.QPushButton("Cohort Trends…")
        b_trends.clicked.connect(self.open_trends)
        bar.addWidget(b_trends)
        bar.addStretch(1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.reject)
        bar.addWidget(close)
        lay.addLayout(bar)
        self.trends: "Optional[CohortTrendsDialog]" = None

    def open_row(self, i: int) -> None:
        if 0 <= i < len(self.rows):
            self.accept()
            self.win.open_path(self.rows[i]["filepath"])

    def open_trends(self) -> None:
        self.trends = CohortTrendsDialog(self, _registry_rows(COHORT_LIMIT))
        self.trends.show()


class CohortTrendsDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget, rows: "list[dict]") -> None:
        super().__init__(parent)
        self.setWindowTitle("Cohort Trends")
        self.resize(820, 680)
        lay = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel("Heart rate, SDNN, and RMSSD across every saved recording in "
                                 "the registry, ordered by save time and coloured by Project "
                                 "name.")
        intro.setProperty("muted", True)
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.rows = usable_rows(rows)
        self.canvas: "Optional[MplCanvas]" = None
        if not self.rows:
            empty = QtWidgets.QLabel("No cohort data yet — save a session for at least one "
                                     "recording to start building trends.")
            empty.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(empty, 1)
            return
        self.canvas = MplCanvas(self, 7.6, 6.4)
        lay.addWidget(self.canvas, 1)
        self.canvas.update_figure(lambda fig: draw_cohort_trends(fig, self.rows))
