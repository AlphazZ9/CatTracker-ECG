# -*- coding: utf-8 -*-
"""
ecg.qt.dialogs.artifact_review
------------------------------
Review artifacts: the RR artifact candidates of the peaks
(ecg.core.detection.detect_rr_artifacts, with the Tk review's parameters:
min R-R of the settings, the HRV max R-R, 11-beat local median, 20 %
deviation), one row each with its context on the signal and a Keep /
Remove decision, then apply_artifact_decisions / merge_artifact_reports
as ECGApp._open_artifact_review does.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from PySide6 import QtGui, QtWidgets

from ecg.core.detection import (apply_artifact_decisions, detect_rr_artifacts,
                                merge_artifact_reports)
from ecg.core.models import MouseECG
from ecg.qt.widgets.mpl_canvas import MplCanvas

if TYPE_CHECKING:
    from ecg.qt.main_window import MainWindow

WINDOW_BEATS = 11
DEV_THRESHOLD = 0.20
TYPE_LABEL = {
    "nonphysio": "Non-physiological (outside HR bounds)",
    "ectopic": "Ectopic beat (premature: RR shorter than local median)",
    "duplicate": "Duplicate / sub-minimum interval",
}
CONTEXT_S = 1.2


def _ms(v: float) -> str:
    return "—" if np.isnan(v) else f"{v:.1f} ms ({60000 / max(v, 1):.0f} bpm)"


class ArtifactReviewDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget, signal: np.ndarray, rpeaks: np.ndarray,
                 fs: float, candidates: "list[dict]") -> None:
        super().__init__(parent)
        self.setWindowTitle("Artifact Review")
        self.resize(1000, 680)
        self.sig, self.rp, self.fs = signal, np.asarray(rpeaks, dtype=int), fs
        self.candidates = [dict(c) for c in candidates]
        lay = QtWidgets.QVBoxLayout(self)
        self.lbl_counter = QtWidgets.QLabel("")
        lay.addWidget(self.lbl_counter)
        self.table = QtWidgets.QTableWidget(len(self.candidates), 7)
        self.table.setHorizontalHeaderLabels(["Time (s)", "Type", "RR before", "RR after",
                                              "Local median", "Deviation", "Decision"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        for i, c in enumerate(self.candidates):
            texts = (f"{c['sample'] / fs:.3f}", c["type"], _ms(c["rr_prev_ms"]),
                     _ms(c["rr_next_ms"]), _ms(c["rr_ref_ms"]), f"{c['deviation'] * 100:.1f} %")
            for col, text in enumerate(texts):
                item = QtWidgets.QTableWidgetItem(text)
                if col == 1:
                    item.setToolTip(TYPE_LABEL.get(c["type"], c["type"]))
                self.table.setItem(i, col, item)
            cb = QtWidgets.QComboBox()
            cb.addItem("Remove", "remove")
            cb.addItem("Keep", "keep")
            cb.setCurrentIndex(0 if c["decision"] == "remove" else 1)
            cb.currentIndexChanged.connect(lambda _i, k=i, w=cb: self._decide(k, w.currentData()))
            self.table.setCellWidget(i, 6, cb)
        self.table.resizeColumnsToContents()
        self.table.currentCellChanged.connect(lambda r, *_a: self.show_candidate(r))
        lay.addWidget(self.table, 2)
        self.canvas = MplCanvas(self, 9.0, 3.2)
        self.canvas.setMinimumHeight(280)
        lay.addWidget(self.canvas, 3)
        bar = QtWidgets.QHBoxLayout()
        b_keep = QtWidgets.QPushButton("Keep all")
        b_keep.clicked.connect(lambda: self._set_all("keep"))
        b_rm = QtWidgets.QPushButton("Remove all")
        b_rm.clicked.connect(lambda: self._set_all("remove"))
        bar.addWidget(b_keep)
        bar.addWidget(b_rm)
        bar.addStretch(1)
        buttons = QtWidgets.QDialogButtonBox()
        buttons.addButton("Apply", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        bar.addWidget(buttons)
        lay.addLayout(bar)
        QtGui.QShortcut(QtGui.QKeySequence("K"), self, activated=lambda: self._key("keep"))
        QtGui.QShortcut(QtGui.QKeySequence("R"), self, activated=lambda: self._key("remove"))
        self._update_counter()
        if self.candidates:
            self.table.selectRow(0)
            self.show_candidate(0)

    # ── decisions ───────────────────────────────────────────
    def _decide(self, k: int, decision: str) -> None:
        self.candidates[k]["decision"] = decision
        self._update_counter()

    def _set_all(self, decision: str) -> None:
        for i in range(len(self.candidates)):
            w = self.table.cellWidget(i, 6)
            w.setCurrentIndex(0 if decision == "remove" else 1)

    def _key(self, decision: str) -> None:
        """K / R: decide the selected candidate and go to the next one."""
        r = self.table.currentRow()
        if r < 0:
            return
        self.table.cellWidget(r, 6).setCurrentIndex(0 if decision == "remove" else 1)
        if r + 1 < len(self.candidates):
            self.table.selectRow(r + 1)

    def _update_counter(self) -> None:
        n_rm = sum(1 for c in self.candidates if c["decision"] == "remove")
        self.lbl_counter.setText(f"{len(self.candidates)} candidates · {n_rm} to remove  "
                                 "(K: keep, R: remove, then the next one)")

    def result_candidates(self) -> "list[dict]":
        return self.candidates

    # ── the context of one candidate ────────────────────────
    def show_candidate(self, idx: int) -> None:
        if not 0 <= idx < len(self.candidates):
            return
        c, fs, sig = self.candidates[idx], self.fs, self.sig
        from ecg.ui.theme import ARTIFACT_TYPE_COLOR, ORANGE
        color = ARTIFACT_TYPE_COLOR.get(c["type"], ORANGE)
        samp = int(c["sample"])
        lo = max(0, samp - int(CONTEXT_S * fs))
        hi = min(len(sig) - 1, samp + int(CONTEXT_S * fs))
        peaks = self.rp[(self.rp >= lo) & (self.rp < hi)]

        def draw(fig) -> None:
            ax = fig.add_subplot(111)
            ax.plot(np.arange(lo, hi) / fs, sig[lo:hi], lw=0.9)
            for p in peaks:
                if p != samp:
                    ax.scatter([p / fs], [sig[p]], color="#2E9E5B", s=40, marker="^", zorder=4)
            ax.axvline(samp / fs, color=color, lw=1.5)
            ax.scatter([samp / fs], [sig[samp]], color=color, s=90, marker="^", zorder=6)
            ax.set_xlim(lo / fs, hi / fs)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (norm.)")
            ax.grid(True, lw=0.5, alpha=0.5)
            ax.set_title(f"Beat at {samp / fs:.3f} s — {c['type']} "
                         f"(deviation {c['deviation'] * 100:.1f}%)", color=color, loc="left")
        self.canvas.update_figure(draw)


def review_artifacts(window: "MainWindow") -> None:
    """ECGApp._open_artifact_review in the Qt interface."""
    rec = window.recording
    s, d, a = rec.signal, rec.detection, rec.analysis
    if s.filtered is None or d.rpeaks_ok is None or len(d.rpeaks_ok) < 4:
        QtWidgets.QMessageBox.warning(window, "Not ready", "Click Detect Peaks first to load peaks.")
        return
    host = window.figures
    rp, fs, sig = d.rpeaks_ok.copy(), s.fs, s.filtered
    rr_min = float(host._safe_float(host.ent_minrr, MouseECG.RR_MIN_MS))
    _lo, rr_max = host.analysis_ctrl.hrv_rr_bounds()
    prev_report = a.artifact_report
    gen = window.generation

    def done(candidates: "list[dict]") -> None:
        if gen != window.generation:
            return
        if not candidates:
            window.set_status("No artifacts detected — signal looks clean ✓")
            return
        counts: "dict[str, int]" = {}
        for c in candidates:
            counts[c["type"]] = counts.get(c["type"], 0) + 1
        window.set_status(f"Reviewing {len(candidates)} candidates ("
                          + "  ·  ".join(f"{v} {k}" for k, v in counts.items()) + ")", warn=True)
        dlg = ArtifactReviewDialog(window, sig, rp, fs, candidates)
        if not dlg.exec():
            window.set_status("Artifact review cancelled")
            return
        apply_review(window, rp, fs, dlg.result_candidates(), prev_report)

    window.run_job("Artifact review", detect_rr_artifacts, rp, fs, rr_min_ms=rr_min,
                   rr_max_ms=rr_max, window_beats=WINDOW_BEATS, dev_threshold=DEV_THRESHOLD,
                   signal=sig, progress=False, on_done=done)


def apply_review(window: "MainWindow", rp: np.ndarray, fs: float, result: "list[dict]",
                 prev_report: "Optional[dict]") -> None:
    """The decisions applied: the Tk review's state writes."""
    rec = window.recording
    d, a = rec.detection, rec.analysis
    corrected, report = apply_artifact_decisions(rp, result, fs=fs)
    removed = report["n_in"] - report["n_out"]
    d.rpeaks_ok = corrected
    a.artifact_report = merge_artifact_reports(
        prev_report, report, corrected, window.figures.analysis_ctrl.break_tol_samples(fs))
    a.artifact_candidates = result
    if removed > 0:
        removed_samples = {c["sample"] for c in result if c["decision"] == "remove"}
        d.manual_excluded -= removed_samples
        d.manual_added -= removed_samples
    art = (f"Artifact review: −{removed} beats (non-physio={report['n_nonphysio']}  "
           f"ectopic={report['n_ectopic']}  dup={report['n_duplicate']}  "
           f"kept={report['n_kept']})")
    if removed > 0:
        window.detection._drop_results()
        a.epoch_df = None
        window.figures.analysis_ctrl.invalidate_autonomic()
        window.set_status(art + "  ⚠ Click Analyze to update HRV metrics.", warn=True)
    else:
        window.set_status(art)
    window.detection.refresh()
    window.peaks_changed()
    window.detection.lbl_peaks.setText(f"Peaks: {len(corrected):,} (after review)")
