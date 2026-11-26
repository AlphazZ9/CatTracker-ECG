# -*- coding: utf-8 -*-
"""
ecg.qt.tabs.compare_tab
-----------------------
Compare segments: 1 to MAX_SEGMENTS windows of the recording typed in, the
statistics of each (ecg.core.analysis.compute_segment_stats in a background
job, with the HRV RR bounds, the artifact breaks and the frequency bands of
the analysis), the descriptive block-mean comparison with a chosen
reference (compare_segments, quality_flags), the metric table, the RR
tachograms or Poincaré plots, and the Excel / CSV export and the table on
the clipboard (ecg.io.segment_export).

The computation is that of the Tk tab (ecg/ui/compare_segments.py
_run_compare / _on_done), and its result is kept the same way
(analysis.last_comparison) for the Report exports.

The configurations, the Prism / figure exports, Add to group table, the
group Poincaré and the setup kept in sessions: ecg.qt.tabs.compare_more.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ecg.core.analysis import compare_segments, compute_segment_stats, segment_comparison_text
from ecg.core.metric_families import by_family, family_of
from ecg.core.models import MouseECG
from ecg.core.segment_plan import (MAX_SEGMENTS, PAIRS, PAIRWISE, comparators_caption,
                                   comparators_reference, comparison_pairs, quality_flags,
                                   reference_comparators, unique_tags)
from ecg.io.segment_export import ComparisonResult, comparison_tsv, write_csv, write_xlsx
from ecg.qt.figure_host import StatusLabel
from ecg.qt.tabs.compare_more import CompareMoreMixin, fmt_s

if TYPE_CHECKING:
    from ecg.qt.main_window import MainWindow

# ecg/ui/compare_segments.py CompareSegmentsMixin._SEGMENT_TABLE_METRICS
METRICS: "list[tuple[str, str, str]]" = [
    ("n_beats", "Beats", ""), ("duration_s", "Duration", "s"),
    ("n_blocks", "Blocks used (n)", ""), ("hr_block_mean", "HR, mean of block means", "bpm"),
    ("hr_block_sd", "HR, SD of block means", "bpm"), ("hr_mean", "Mean HR (60000/mean NN)", "bpm"),
    ("hr_sd", "Instantaneous HR SD", "bpm"), ("hr_min", "Min inst. HR (2nd %ile)", "bpm"),
    ("hr_max", "Max inst. HR (98th %ile)", "bpm"), ("rr_mean", "Mean RR", "ms"),
    ("rr_sd", "RR SD", "ms"), ("rr_cv", "RR CV", "%"), ("sdnn", "SDNN", "ms"),
    ("rmssd", "RMSSD", "ms"), ("pnn6", "pNN6", "%"), ("lf", "LF power", "ms²"),
    ("hf", "HF power", "ms²"), ("lf_nu", "LF (n.u.)", "%"), ("hf_nu", "HF (n.u.)", "%"),
    ("lf_hf", "LF/HF", ""), ("sd1", "Poincaré SD1", "ms"), ("sd2", "Poincaré SD2", "ms"),
    ("sd_ratio", "SD1/SD2", "")]
_COUNT_KEYS = {"n_beats", "n_blocks", "duration_s"}


def fmt_value(key: str, v: Any) -> str:
    """The table's number format (_fmt / _fmt_cell of the Tk tab)."""
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v) if v else "—"
    if not np.isfinite(fv):
        return "—"
    if key in _COUNT_KEYS and fv.is_integer():
        return f"{fv:.0f}"
    if abs(fv) >= 100:
        return f"{fv:.1f}"
    if abs(fv) >= 10:
        return f"{fv:.2f}"
    return f"{fv:.3f}"


def fmt_delta(va: float, vb: float) -> str:
    d = (100.0 * (vb - va) / abs(va)
         if np.isfinite(va) and np.isfinite(vb) and abs(va) > 1e-9 else float("nan"))
    if not np.isfinite(d):
        return "—"
    if abs(d) < 0.05:
        return "0.0 %"
    return f"{'▲' if d > 0 else '▼'} {abs(d):.1f} %"


def segment_names(segs: "list[dict]") -> "list[str]":
    """_seg_names: the labels, or the unique tags when two clash."""
    labels = [str(s.get("label") or "").strip() for s in segs]
    tags = unique_tags(labels)
    if tags == labels:
        return labels
    return [f"{t}: {lbl}" if lbl else t for t, lbl in zip(tags, labels)]


def run_segments(sig: np.ndarray, rp: np.ndarray, fs: float,
                 windows: "list[tuple[str, float, float]]", **seg_kw: Any) -> "list[dict]":
    """The background part of _run_compare."""
    return [compute_segment_stats(sig, rp, fs, lo, hi, lbl, **seg_kw) for lbl, lo, hi in windows]


class CompareTab(CompareMoreMixin, QtWidgets.QScrollArea):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.win = window
        host = window.figures
        self.setWidgetResizable(True)
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        self.setWidget(page)
        lay.setContentsMargins(24, 16, 24, 16)
        lay.setSpacing(12)
        from ecg.qt.widgets.shell import autohide, page_header
        lay.addLayout(page_header('Compare segments'))
        self.segs: "list[dict]" = []
        self.ref: Any = 0
        self.jitter = True

        card = QtWidgets.QFrame(objectName="Card")
        cl = QtWidgets.QVBoxLayout(card)
        cl.addWidget(QtWidgets.QLabel("<b>Segments</b> (label, start and end in seconds)"))
        self._build_config_bar(cl)
        self.rows = QtWidgets.QTableWidget(0, 3)
        self.rows.setHorizontalHeaderLabels(["Label", "Start (s)", "End (s)"])
        self.rows.verticalHeader().setVisible(False)
        hh = self.rows.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for c in (1, 2):
            hh.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeMode.Fixed)
            self.rows.setColumnWidth(c, 140)
        # As tall as its rows (fit_rows): a fixed 220 px box left a blank
        # band under the usual two segments.
        model = self.rows.model()
        for sig in (model.rowsInserted, model.rowsRemoved, model.modelReset):
            sig.connect(self._fit_rows)          # a method: dropped with the page
        cl.addWidget(self.rows)
        bar = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add segment")
        self.btn_add.clicked.connect(self.add_segment)
        self.btn_remove = QtWidgets.QPushButton("Remove selected")
        self.btn_remove.clicked.connect(self.remove_segments)
        bar.addWidget(self.btn_add)
        bar.addWidget(self.btn_remove)
        bar.addSpacing(16)
        bar.addWidget(QtWidgets.QLabel("Block (s):"))
        self.sp_block = QtWidgets.QDoubleSpinBox()
        self.sp_block.setRange(0.1, 36000)
        self.sp_block.setDecimals(1)
        self.sp_block.setValue(float(MouseECG.SEGMENT_BLOCK_S))
        self.sp_block.setToolTip("Each segment is summarised by its non-overlapping blocks of "
                                 "this length (no beat-level test: beats are not replicates)")
        bar.addWidget(self.sp_block)
        self.btn_run = QtWidgets.QPushButton("Compare")
        self.btn_run.setProperty("action", True)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run)
        bar.addWidget(self.btn_run)
        self.lbl_status = StatusLabel("Detect the peaks first.")
        self.lbl_status.setProperty("muted", True)
        self.lbl_status.setWordWrap(True)
        bar.addWidget(self.lbl_status, 1)
        cl.addLayout(bar)
        lay.addWidget(card)

        res = QtWidgets.QFrame(objectName="Card")
        rl = QtWidgets.QVBoxLayout(res)
        rl.addWidget(QtWidgets.QLabel("<b>Results</b>"))
        ref_row = QtWidgets.QHBoxLayout()
        ref_row.addWidget(QtWidgets.QLabel("Δ % against:"))
        self.cb_ref = QtWidgets.QComboBox()
        self.cb_ref.setMinimumWidth(240)
        self.cb_ref.activated.connect(self._ref_picked)
        ref_row.addWidget(self.cb_ref)
        self.lbl_ref = QtWidgets.QLabel("")
        self.lbl_ref.setProperty("muted", True)
        ref_row.addWidget(self.lbl_ref, 1)
        self.cb_view = QtWidgets.QComboBox()
        self.cb_view.addItems(["RR tachograms", "Poincaré"])
        self.cb_view.currentIndexChanged.connect(lambda *_a: self.redraw_plot())
        ref_row.addWidget(self.cb_view)
        self.chk_jitter = QtWidgets.QCheckBox("Jitter")
        self.chk_jitter.setChecked(True)
        self.chk_jitter.setToolTip("Poincaré: spread the points that RR quantisation stacks")
        self.chk_jitter.toggled.connect(lambda on: (setattr(self, "jitter", on),
                                                    self.redraw_plot()))
        ref_row.addWidget(self.chk_jitter)
        rl.addLayout(ref_row)
        self.lbl_cmp = QtWidgets.QLabel("")
        self.lbl_cmp.setWordWrap(True)
        rl.addWidget(autohide(self.lbl_cmp))
        self.lbl_flags = QtWidgets.QLabel("")
        self.lbl_flags.setWordWrap(True)
        rl.addWidget(autohide(self.lbl_flags))
        self.table = QtWidgets.QTableWidget(0, 0)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        # The table and the figure side by side, sharing the height of the
        # card (they were stacked: two 420 px blocks, mostly empty).
        self.table.setMinimumHeight(360)
        self.plot = host.slot("compare", page, 8.0, 4.4)
        self.plot.setMinimumHeight(360)
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self.table, 2)
        body.addWidget(self.plot, 3)
        rl.addLayout(body, 1)
        ex = QtWidgets.QHBoxLayout()
        self.btn_excel = QtWidgets.QPushButton("Export Excel / CSV…")
        self.btn_excel.clicked.connect(self.export_excel)
        self.btn_copy = QtWidgets.QPushButton("Copy table")
        self.btn_copy.clicked.connect(self.copy_table)
        self.export_buttons = [self.btn_excel, self.btn_copy]
        for b in self.export_buttons:
            ex.addWidget(b)
        self.export_buttons += self._build_more_exports(ex)
        for b in self.export_buttons:
            b.setEnabled(False)
        self.lbl_exp = QtWidgets.QLabel("")
        self.lbl_exp.setWordWrap(True)
        ex.addWidget(self.lbl_exp, 1)
        rl.addLayout(ex)
        lay.addWidget(res, 1)
        self._build_group_card(lay)
        self._fit_rows()

    # ── segments ────────────────────────────────────────────
    def _fit_rows(self, *_args) -> None:
        """The segment table as tall as its rows (at least two)."""
        t = self.rows
        row_h = t.verticalHeader().defaultSectionSize()
        n = max(t.rowCount(), 2)
        t.setFixedHeight(t.horizontalHeader().sizeHint().height()
                         + n * row_h + 2 * t.frameWidth() + 2)

    def _duration(self) -> float:
        s = self.win.recording.signal
        return float(len(s.filtered)) / s.fs if s.filtered is not None and s.fs else 0.0

    def set_rows(self, values: "list[tuple[str, float, float]]") -> None:
        self.rows.setRowCount(0)
        for lbl, lo, hi in values:
            self._append_row(lbl, lo, hi)
        self.btn_add.setEnabled(self.rows.rowCount() < MAX_SEGMENTS)

    def _append_row(self, lbl: str, lo: float, hi: float) -> None:
        r = self.rows.rowCount()
        self.rows.insertRow(r)
        for c, v in enumerate((lbl, fmt_s(lo), fmt_s(hi))):
            self.rows.setItem(r, c, QtWidgets.QTableWidgetItem(v))

    def read_rows(self) -> "list[tuple[str, float, float]]":
        out = []
        for i in range(self.rows.rowCount()):
            texts = [(self.rows.item(i, c).text() if self.rows.item(i, c) else "")
                     for c in range(3)]
            try:
                lo, hi = float(texts[1]), float(texts[2])
            except ValueError:
                raise ValueError(f"segment {i + 1}: invalid time") from None
            out.append((texts[0].strip() or f"Segment {chr(65 + i)}", lo, hi))
        return out

    def add_segment(self) -> None:
        """A new row after the last one (as long as the last, to the end)."""
        n = self.rows.rowCount()
        if n >= MAX_SEGMENTS:
            return
        dur = self._duration()
        try:
            rows = self.read_rows()
            lo = rows[-1][2] if rows else 0.0
            length = (rows[-1][2] - rows[-1][1]) if rows else dur
        except ValueError:
            lo, length = 0.0, dur
        self._append_row(f"Segment {chr(65 + n)}", lo, min(dur, lo + length) if dur else lo)
        self.btn_add.setEnabled(self.rows.rowCount() < MAX_SEGMENTS)

    def remove_segments(self) -> None:
        for r in sorted({i.row() for i in self.rows.selectedIndexes()}, reverse=True):
            if self.rows.rowCount() > 1:
                self.rows.removeRow(r)
        self.btn_add.setEnabled(self.rows.rowCount() < MAX_SEGMENTS)

    def peaks_changed(self) -> None:
        """New peaks: the comparison is computed again on demand; new
        recording: the default two halves."""
        rec = self.win.recording
        ok = rec.detection.rpeaks_ok is not None and rec.signal.filtered is not None
        self.btn_run.setEnabled(ok)
        if ok:
            self.fill_markers()
        if ok and self.rows.rowCount() == 0:
            dur = self._duration()
            self.set_rows([("Segment A", 0.0, float(int(dur / 2))),
                           ("Segment B", float(int(dur / 2)), float(int(dur)))])
        if self.segs:
            self._clear_results("The peaks changed — click Compare again.")
        elif ok:
            self.lbl_status.configure(text="")

    def file_opened(self) -> None:
        self.rows.setRowCount(0)
        self.ed_stim.clear()
        self.cfg_say("")
        self.fill_markers()
        self._clear_results("Detect the peaks first.")
        self.btn_run.setEnabled(False)

    def _clear_results(self, status: str) -> None:
        self.segs = []
        self.win.recording.analysis.last_comparison = None
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.lbl_cmp.setText("")
        self.lbl_flags.setText("")
        self.lbl_ref.setText("")
        self.cb_ref.clear()
        self.plot.clear()
        for b in self.export_buttons:
            b.setEnabled(False)
        self.lbl_exp.setText("")
        self.lbl_status.configure(text=status)

    # ── Compare ─────────────────────────────────────────────
    def run(self, windows: "Optional[list]" = None, block_s: "Optional[float]" = None) -> None:
        """Compare the rows typed (or *windows* / *block_s*: the comparison
        a restored session showed)."""
        rec, host = self.win.recording, self.win.figures
        s, d = rec.signal, rec.detection
        if s.filtered is None or d.rpeaks_ok is None:
            return
        dur = self._duration()
        if windows is None:
            try:
                windows = self.read_rows()
            except ValueError as exc:
                self.lbl_status.configure(text=f"⚠  {exc}", text_color="#C62828")
                return
        block_s = float(self.sp_block.value()) if block_s is None else float(block_s)
        for i, (_lbl, lo, hi) in enumerate(windows):
            if lo >= hi:
                self.lbl_status.configure(text=f"⚠  Segment {i + 1}: start ≥ end",
                                          text_color="#C62828")
                return
            if lo < 0 or hi > dur:
                self.lbl_status.configure(
                    text=f"⚠  Segment {i + 1}: outside recording ({dur:.1f} s)",
                    text_color="#C62828")
                return
        ctl = host.analysis_ctrl
        rp = d.rpeaks_ok.copy()
        rr_lo, rr_hi = ctl.hrv_rr_bounds()
        vlf, lf, hf = ctl.get_freq_bands()
        seg_kw = dict(lf_band=lf, hf_band=hf, vlf_band=vlf, rr_min_ms=rr_lo, rr_max_ms=rr_hi,
                      breaks=ctl.hrv_breaks(rp), block_s=block_s)
        self.btn_run.setEnabled(False)
        self.lbl_status.configure(text="Computing…")
        gen = self.win.generation
        self.win.run_job("Compare segments", run_segments, s.filtered, rp, s.fs, windows,
                         progress=False, peaks_bound=True, **seg_kw,
                         on_done=lambda segs: self._done(segs, windows, block_s, lf, hf)
                         if gen == self.win.generation else None,
                         on_end=lambda: self.btn_run.setEnabled(
                             self.win.recording.detection.rpeaks_ok is not None))

    def _fill_ref_choices(self) -> None:
        self._ref_opts: "dict[str, Any]" = {}
        for k, name in enumerate(segment_names(self.segs)):
            self._ref_opts[f"{name} (reference)"] = k
        if len(self.segs) > 2:
            self._ref_opts["Previous segment (pairwise)"] = PAIRWISE
        if len(self.segs) > 3:
            self._ref_opts["Pairs (2 vs 1, 4 vs 3, 6 vs 5)"] = PAIRS
        if self.ref not in self._ref_opts.values():
            self.ref = 0
        self.cb_ref.clear()
        self.cb_ref.addItems(list(self._ref_opts))
        self.cb_ref.setCurrentIndex(list(self._ref_opts.values()).index(self.ref))
        self.cb_ref.setEnabled(len(self.segs) > 1)

    def _done(self, segs: "list[dict]", windows, block_s: float, lf, hf) -> None:
        """_on_done of the Tk tab."""
        self.segs = segs
        self._fill_ref_choices()
        self._compare()
        a = self.win.recording.analysis
        errs = [s["error"] for s in segs if s.get("error")]
        n_flag = sum(1 for f in self.flags if f)
        a.last_comparison = ComparisonResult(
            segs=segs, metrics=list(METRICS), cmps=self.cmps, flags=self.flags, template="",
            subject=self.win.subject(), filepath=self.win.recording.signal.filepath or "",
            event=None, artefact_margin_s=None, block_s=block_s, lf_band=tuple(lf),
            hf_band=tuple(hf), comparators=list(self.comps))
        a.last_comparison_run = {"rows": [[str(l), float(lo), float(hi)]
                                          for l, lo, hi in windows], "block_s": float(block_s)}
        for b in self.export_buttons:
            b.setEnabled(True)
        self.lbl_exp.setText("")
        if errs:
            self.lbl_status.configure(text=f"⚠  {errs[0]}", text_color="#C62828")
        else:
            withheld = [s["label"] for s in segs if s.get("freq_withheld_reason")]
            self.lbl_status.configure(
                text="✓  " + "  ·  ".join(f"{s['label']}: {s['n_beats']} beats" for s in segs)
                     + (f"  ·  LF/HF withheld for {', '.join(withheld)}" if withheld else "")
                     + (f"  ·  ⚠ {n_flag} segment(s) to check" if n_flag else ""))
        self.render()
        self.redraw_plot()
        self.group_refresh()

    def _compare(self) -> None:
        self.comps = reference_comparators(len(self.segs), self.ref)
        self.cmps = [compare_segments(self.segs[j], self.segs[k])
                     for j, k in comparison_pairs(self.comps)]
        self.flags = quality_flags(self.segs, None, comparators=self.comps)

    def _ref_picked(self, index: int) -> None:
        """A new reference: the comparisons, flags and table follow (the
        segment statistics are not recomputed)."""
        if not self.segs:
            return
        self.ref = list(self._ref_opts.values())[index]
        self._compare()
        res = self.win.recording.analysis.last_comparison
        if res is not None:
            res.cmps, res.flags, res.comparators = self.cmps, self.flags, list(self.comps)
        self.render()
        self.redraw_plot()

    # ── display ─────────────────────────────────────────────
    def render(self) -> None:
        segs, comps = self.segs, self.comps
        names = segment_names(segs)
        pairs = comparison_pairs(comps)
        lines = [segment_comparison_text(c, names[j], names[k]).split("\n")
                 for c, (j, k) in zip(self.cmps, pairs)]
        self.lbl_cmp.setText("\n".join(ln[0] for ln in lines)
                             + (f"\n{lines[0][1]}" if lines and len(lines[0]) > 1 else ""))
        single_ref = isinstance(comparators_reference(comps), int)
        self.lbl_ref.setText((comparators_caption(comps, names)
                              + ("  ·  the reference column is marked (ref.)"
                                 if single_ref else "")) if len(segs) > 1 else "")
        self.lbl_flags.setText("\n".join(
            f"{name}: " + (("⚠  " + "  ·  ".join(fl)) if fl else "✓  quality ok")
            for name, fl in zip(names, self.flags)))
        from ecg.app.compare_helpers import segment_color
        heads, cols = ["Metric", "Unit"], []
        n = len(segs)
        for k in range(n):
            heads.append(names[k] + (" (ref.)" if comps[k] is None and n > 1 and single_ref
                                     else ""))
            cols.append((k, None))
            if comps[k] is not None:
                j = comps[k]
                heads.append("Δ %" if single_ref else f"Δ % vs {names[j][:18]}")
                cols.append((k, j))
        self.table.clear()
        self.table.setColumnCount(len(heads))
        self.table.setHorizontalHeaderLabels(heads)
        for c, (k, j) in enumerate(cols, start=2):
            if j is None:
                self.table.horizontalHeaderItem(c).setForeground(QtGui.QColor(segment_color(k)))
        self.table.setRowCount(0)
        bold = QtGui.QFont()
        bold.setBold(True)
        for fam, items in by_family(METRICS, lambda m: family_of(m[1])):
            r = self.table.rowCount()
            self.table.insertRow(r)
            item = QtWidgets.QTableWidgetItem(fam.upper())
            item.setFont(bold)
            self.table.setItem(r, 0, item)
            self.table.setSpan(r, 0, 1, len(heads))
            for key, name, unit in items:
                r = self.table.rowCount()
                self.table.insertRow(r)
                self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
                self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(unit))
                vals = [float(s.get(key, float("nan"))) for s in segs]
                for c, (k, j) in enumerate(cols, start=2):
                    text = fmt_value(key, vals[k]) if j is None else fmt_delta(vals[j], vals[k])
                    it = QtWidgets.QTableWidgetItem(text)
                    it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                                        | QtCore.Qt.AlignmentFlag.AlignVCenter)
                    self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()

    def redraw_plot(self) -> None:
        if not self.segs:
            return
        from ecg.app.compare_helpers import draw_rr_tachograms, segment_color
        from ecg.app.poincare_figure import draw_poincare_set
        segs = self.segs
        colors = [segment_color(k) for k in range(len(segs))]
        if self.cb_view.currentIndex() == 1:
            n = len(segs)
            ncols = max(1, min(n, 4))           # _poincare_grid: at most 4 per row
            self.plot.update(lambda fig: draw_poincare_set(fig, segs, colors, jitter=self.jitter,
                                                           ncols=ncols))
        else:
            comps, names = list(self.comps), segment_names(segs)
            self.plot.update(lambda fig: draw_rr_tachograms(fig, segs, colors, comps, names))

    # ── export ──────────────────────────────────────────────
    def _result(self) -> "Optional[ComparisonResult]":
        res = self.win.recording.analysis.last_comparison
        if res is None or not self.segs:
            QtWidgets.QMessageBox.warning(self, "No results", "Run the comparison first.")
            return None
        res.subject = self.win.subject()
        return res

    def _base_name(self) -> str:
        from datetime import datetime
        return f"{self.win.subject() or 'subject'}_segments_{datetime.now():%Y%m%d_%H%M%S}"

    def export_excel(self) -> None:
        from datetime import datetime
        res = self._result()
        if res is None:
            return
        path, _f = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export the segment comparison", self._base_name() + ".xlsx",
            "Excel workbook (*.xlsx);;CSV (metric table) (*.csv);;All files (*)")
        if not path:
            return
        try:
            if path.lower().endswith(".csv"):
                write_csv(res, path)
            else:
                write_xlsx(res, path, exported_at=datetime.now().strftime("%Y-%m-%d %H:%M"))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export", str(exc))
            self.lbl_exp.setText(f"⚠  export failed: {exc}")
            return
        from pathlib import Path
        self.lbl_exp.setText(f"✓  saved → {Path(path).name}")

    def copy_table(self) -> None:
        res = self._result()
        if res is None:
            return
        QtWidgets.QApplication.clipboard().setText(comparison_tsv(res))
        self.lbl_exp.setText("✓  metric table copied — paste it into Excel or Prism")
