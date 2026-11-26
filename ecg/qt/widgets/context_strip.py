# -*- coding: utf-8 -*-
"""
ecg.qt.widgets.context_strip
----------------------------
The context strip under the header, as in the Tk interface: what the
results rest on (analysed or not, the analysis window with its editor,
peaks, RR bounds, RR excluded, artifact correction, QTc) and when they were
computed. The fields come from ecg.app.context_fields, shared with Tk.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PySide6 import QtWidgets

from ecg.app.context_fields import TONE_HEAD, TONE_MUTED, TONE_WARN, context_fields, times_text

if TYPE_CHECKING:
    from ecg.qt.main_window import MainWindow


class ContextStrip(QtWidgets.QFrame):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window, objectName="Strip")
        self.win = window
        self.setFixedHeight(40)
        self._row = QtWidgets.QHBoxLayout(self)
        self._row.setContentsMargins(16, 0, 16, 0)
        self._row.setSpacing(14)
        self._labels: "dict[str, QtWidgets.QLabel]" = {}
        self._editor: "Optional[QtWidgets.QWidget]" = None
        for key in ("head", "window", "beats", "rr", "excluded", "artifact", "qtc"):
            lbl = QtWidgets.QLabel("")
            self._labels[key] = lbl
            self._row.addWidget(lbl)
        self._row.addStretch(1)
        self.lbl_times = QtWidgets.QLabel("")
        self.lbl_times.setProperty("muted", True)
        self._row.addWidget(self.lbl_times)
        self._cache: tuple = ()

    def attach_window_editor(self, det) -> None:
        """The Signal page's analysis-window entries, Apply and Full, in place
        of the window field (as the Tk strip)."""
        ed = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(ed)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(QtWidgets.QLabel("Window"))
        for w in (det.ed_win0,):
            w.setMaximumWidth(70)
            row.addWidget(w)
        row.addWidget(QtWidgets.QLabel("–"))
        det.ed_win1.setMaximumWidth(70)
        row.addWidget(det.ed_win1)
        row.addWidget(QtWidgets.QLabel("s"))
        apply = QtWidgets.QPushButton("Apply")
        apply.clicked.connect(det.apply_analysis_window)
        full = QtWidgets.QPushButton("Full")
        full.setToolTip("The whole signal")
        full.clicked.connect(det.reset_analysis_window)
        for b in (apply, full):
            b.setStyleSheet("padding: 2px 10px;")
            row.addWidget(b)
        i = self._row.indexOf(self._labels["window"])
        self._labels["window"].hide()
        self._row.insertWidget(i, ed)
        self._editor = ed

    def refresh(self) -> None:
        """The fields from the current state (cheap: the metric tables are
        rebuilt only when the results, the report or the peaks change)."""
        from ecg.core.metric_table import build_metric_table
        w = self.win
        host, a, sig = w.figures, w.recording.analysis, w.recording.signal
        r, rep, rp = a.results, a.artifact_report, w.recording.detection.rpeaks_ok
        if not self._cache or self._cache[0] is not r or self._cache[1] is not rep \
                or self._cache[2] is not rp:
            table = build_metric_table(r, asymmetry=None) if r is not None else None
            art = build_metric_table(
                None, artifact_report=host.analysis_ctrl.current_artifact_report())
            self._cache = (r, rep, rp, table, art)
        table, art = self._cache[3], self._cache[4]
        ctrl = host.plot_ctrl
        peaks = host._windowed_peaks()
        analysed = ctrl._report_window() if r is not None else (a.t_start, a.t_end)
        fields = context_fields(
            r, table, art, has_signal=sig.filtered is not None or sig.raw is not None,
            n_peaks=len(peaks) if peaks is not None else None, window=analysed,
            signal_end_s=ctrl._signal_end_s(),
            current_rr=host.analysis_ctrl.hrv_rr_bounds(quiet=True),
            qtc_formula=host._qtc_formula(), noisy=bool(getattr(a, "noise_regions", None)))
        from ecg.qt import theme
        p = theme.current()
        shown = set()
        for f in fields:
            lbl = self._labels.get(f.key)
            if lbl is None or (f.key == "window" and self._editor is not None):
                continue
            shown.add(f.key)
            color = {TONE_HEAD: p.ACCENT_TEXT, TONE_WARN: p.WARN_TEXT,
                     TONE_MUTED: p.MUTED}.get(f.tone, p.TEXT)
            weight = "600" if f.tone == TONE_HEAD else "normal"
            style = f"color: {color}; font-weight: {weight};"
            if lbl.text() != f.text:
                lbl.setText(f.text)
            if lbl.styleSheet() != style:
                lbl.setStyleSheet(style)
            lbl.show()
        for key, lbl in self._labels.items():
            if key not in shown and not (key == "window" and self._editor is not None):
                lbl.hide()
        times = times_text(host.ui.computed_at, host.ui.peaks_edited_at)
        if self.lbl_times.text() != times:
            self.lbl_times.setText(times)
