# -*- coding: utf-8 -*-
"""
ecg.qt.dialogs.agreement_dialog
-------------------------------
Check agreement's report (the Tk MethodAgreementDialog): the methods
compared, how many beats k of them confirmed, and every beat none of them
did, each with a View button that centres the Signal & Detection trace on
it. Non-modal, so several beats can be viewed in a row.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtWidgets

from ecg.qt import theme

if TYPE_CHECKING:
    from ecg.qt.main_window import MainWindow

MAX_ROWS = 300


def agreement_color(n_disagree: int, pct_confirmed: float) -> str:
    p = theme.current()
    return p.GREEN if n_disagree == 0 else (p.ORANGE if pct_confirmed >= 90 else p.RED)


class AgreementDialog(QtWidgets.QDialog):
    def __init__(self, window: "MainWindow", result: dict, fs: float) -> None:
        super().__init__(window)
        self.win = window
        self.setWindowTitle("Cross-Method Agreement")
        self.resize(520, 700)
        n_methods, n_beats, n_disagree = result["n_methods"], result["n_beats"], result["n_disagree"]
        pct = 100.0 * (n_beats - n_disagree) / max(n_beats, 1)
        lay = QtWidgets.QVBoxLayout(self)

        head = QtWidgets.QLabel(f"{pct:.0f}% of {n_beats} beats confirmed by ≥ 1 other method"
                                f"  ·  {n_disagree} unconfirmed")
        head.setStyleSheet(f"color: {agreement_color(n_disagree, pct)}; font-weight: 600;")
        lay.addWidget(head)

        box = QtWidgets.QGroupBox("Methods compared")
        bl = QtWidgets.QVBoxLayout(box)
        for m in result.get("methods_run", []):
            bl.addWidget(QtWidgets.QLabel(f"✔  {m}"))
        for m in result.get("methods_skipped", []):
            lbl = QtWidgets.QLabel(f"—  {m}")
            lbl.setProperty("muted", True)
            lbl.setWordWrap(True)
            bl.addWidget(lbl)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("Agreement breakdown")
        grid = QtWidgets.QGridLayout(box)
        histogram = result.get("histogram") or []
        max_count = max(histogram) if histogram else 1
        p = theme.current()
        self.bars: "list[QtWidgets.QProgressBar]" = []
        for row, k in enumerate(range(len(histogram) - 1, -1, -1)):
            label = (f"All {n_methods} agree" if k == n_methods
                     else "None agree" if k == 0 else f"{k} of {n_methods} agree")
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, max(max_count, 1))
            bar.setValue(histogram[k])
            bar.setTextVisible(False)
            col = p.GREEN if k == n_methods else (p.RED if k == 0 else p.ORANGE)
            bar.setStyleSheet(f"QProgressBar::chunk {{ background: {col}; }}")
            grid.addWidget(bar, row, 1)
            grid.addWidget(QtWidgets.QLabel(str(histogram[k])), row, 2)
            self.bars.append(bar)
        lay.addWidget(box)

        lay.addWidget(QtWidgets.QLabel(f"Unconfirmed beats ({n_disagree})"))
        samples = result.get("disagree_samples")
        self.times = sorted(float(s) / fs for s in samples) if samples is not None else []
        if not self.times:
            ok = QtWidgets.QLabel("✔  Every accepted beat was confirmed by at least one "
                                  "other method.")
            ok.setStyleSheet(f"color: {p.GREEN};")
            lay.addWidget(ok)
            lay.addStretch(1)
        else:
            self.table = QtWidgets.QTableWidget(min(len(self.times), MAX_ROWS), 2)
            self.table.setHorizontalHeaderLabels(["Time (s)", ""])
            self.table.verticalHeader().setVisible(False)
            self.table.horizontalHeader().setStretchLastSection(True)
            for i, t in enumerate(self.times[:MAX_ROWS]):
                self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(f"t = {t:.3f} s"))
                b = QtWidgets.QPushButton("→ View")
                b.clicked.connect(lambda _c=False, t=t: self.jump_to(t))
                self.table.setCellWidget(i, 1, b)
            lay.addWidget(self.table, 1)
            if len(self.times) > MAX_ROWS:
                more = QtWidgets.QLabel(f"… {len(self.times) - MAX_ROWS} more not shown "
                                        f"(showing the first {MAX_ROWS} by time)")
                more.setProperty("muted", True)
                lay.addWidget(more)
        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.close)
        lay.addWidget(close)

    def jump_to(self, t: float) -> None:
        """Centre the Signal & Detection trace on beat time *t* (s), the
        window width kept."""
        det = self.win.detection
        t0, t1 = det.plot.window()
        width = max(t1 - t0, 0.1)
        time = self.win.recording.signal.time
        end = float(time[-1]) if time is not None and len(time) else t + width
        start = max(0.0, min(t - width / 2, max(0.0, end - width)))
        self.win.tabs.setCurrentWidget(det)
        det.plot.set_window(start, start + width)
        self.raise_()
