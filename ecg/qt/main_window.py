# -*- coding: utf-8 -*-
"""
ecg.qt.main_window
------------------
The main window: a header (file, Open, theme, progress and Cancel), the
eight tabs of the Tk interface in the same order, and a status bar.

It owns the Recording (ecg.core.recording) the tabs read and write, and
runs their background jobs (ecg.qt.workers): a job's result lands only if
the recording is still the one it started on (the "generation" guard of
the Tk interface), and Cancel stops the running jobs at their next
progress checkpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets

import ecg
from ecg.core.analysis import analyse_hrv_freq, analyse_hrv_nonlinear
from ecg.core.recording import Recording
from ecg.qt import theme
from ecg.qt.figure_host import FigureHost, sync_figure_theme
from ecg.qt.tabs.detection_tab import DetectionTab
from ecg.qt.tabs.rhythm_tab import RhythmTab
from ecg.qt.workers import Job, JobRunner
from ecg.services.analysis_service import run_core_analysis
from ecg.ui.tab_keys import TAB_DETECTION, TAB_HRV, TAB_ORDER

FILE_FILTER = "Recordings (*.mat *.smr);;All files (*)"


class MainWindow(QtWidgets.QMainWindow):
    recording_changed = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"ECG Analysis {ecg.__version__} (Qt preview)")
        self.resize(1400, 860)
        self.setMinimumSize(1100, 640)
        self.recording = Recording()
        self.generation = 0
        self.jobs = JobRunner(self)
        self._current_job: "Optional[Job]" = None
        self.figures = FigureHost(self)          # the shared figure code's "app"

        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.addWidget(self._build_header())
        self.tabs = QtWidgets.QTabWidget()
        self.tab_widgets: "dict[str, QtWidgets.QWidget]" = {}
        for name in TAB_ORDER:
            if name == TAB_DETECTION:
                w = DetectionTab(self)
            elif name == TAB_HRV:
                w = RhythmTab(self)
            else:
                w = self._placeholder(name)
            self.tab_widgets[name] = w
            self.tabs.addTab(w, name.replace("&", "&&"))   # "&" alone marks a shortcut
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.setStatusBar(QtWidgets.QStatusBar())
        self.detection: DetectionTab = self.tab_widgets[TAB_DETECTION]   # type: ignore[assignment]
        self.rhythm: RhythmTab = self.tab_widgets[TAB_HRV]                # type: ignore[assignment]

        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Open, self, activated=self.open_dialog)
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, self, activated=self.detection.undo)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Y"), self, activated=self.detection.redo)

    # ── layout ──────────────────────────────────────────────
    def _build_header(self) -> QtWidgets.QFrame:
        bar = QtWidgets.QFrame(objectName="Header")
        row = QtWidgets.QHBoxLayout(bar)
        title = QtWidgets.QLabel("<b>ECG Analysis</b>")
        row.addWidget(title)
        self.lbl_file = QtWidgets.QLabel("No recording")
        self.lbl_file.setProperty("muted", True)
        row.addWidget(self.lbl_file, 1)
        self.btn_open = QtWidgets.QPushButton("Open…")
        self.btn_open.clicked.connect(self.open_dialog)
        row.addWidget(self.btn_open)
        self.btn_analyze = QtWidgets.QPushButton("Analyze")
        self.btn_analyze.setProperty("primary", True)
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setToolTip("RR / HR, time-domain HRV and beat template, then frequency "
                                    "and non-linear HRV")
        self.btn_analyze.clicked.connect(self.analyze)
        row.addWidget(self.btn_analyze)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFixedWidth(220)
        self.progress.setVisible(False)
        row.addWidget(self.progress)
        self.lbl_progress = QtWidgets.QLabel("")
        self.lbl_progress.setProperty("muted", True)
        row.addWidget(self.lbl_progress)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self.cancel_jobs)
        row.addWidget(self.btn_cancel)
        self.btn_theme = QtWidgets.QPushButton("Dark mode")
        self.btn_theme.clicked.connect(self.toggle_theme)
        row.addWidget(self.btn_theme)
        return bar

    @staticmethod
    def _placeholder(name: str) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lbl = QtWidgets.QLabel(f"{name}: moves to the Qt interface in a later step of the "
                               "migration. Use the current interface (python -m ecg) meanwhile.")
        lbl.setProperty("muted", True)
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lbl)
        return w

    # ── recording ───────────────────────────────────────────
    def open_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open a recording", "", FILE_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> None:
        """Make *path* the recording; results of jobs still running for the
        previous one are discarded when they land."""
        self.cancel_jobs()
        self.generation += 1
        self.recording = Recording()
        self.recording.signal.filepath = str(path)
        self.figures._generation = self.generation
        self.btn_analyze.setEnabled(False)
        self.rhythm.clear()
        self.lbl_file.setText(Path(path).name)
        self.lbl_file.setToolTip(str(path))
        self.detection.file_opened()
        self.recording_changed.emit()
        self.set_status(f"Opened {Path(path).name} — click Detect Peaks")

    # ── Analyze ─────────────────────────────────────────────
    def peaks_changed(self) -> None:
        """Detection or a hand edit changed the peaks: results are stale."""
        has_peaks = self.recording.detection.rpeaks_ok is not None
        self.btn_analyze.setEnabled(has_peaks and len(self.recording.detection.rpeaks_ok) >= 5)
        if self.recording.analysis.results is None:
            self.rhythm.clear()

    def analysis_params(self) -> dict:
        """AnalysisController.run_analysis's snapshot."""
        ctl = self.figures.analysis_ctrl
        params = self.detection.params()
        params["hrv_rr_bounds"] = ctl.hrv_rr_bounds()
        params["break_tol_samples"] = ctl.break_tol_samples()
        params["noise_regions"] = [dict(r) for r in self.recording.analysis.noise_regions or []]
        return params

    def analyze(self) -> None:
        """Analyze: core analysis, then frequency and non-linear HRV, each
        drawn when it lands (as the Tk interface's chain)."""
        rec = self.recording
        d, s = rec.detection, rec.signal
        if d.rpeaks_ok is None or len(d.rpeaks_ok) < 5:
            self.set_status("Too few peaks — adjust the detection first.", warn=True)
            return
        self.btn_analyze.setEnabled(False)
        self.run_job("Analyze", run_core_analysis, d.rpeaks_ok.copy(), s.filtered, s.fs,
                     self.analysis_params(), rec.analysis.artifact_report,
                     on_done=self._core_done, on_end=lambda: self.btn_analyze.setEnabled(True))

    def _core_done(self, bundle: dict) -> None:
        """AnalysisController.on_analysis_done's state writes, then drawing."""
        a, d = self.recording.analysis, self.recording.detection
        a.results = bundle["results"]
        d.rpeaks_ok = bundle["rpeaks_ok"]
        a.artifact_report = bundle["artifact_report"]
        a.analysed_window = bundle.get("analysis_window")
        self.detection.refresh()
        hr = a.results.get("hr") or {}
        self.rhythm.lbl_info.setText(
            f"{len(bundle['rpeaks_analysed']):,} beats analysed · mean HR "
            f"{hr.get('mean', float('nan')):.1f} bpm")
        self.figures.plot_ctrl.draw_core_results(on_complete=self._run_freq)

    def _run_freq(self) -> None:
        ctl, s = self.figures.analysis_ctrl, self.recording.signal
        rp = self.figures._windowed_peaks()
        lo, hi = ctl.results_rr_bounds()
        self.figures.lbl_freq_status.configure(text="Computing…")
        self.run_job("Frequency HRV", analyse_hrv_freq, rp, s.fs, rr_min_ms=lo, rr_max_ms=hi,
                     breaks=ctl.hrv_breaks(), bands=ctl.get_freq_bands(),
                     on_done=self._freq_done)

    def _freq_done(self, df) -> None:
        r = self.recording.analysis.results
        if r is None:
            return
        r["hrv_freq"] = df
        text, color = self.figures.analysis_ctrl._freq_status(df, just_computed=True)
        self.figures.lbl_freq_status.configure(text=text.strip(), text_color=color)
        pc = self.figures.plot_ctrl
        pc.run_plot_chain([("PSD", lambda: pc.plot_psd(r)),
                           ("HRV tables (freq)", lambda: pc.plot_hrv_tables(r)),
                           ("Metrics", lambda: pc.plot_table_views(r))],
                          on_complete=self._run_nonlinear)

    def _run_nonlinear(self) -> None:
        ctl, s = self.figures.analysis_ctrl, self.recording.signal
        rp = self.figures._windowed_peaks()
        lo, hi = ctl.results_rr_bounds()
        self.figures.lbl_nonlin_status.configure(text="Computing SampEn / DFA…")
        self.run_job("Non-linear HRV", analyse_hrv_nonlinear, rp, s.fs,
                     max_beats=ctl.entropy_cap(), rr_min_ms=lo, rr_max_ms=hi,
                     breaks=ctl.hrv_breaks(), on_done=self._nonlinear_done)

    def _nonlinear_done(self, df) -> None:
        r = self.recording.analysis.results
        if r is None:
            return
        r["hrv_nonlin"] = df
        text, color = self.figures.analysis_ctrl._nonlin_status(df, just_computed=True)
        self.figures.lbl_nonlin_status.configure(text=text.strip(), text_color=color)
        pc = self.figures.plot_ctrl
        pc.run_plot_chain([("Non-linear metrics", lambda: pc.plot_nonlinear(r)),
                           ("Metrics", lambda: pc.plot_table_views(r))],
                          on_complete=lambda: self.set_status("Analysis done"))

    def redraw_results(self) -> None:
        """Draw the current results again (after a theme change)."""
        r = self.recording.analysis.results
        if r is None:
            return
        pc = self.figures.plot_ctrl
        tasks = [("RR / HR tachogram", lambda: pc.plot_rr(r)),
                 ("HRV tables", lambda: pc.plot_hrv_tables(r)),
                 ("Poincaré", lambda: pc.plot_nonlinear(r))]
        if r.get("hrv_freq") is not None and not r["hrv_freq"].empty:
            tasks.append(("PSD", lambda: pc.plot_psd(r)))
        pc.run_plot_chain(tasks)

    # ── jobs ────────────────────────────────────────────────
    def run_job(self, what: str, fn: Callable[..., Any], *args: Any,
                on_done: Callable[[Any], None],
                on_end: "Optional[Callable[[], None]]" = None, **kwargs: Any) -> Job:
        """Run ``fn(*args, progress_cb=..., **kwargs)`` in the background.
        *on_done(result)* runs on the main thread if the recording is still
        the one the job started on; *on_end()* always runs once it ends."""
        gen = self.generation
        job = Job(fn, *args, **kwargs)
        self._current_job = job
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.btn_cancel.setVisible(True)
        self.lbl_progress.setText(what + "…")

        def _end() -> None:
            if self._current_job is job:
                self._current_job = None
                self.progress.setVisible(False)
                self.btn_cancel.setVisible(False)
                self.lbl_progress.setText("")
            if on_end is not None:
                on_end()

        def _finished(result: Any) -> None:
            _end()
            if gen != self.generation:
                self.set_status(f"{what}: result discarded — another recording was opened",
                                warn=True)
                return
            on_done(result)

        def _failed(msg: str, tb: str) -> None:
            _end()
            self.set_status(f"{what} failed: {msg}", warn=True)
            if gen == self.generation:
                box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Warning, what,
                                            f"{what} failed:\n\n{msg}", parent=self)
                box.setDetailedText(tb)
                box.open()

        def _cancelled() -> None:
            _end()
            self.set_status(f"{what} cancelled", warn=True)

        job.signals.progress.connect(self._progress)
        job.signals.finished.connect(_finished)
        job.signals.failed.connect(_failed)
        job.signals.cancelled.connect(_cancelled)
        return self.jobs.start(job)

    def _progress(self, pct: int, msg: str) -> None:
        self.progress.setValue(pct)
        self.lbl_progress.setText(msg)

    def cancel_jobs(self) -> None:
        self.jobs.cancel_all()

    # ── status, theme ───────────────────────────────────────
    def set_status(self, text: str, warn: bool = False) -> None:
        p = theme.current()
        self.statusBar().setStyleSheet(f"color: {p.WARN_TEXT if warn else p.MUTED};")
        self.statusBar().showMessage(text)

    def toggle_theme(self) -> None:
        app = QtWidgets.QApplication.instance()
        p = theme.apply(app, theme.LIGHT.name if theme.current().is_dark else theme.DARK.name)
        self.btn_theme.setText("Light mode" if p.is_dark else "Dark mode")
        sync_figure_theme(p.is_dark)
        self.detection.apply_palette(p)
        self.redraw_results()

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        self.cancel_jobs()
        self.jobs.wait(5000)
        super().closeEvent(ev)
