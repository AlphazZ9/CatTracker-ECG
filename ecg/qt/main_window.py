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

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets

import ecg
from ecg.core.analysis import analyse_hrv_freq, analyse_hrv_nonlinear
from ecg.core.recording import Recording
from ecg.qt import theme
from ecg.qt.figure_host import FigureHost, sync_figure_theme
from ecg.qt.tabs.detection_tab import DetectionTab
from ecg.qt.tabs.events_tab import EventsTab
from ecg.qt.tabs.report_tab import ReportTab
from ecg.qt.tabs.morphology_tab import MorphologyTab
from ecg.qt.tabs.time_course_tab import TimeCourseTab
from ecg.qt.tabs.rhythm_tab import RhythmTab
from ecg.qt.tabs.compare_tab import CompareTab
from ecg.qt.tabs.group_tab import GroupTab
from ecg.qt.workers import Job, JobRunner
from ecg.services.analysis_service import run_core_analysis
from ecg.ui.tab_keys import (TAB_COMPARE, TAB_DETECTION, TAB_GROUP, TAB_EVENTS, TAB_HRV, TAB_MORPHOLOGY, TAB_ORDER,
                             TAB_REPORT,
                             TAB_TIME_COURSE, TIME_COURSE_MODES)

FILE_FILTER = "Recordings (*.mat *.smr);;All files (*)"
SHORTCUTS_TEXT = (
    "Space          Detect Peaks\n"
    "Ctrl+R         Analyze\n"
    "Ctrl+O         Open a recording\n"
    "Ctrl+S         Save session\n"
    "Ctrl+E         Export Excel\n"
    "Ctrl+Shift+E   Export RR intervals CSV\n"
    "Ctrl+M         Compare segments\n"
    "Ctrl+Z / Y     Undo / Redo peak edits\n"
    "←  / →         Move the ECG view (Signal & Detection tab)\n"
    "F1             This help\n")


log = logging.getLogger("ecg")

class _TabSwitcher:
    """app.tabs.set(name) / .get() of the Tk interface, on the Qt tab widget."""

    def __init__(self, tabs: QtWidgets.QTabWidget) -> None:
        self._tabs = tabs

    def set(self, name: str) -> None:
        for i in range(self._tabs.count()):
            if self._tabs.tabText(i).replace("&&", "&") == name:
                self._tabs.setCurrentIndex(i)

    def get(self) -> str:
        return self._tabs.tabText(self._tabs.currentIndex()).replace("&&", "&")


class MainWindow(QtWidgets.QMainWindow):
    recording_changed = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"ECG Analysis {ecg.__version__} (Qt preview)")
        self.resize(1400, 860)
        self.setMinimumSize(1100, 640)
        self.recording = Recording()
        self.session_dirty = False
        self.generation = 0
        self.peaks_version = 0          # +1 each time the peaks change (peaks_changed)
        self.jobs = JobRunner(self)
        self._active: "list[tuple[Job, str]]" = []     # the jobs running, oldest first
        from ecg.qt.errors import ErrorReporter
        self.errors = ErrorReporter(self)
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
            elif name == TAB_MORPHOLOGY:
                w = MorphologyTab(self)
            elif name == TAB_TIME_COURSE:
                w = TimeCourseTab(self)
            elif name == TAB_EVENTS:
                w = EventsTab(self)
            elif name == TAB_GROUP:
                w = GroupTab(self)
            elif name == TAB_COMPARE:
                w = CompareTab(self)
            elif name == TAB_REPORT:
                w = ReportTab(self)
            else:
                w = self._placeholder(name)
            self.tab_widgets[name] = w
            self.tabs.addTab(w, name.replace("&", "&&"))   # "&" alone marks a shortcut
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.setStatusBar(QtWidgets.QStatusBar())
        self.detection: DetectionTab = self.tab_widgets[TAB_DETECTION]   # type: ignore[assignment]
        self.rhythm: RhythmTab = self.tab_widgets[TAB_HRV]                # type: ignore[assignment]
        self.morphology: MorphologyTab = self.tab_widgets[TAB_MORPHOLOGY]  # type: ignore[assignment]
        self.time_course: TimeCourseTab = self.tab_widgets[TAB_TIME_COURSE]  # type: ignore[assignment]
        self.events: EventsTab = self.tab_widgets[TAB_EVENTS]  # type: ignore[assignment]
        self.report: ReportTab = self.tab_widgets[TAB_REPORT]  # type: ignore[assignment]
        self.compare: CompareTab = self.tab_widgets[TAB_COMPARE]  # type: ignore[assignment]
        self.group: GroupTab = self.tab_widgets[TAB_GROUP]  # type: ignore[assignment]
        from ecg.qt.dialogs.settings_dialog import SettingsDialog
        from ecg.qt.session import QtSession
        self.settings = SettingsDialog(self)
        from ecg.qt.dialogs.recording_dialog import RecordingDialog
        self.recording_sheet = RecordingDialog(self)
        self.session = QtSession(self)
        self.tabs.currentChanged.connect(
            lambda i: self.report.refresh() if self.tabs.widget(i) is self.report else None)
        self.tabs.currentChanged.connect(       # the group table may have changed
            lambda i: self.compare.group_refresh() if self.tabs.widget(i) is self.compare
            else None)
        # What the Tk controllers call to show a tab / a Time course mode.
        self.figures.tabs = _TabSwitcher(self.tabs)
        self.figures._show_time_mode = lambda mode: self.time_course.setCurrentIndex(
            TIME_COURSE_MODES.index(mode) if mode in TIME_COURSE_MODES else 0)

        self._build_menu()
        # The out-of-date banners follow the state on a tick, as Tk's shell
        # tick: only the tab shown is looked at (cheap: freshness.status).
        self._tab_state_shown: "dict[str, tuple]" = {}
        self._banner_timer = QtCore.QTimer(self, interval=1000)
        self._banner_timer.timeout.connect(self.refresh_banners)
        self._banner_timer.start()
        self.tabs.currentChanged.connect(lambda _i: self.refresh_banners())
        QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, self, activated=self.detection.undo)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Y"), self, activated=self.detection.redo)
        # The Tk interface's shortcuts (Help > Keyboard shortcuts).
        for key, fn in (("Space", self._shortcut_detect), ("Ctrl+R", self._shortcut_analyze),
                        ("Ctrl+E", lambda: self.figures.export_ctrl.export_excel()),
                        ("Ctrl+Shift+E", lambda: self.figures.export_ctrl.export_rr_csv()),
                        ("Ctrl+M", lambda: self.tabs.setCurrentWidget(self.compare)),
                        ("Left", lambda: self._shortcut_pan(-1)),
                        ("Right", lambda: self._shortcut_pan(+1))):
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=fn)

    # ── layout ──────────────────────────────────────────────
    def _build_menu(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&File")
        for text, key, fn in (("Open…", QtGui.QKeySequence.StandardKey.Open, self.open_dialog),
                              ("Save Session", QtGui.QKeySequence.StandardKey.Save,
                               lambda: self.session.save()),
                              (None, None, None),
                              ("Recent recordings…", None, self.open_recent),
                              ("Recording (subject, project, notes)…", None,
                               lambda: (self.recording_sheet.show(),
                                        self.recording_sheet.raise_())),
                              (None, None, None),
                              ("Quit", QtGui.QKeySequence.StandardKey.Quit, self.close)):
            if text is None:
                m.addSeparator()
                continue
            act = m.addAction(text)
            if key is not None:
                act.setShortcut(key)
            act.triggered.connect(fn)
            if text == "Open…":
                self.menu_recent = m.addMenu("Open Recent")
                self.menu_recent.aboutToShow.connect(self._fill_recent_menu)
        m.insertAction(m.actions()[-1], m.addSeparator())
        clear = QtGui.QAction("Clear session cache…", self)
        clear.triggered.connect(self.clear_session_cache)
        m.insertAction(m.actions()[-2], clear)
        v = mb.addMenu("&View")
        v.addAction("Light / dark mode").triggered.connect(self.toggle_theme)
        v.addAction("Text size…").triggered.connect(self.choose_text_size)
        h = mb.addMenu("&Help")
        act = h.addAction("Keyboard shortcuts")
        act.setShortcut(QtGui.QKeySequence("F1"))
        act.triggered.connect(self.show_shortcuts)
        h.addAction("Open log folder").triggered.connect(self.open_log_folder)
        h.addAction("About").triggered.connect(lambda: QtWidgets.QMessageBox.about(
            self, "ECG Analysis", f"ECG Analysis {ecg.__version__}\nQt interface (PySide6)."))

    def _fill_recent_menu(self) -> None:
        """The last files opened (newest first), then the registry view."""
        import os
        m = self.menu_recent
        m.clear()
        paths = [p for p in self.figures.session.recent_files if os.path.exists(p)]
        if not paths:
            from ecg.qt.dialogs.recent_dialog import recent_rows
            paths = [r["filepath"] for r in recent_rows([])][:8]
        if not paths:
            m.addAction("No recent files yet").setEnabled(False)
        for p in paths:
            act = m.addAction(Path(p).name)
            act.setToolTip(p)
            act.triggered.connect(lambda _c=False, p=p: self.open_path(p))
        m.addSeparator()
        m.addAction("All recent recordings (stats, cohort trends)…").triggered.connect(
            self.open_recent)

    def open_recent(self) -> None:
        from ecg.qt.dialogs.recent_dialog import RecentDialog
        self.recent_dialog = RecentDialog(self)
        self.recent_dialog.show()

    def open_log_folder(self) -> None:
        from ecg.qt.errors import default_log_dir
        d = default_log_dir()
        d.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(d)))

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
        self.btn_recording = QtWidgets.QPushButton("Recording…")
        self.btn_recording.setToolTip("Subject ID, sex, project and notes of this recording")
        self.btn_recording.clicked.connect(
            lambda: (self.recording_sheet.show(), self.recording_sheet.raise_()))
        row.addWidget(self.btn_recording)
        self.btn_settings = QtWidgets.QPushButton("Settings…")
        self.btn_settings.setToolTip("Channel, crop, filters, RR bounds, artifact correction")
        self.btn_settings.clicked.connect(lambda: (self.settings.show(), self.settings.raise_()))
        row.addWidget(self.btn_settings)
        self.btn_ml = QtWidgets.QPushButton("ML detector…")
        self.btn_ml.setToolTip("Training examples and training of the ML R-peak detector")
        self.btn_ml.clicked.connect(self.open_ml_dialog)
        row.addWidget(self.btn_ml)
        self.btn_save = QtWidgets.QPushButton("Save Session")
        self.btn_save.setToolTip("Save the peaks, edits, results and settings of this recording; "
                                 "they are offered again when the file is reopened")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(lambda: self.session.save())
        row.addWidget(self.btn_save)
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
        if not self.confirm_discard("Open another recording"):
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open a recording", "", FILE_FILTER)
        if path:
            self.open_path(path)

    def confirm_discard(self, action: str) -> bool:
        """Unsaved changes to this recording (peaks, edits, results, regions):
        Save / Discard / Cancel. True when *action* may go on."""
        if not self.session_dirty or self.recording.signal.filtered is None:
            return True
        B = QtWidgets.QMessageBox.StandardButton
        answer = QtWidgets.QMessageBox.question(
            self, action, f"{Path(self.recording.signal.filepath or '').name} has unsaved "
            "changes (peaks, edits, results or regions).\n\nSave the session first?",
            B.Save | B.Discard | B.Cancel, B.Save)
        if answer == B.Cancel:
            return False
        if answer == B.Save:
            return self.session.save_now()
        return True

    def open_path(self, path: str, check_session: bool = True) -> None:
        """Make *path* the recording; results of jobs still running for the
        previous one are discarded when they land."""
        self.cancel_jobs()
        self.generation += 1
        self.recording = Recording()
        self.recording.signal.filepath = str(path)
        self.session_dirty = False
        self.figures.freshness.reset()                 # SignalController.load_path
        self.figures.ui.computed_at = None
        self.figures.ui.peaks_edited_at = None
        self.figures._generation = self.generation
        self.btn_analyze.setEnabled(False)
        self.rhythm.clear()
        self.morphology.results_changed()
        self.time_course.results_changed()
        self.events.clear()
        self.events.peaks_changed()
        self.lbl_file.setToolTip(str(path))
        # Notes, subject and sex from the file name, and the recent list
        # (SignalController.load_path); a saved session then restores its own.
        from ecg.services.recording_fields import autofill_subject_from_name
        self.recording_sheet.file_opened(str(path))
        kept_note = autofill_subject_from_name(self.figures, str(path))
        self.session.ctrl.add_recent(str(path))
        self.update_file_label()
        self.detection.file_opened()
        self.compare.file_opened()
        self.btn_save.setEnabled(False)
        self.recording_changed.emit()
        self.set_status(kept_note or f"Opened {Path(path).name} — click Detect Peaks",
                        warn=bool(kept_note))
        if check_session:
            self.session.check(str(path))

    def subject(self) -> str:
        """The Subject ID of the Recording sheet (filled from the file name,
        as in Tk, unless typed there)."""
        return self.figures.ent_subject.get().strip()

    def update_file_label(self) -> None:
        """The header's file chip: name · channel · rate · subject, and the
        project (ECGApp._update_topbar_filename / _on_project_name_change)."""
        s = self.recording.signal
        if not s.filepath:
            self.lbl_file.setText("No recording")
            return
        text = Path(s.filepath).name
        if s.channel is not None:
            channel = str(s.channel)
            if len(channel) > 12:
                channel = channel.rsplit("_", 1)[-1]
            text += f" · {channel} · {s.fs:g} Hz"
        subject = self.subject()
        if subject:
            text += f" · {subject}"
        project = self.figures.ent_project_name.get().strip()
        if project:
            text = f"[{project}]  {text}"
        self.lbl_file.setText(text)

    # ── Analyze ─────────────────────────────────────────────
    def open_ml_dialog(self) -> None:
        from ecg.qt.dialogs.ml_training import MLTrainingDialog
        MLTrainingDialog(self).exec()

    def results_restored(self) -> None:
        """Results brought back by a session: shown as after Analyze."""
        a = self.recording.analysis
        hr = a.results.get("hr") or {}
        n = len(self.figures._windowed_peaks() if self.figures._windowed_peaks() is not None else [])
        self.rhythm.lbl_info.setText(f"{n:,} beats analysed · mean HR "
                                     f"{hr.get('mean', float('nan')):.1f} bpm (from the session)")
        self.morphology.results_changed()
        self.time_course.results_changed()
        self.figures.plot_ctrl.draw_all_results()

    def peaks_changed(self) -> None:
        """Detection or a hand edit changed the peaks: results are stale."""
        self.session_dirty = True
        self.peaks_version += 1
        self.btn_save.setEnabled(self.recording.signal.filtered is not None)
        has_peaks = self.recording.detection.rpeaks_ok is not None
        self.btn_analyze.setEnabled(has_peaks and len(self.recording.detection.rpeaks_ok) >= 5)
        if self.recording.analysis.results is None:
            self.rhythm.clear()
            self.morphology.results_changed()
        self.time_course.results_changed()
        self.events.peaks_changed()
        self.compare.peaks_changed()

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
                     peaks_bound=True, on_done=self._core_done,
                     on_end=lambda: self.btn_analyze.setEnabled(True))

    def _core_done(self, bundle: dict) -> None:
        """AnalysisController.on_analysis_done's state writes, then drawing."""
        a, d = self.recording.analysis, self.recording.detection
        a.results = bundle["results"]
        d.rpeaks_ok = bundle["rpeaks_ok"]
        a.artifact_report = bundle["artifact_report"]
        a.analysed_window = bundle.get("analysis_window")
        self.session_dirty = True
        # AnalysisController.on_analysis_done: the basis of the core results,
        # from the bundle (the correction can change the peaks).
        import datetime as dt
        from ecg.ui.freshness import recorded_rr_bounds
        fresh = self.figures.freshness
        fresh.stamp("core", fresh.capture(window=a.analysed_window,
                                          rr_bounds=recorded_rr_bounds(a.results)))
        self.figures.ui.computed_at = dt.datetime.now().isoformat(timespec="seconds")
        self._auto_epochs = bool(bundle.get("auto_epochs", False))
        self.detection.refresh()
        hr = a.results.get("hr") or {}
        self.rhythm.lbl_info.setText(
            f"{len(bundle['rpeaks_analysed']):,} beats analysed · mean HR "
            f"{hr.get('mean', float('nan')):.1f} bpm")
        self.morphology.results_changed()
        self.time_course.results_changed()
        self.morphology.wave_template()       # the saved default, first time (as Tk)
        self.figures.plot_ctrl.draw_core_results(
            on_complete=lambda: self._run_freq(then=lambda: self._run_nonlinear(
                then=self._analysis_chain_done)))

    def _run_freq(self, then: "Optional[Callable[[], None]]" = None) -> None:
        """Frequency HRV of the analysis window (AnalysisController.run_freq),
        then *then* once drawn: Analyze's chain, or nothing for Recompute."""
        self._freq_then = then
        ctl, s = self.figures.analysis_ctrl, self.recording.signal
        rp = self.figures._windowed_peaks()
        lo, hi = ctl.results_rr_bounds()
        self._freq_basis = self.figures.freshness.capture(rr_bounds=(lo, hi))
        self.figures.lbl_freq_status.configure(text="Computing…")
        self.run_job("Frequency HRV", analyse_hrv_freq, rp, s.fs, rr_min_ms=lo, rr_max_ms=hi,
                     breaks=ctl.hrv_breaks(), bands=ctl.get_freq_bands(),
                     peaks_bound=True, on_done=self._freq_done)

    def _freq_done(self, df) -> None:
        r = self.recording.analysis.results
        if r is None:
            return
        r["hrv_freq"] = df
        self.figures.freshness.stamp("frequency", self._freq_basis)
        text, color = self.figures.analysis_ctrl._freq_status(df, just_computed=True)
        self.figures.lbl_freq_status.configure(text=text.strip(), text_color=color)
        pc = self.figures.plot_ctrl
        pc.run_plot_chain([("PSD", lambda: pc.plot_psd(r)),
                           ("HRV tables (freq)", lambda: pc.plot_hrv_tables(r)),
                           ("Metrics", lambda: pc.plot_table_views(r))],
                          on_complete=self._freq_then)

    def _run_nonlinear(self, then: "Optional[Callable[[], None]]" = None) -> None:
        self._nonlin_then = then
        ctl, s = self.figures.analysis_ctrl, self.recording.signal
        rp = self.figures._windowed_peaks()
        lo, hi = ctl.results_rr_bounds()
        self._nonlin_basis = self.figures.freshness.capture(rr_bounds=(lo, hi))
        self.figures.lbl_nonlin_status.configure(text="Computing SampEn / DFA…")
        self.run_job("Non-linear HRV", analyse_hrv_nonlinear, rp, s.fs,
                     max_beats=ctl.entropy_cap(), rr_min_ms=lo, rr_max_ms=hi,
                     breaks=ctl.hrv_breaks(), peaks_bound=True,
                     on_done=self._nonlinear_done)

    def _nonlinear_done(self, df) -> None:
        r = self.recording.analysis.results
        if r is None:
            return
        r["hrv_nonlin"] = df
        self.figures.freshness.stamp("nonlinear", self._nonlin_basis)
        text, color = self.figures.analysis_ctrl._nonlin_status(df, just_computed=True)
        self.figures.lbl_nonlin_status.configure(text=text.strip(), text_color=color)
        pc = self.figures.plot_ctrl
        pc.run_plot_chain([("Non-linear metrics", lambda: pc.plot_nonlinear(r)),
                           ("Metrics", lambda: pc.plot_table_views(r))],
                          on_complete=self._nonlin_then)

    def refresh_tab_states(self) -> None:
        """ECGApp._refresh_tab_states: each tab's state (up to date, out of
        date, not computed) as a coloured dot on its head, why in its tooltip."""
        from ecg.ui.context_strip import tab_states
        from ecg.ui.freshness import NOT_COMPUTED, OUT_OF_DATE, UP_TO_DATE
        from ecg.ui.tab_keys import TAB_MODULES
        fresh = self.figures.freshness
        statuses = {m: fresh.status(m) for m in fresh.states()}
        rp = self.recording.detection.rpeaks_ok
        per_tab = tab_states({m: s.state for m, s in statuses.items()},
                             rp is not None and len(rp) > 0)
        p = theme.current()
        colors = {UP_TO_DATE: p.GREEN, OUT_OF_DATE: p.ORANGE, NOT_COMPUTED: p.LIGHT}
        names = [self.tabs.tabText(i).replace("&&", "&") for i in range(self.tabs.count())]
        for tab, state in per_tab.items():
            if tab not in names:
                continue
            reasons: "list[str]" = []
            for m in TAB_MODULES.get(tab, ()):
                reasons += [f"{m.replace('_', ' ')}: {why}" for why in statuses[m].reasons]
                if state == UP_TO_DATE and statuses[m].state == NOT_COMPUTED:
                    reasons.append(f"{m.replace('_', ' ')}: not computed")
            i = names.index(tab)
            key = (state, tuple(reasons))
            if self._tab_state_shown.get(tab) == key:
                continue
            self._tab_state_shown[tab] = key
            pix = QtGui.QPixmap(10, 10)
            pix.fill(QtCore.Qt.GlobalColor.transparent)
            painter = QtGui.QPainter(pix)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setBrush(QtGui.QColor(colors.get(state, p.LIGHT)))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawEllipse(1, 1, 8, 8)
            painter.end()
            self.tabs.setTabIcon(i, QtGui.QIcon(pix))
            self.tabs.setTabToolTip(i, f"{tab}: {state}" + ("\n" + "\n".join(reasons)
                                                            if reasons else ""))

    def refresh_banners(self) -> None:
        try:
            self.refresh_tab_states()
        except Exception as exc:
            log.debug("tab states refresh failed: %s", exc, exc_info=True)
        w = self.tabs.currentWidget()
        refresh = getattr(w, "refresh_banner", None)
        if callable(refresh):
            try:
                refresh()
            except Exception as exc:       # a banner never breaks the window
                log.debug("banner refresh failed: %s", exc, exc_info=True)

    def recompute_freq(self) -> None:
        """Rhythm & HRV > Recompute Frequency: on the current peaks, window
        and RR bounds (the core results kept)."""
        if not self._can_recompute():
            return
        self._run_freq(then=lambda: self.set_status("Frequency HRV recomputed"))

    def recompute_nonlinear(self) -> None:
        """Rhythm & HRV > Recompute Non-linear (e.g. after an entropy cap change)."""
        if not self._can_recompute():
            return
        self._run_nonlinear(then=lambda: self.set_status("Non-linear HRV recomputed"))

    def _can_recompute(self) -> bool:
        if self.recording.analysis.results is None:
            self.set_status("Click Analyze first.", warn=True)
            return False
        rp = self.figures._windowed_peaks()
        if rp is None or len(rp) < 5:
            self.set_status("Not enough peaks in the analysis window.", warn=True)
            return False
        return True

    def _analysis_chain_done(self) -> None:
        """The end of Analyze; the epochs last when asked (Settings: "Compute
        epochs after Analyze", AnalysisController.on_analysis_done)."""
        self.set_status("Analysis done")
        if getattr(self, "_auto_epochs", False):
            self._auto_epochs = False
            self.figures.analysis_ctrl.compute_epochs()

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
                on_end: "Optional[Callable[[], None]]" = None, peaks_bound: bool = False,
                **kwargs: Any) -> Job:
        """Run ``fn(*args, progress_cb=..., **kwargs)`` in the background.
        *on_done(result)* runs on the main thread if the recording is still
        the one the job started on; *on_end()* always runs once it ends.

        Errors: a failing job is logged with its traceback and shown; an
        error while showing its result (in *on_done*) too -- the window and
        the state the job did not touch stay as they were. Several jobs may
        run at once: the progress row follows the newest, Cancel stops all.

        *peaks_bound*: the job computes from the current peaks; its result is
        dropped if they changed while it ran (a peak edit, the threshold, a
        new detection), instead of being shown for peaks no longer there."""
        gen = self.generation
        version = self.peaks_version
        job = Job(fn, *args, **kwargs)
        self._active.append((job, what))
        self._show_progress()
        log.info("job started: %s", what)

        def _guard(label: str, fn_: Callable[..., Any], *a: Any) -> None:
            try:
                fn_(*a)
            except Exception:
                import traceback
                exc = sys.exc_info()[1]
                self.errors.report(f"{what}: {label}", str(exc) or type(exc).__name__,
                                   traceback.format_exc())

        def _end() -> None:
            self._active = [(j, w) for j, w in self._active if j is not job]
            self._show_progress()
            if on_end is not None:
                _guard("finishing failed", on_end)

        def _finished(result: Any) -> None:
            log.info("job done: %s", what)
            _end()
            if gen != self.generation:
                self.set_status(f"{what}: result discarded — another recording was opened",
                                warn=True)
                return
            if peaks_bound and version != self.peaks_version:
                log.info("job result discarded (peaks changed): %s", what)
                self.set_status(f"{what}: result discarded — the peaks changed while it ran; "
                                "run it again", warn=True)
                return
            _guard("showing the result failed", on_done, result)

        def _failed(msg: str, tb: str) -> None:
            _end()
            self.set_status(f"{what} failed: {msg}", warn=True)
            if gen == self.generation:
                self.errors.report(f"{what} failed", msg, tb)
            else:
                log.error("%s failed (recording since closed): %s\n%s", what, msg, tb)

        def _cancelled() -> None:
            log.info("job cancelled: %s", what)
            _end()
            self.set_status(f"{what} cancelled", warn=True)

        job.signals.progress.connect(lambda pct, msg: self._progress(job, pct, msg))
        job.signals.finished.connect(_finished)
        job.signals.failed.connect(_failed)
        job.signals.cancelled.connect(_cancelled)
        return self.jobs.start(job)

    def _show_progress(self) -> None:
        """The progress row: the newest running job (and how many others)."""
        busy = bool(self._active)
        self.progress.setVisible(busy)
        self.btn_cancel.setVisible(busy)
        if not busy:
            self.lbl_progress.setText("")
            return
        job, what = self._active[-1]
        others = len(self._active) - 1
        self.progress.setValue(0)
        self.lbl_progress.setText(what + "…" + (f" (+{others} running)" if others else ""))

    def _progress(self, job: Job, pct: int, msg: str) -> None:
        if self._active and self._active[-1][0] is job:
            self.set_progress(pct, msg)

    def set_progress(self, pct: int, msg: str) -> None:
        """The progress row's value and text (the newest job's; the Tk
        controllers' _set_progress)."""
        if not self._active:
            return
        self.progress.setValue(int(pct))
        others = len(self._active) - 1
        self.lbl_progress.setText(msg + (f" (+{others} running)" if others else ""))

    def cancel_jobs(self) -> None:
        self.jobs.cancel_all()

    # ── status, theme ───────────────────────────────────────
    def set_status(self, text: str, warn: bool = False) -> None:
        p = theme.current()
        self.statusBar().setStyleSheet(f"color: {p.WARN_TEXT if warn else p.MUTED};")
        self.statusBar().showMessage(text)

    def _shortcut_detect(self) -> None:
        if self.detection.btn_detect.isEnabled() and not self._typing():
            self.detection.detect()

    def _shortcut_analyze(self) -> None:
        if self.btn_analyze.isEnabled():
            self.analyze()

    def _shortcut_pan(self, direction: int) -> None:
        """← / →: move the Signal & Detection view by 70 % of its width."""
        if self.tabs.currentWidget() is not self.detection or self._typing():
            return
        t0, t1 = self.detection.plot.window()
        self.detection.go_to(t0 + direction * 0.7 * (t1 - t0))

    def _typing(self) -> bool:
        """A text box has the focus: Space and the arrows are typed there."""
        w = QtWidgets.QApplication.focusWidget()
        return isinstance(w, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox,
                              QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit,
                              QtWidgets.QComboBox, QtWidgets.QAbstractItemView))

    def toggle_theme(self) -> None:
        app = QtWidgets.QApplication.instance()
        p = theme.apply(app, theme.LIGHT.name if theme.current().is_dark else theme.DARK.name)
        self.btn_theme.setText("Light mode" if p.is_dark else "Dark mode")
        sync_figure_theme(p.is_dark)
        self.detection.apply_palette(p)
        self.redraw_results()
        self._tab_state_shown.clear()          # the dots in the new colours
        theme.save_prefs()

    def choose_text_size(self) -> None:
        """View > Text size: the interface's text, kept for the next runs."""
        lo, hi = theme.FONT_SCALE_RANGE
        scale, ok = QtWidgets.QInputDialog.getDouble(
            self, "Text size", "Text size (× the default):", theme.font_scale(), lo, hi, 2)
        if ok:
            self.set_text_size(scale)

    def set_text_size(self, scale: float) -> None:
        theme.apply(QtWidgets.QApplication.instance(), theme.current().name, scale)
        theme.save_prefs()

    def show_shortcuts(self) -> None:
        QtWidgets.QMessageBox.information(self, "Keyboard shortcuts", SHORTCUTS_TEXT)

    def clear_session_cache(self) -> None:
        """⋯ > Clear session cache (SessionController.delete_session)."""
        from ecg.io.session import delete_session
        path = self.recording.signal.filepath
        if not path:
            return
        if QtWidgets.QMessageBox.question(
                self, "Clear session cache",
                f"Delete the saved session of {Path(path).name}? The recording itself is not "
                "touched.") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        if delete_session(path):
            self.session.saved_at = None
            self.set_status("Session cache deleted.")
        else:
            self.set_status("No session cache to delete.")

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        """Unsaved changes are offered for saving; the jobs running are
        cancelled (they stop at their next checkpoint) and waited for."""
        if not self.confirm_discard("Quit"):
            ev.ignore()
            return
        self.cancel_jobs()
        if not self.jobs.wait(5000):
            log.warning("jobs still running at exit")
        super().closeEvent(ev)
