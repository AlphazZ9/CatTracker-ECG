# -*- coding: utf-8 -*-
"""
tools/capture_screenshots.py -- retake the User Guide screenshots.

Runs the Qt interface headless (offscreen platform) on the synthetic
recording tests/fixtures/synthetic_mouse_ecg.mat (150 s) with two
"Stimulus" annotations (50 s and 100 s): Detect Peaks, Analyze, Measure
intervals, Classify events, Epochs, Sliding, Autonomic and Event-locked,
then grabs each view into docs/screenshots/ at a device pixel ratio of 2.

    python tools/capture_screenshots.py [--out DIR] [--only NAME ...]

Nothing is written to the user's profile: sessions, the registry, the ML
cache and the preferences go to a temporary folder.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_SCALE_FACTOR", "2")      # grabs at a device pixel ratio of 2

FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"
WIDTH, HEIGHT = 1440, 900                           # main window, logical pixels
STIMULI = [{"t_start": t, "t_end": t + 10.0, "label": "Stimulus", "color": "#E53935"}
           for t in (50.0, 100.0)]


def _isolate_profile(d: Path) -> None:
    import ecg.core.ml_detector as ml
    import ecg.io.db as db
    import ecg.io.session as io_session
    import ecg.qt.theme as qt_theme
    io_session.SESSION_DIR = d / "sessions"
    db.SESSION_DIR = d / "sessions"
    db._DB_PATH = d / "sessions" / "ecg_registry.db"
    db._schema_ready = False
    ml.TRAINING_DATA_DIR = d / "ml_training"
    ml.MODEL_PATH = d / "model.skops"
    qt_theme.PREFS_PATH = d / "ecg_qt_prefs.json"
    qt_theme.LEGACY_THEME_PATH = d / "ecg_theme.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Retake the User Guide screenshots.")
    ap.add_argument("--out", default=str(ROOT / "docs" / "screenshots"))
    ap.add_argument("--only", nargs="*", default=None, help="file names, without .png")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _isolate_profile(Path(tempfile.mkdtemp(prefix="ecg_shots_")))
    from PySide6 import QtCore, QtWidgets
    from ecg.qt import theme
    from ecg.qt.main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(app, theme.LIGHT.name, 1.0)
    MainWindow.confirm_discard = lambda self, action: True      # no modal box

    def pump(seconds: float = 0.3) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
            time.sleep(0.01)

    def wait(cond, what: str, timeout: float = 300.0) -> None:
        end = time.monotonic() + timeout
        while not cond():
            if time.monotonic() > end:
                raise TimeoutError(what)
            pump(0.1)
        pump(0.5)

    def save(widget: QtWidgets.QWidget, name: str) -> None:
        if args.only is not None and name not in args.only:
            return
        w.refresh_banners()
        pump(1.0)
        pix = widget.grab()
        path = out / f"{name}.png"
        pix.save(str(path))
        print(f"{name}.png  {pix.width()}x{pix.height()} px, dpr {pix.devicePixelRatio():g}")

    def scroll_to(area: QtWidgets.QScrollArea, title: str, margin: int = 8) -> None:
        """The card whose bold title starts with *title* at the top of *area*."""
        for lab in area.widget().findChildren(QtWidgets.QLabel):
            if lab.text().replace("<b>", "").startswith(title):
                card = lab.parentWidget()
                y = card.mapTo(area.widget(), QtCore.QPoint(0, 0)).y()
                area.verticalScrollBar().setValue(max(0, y - margin))
                return
        raise LookupError(title)

    def status() -> str:
        return w.statusBar().currentMessage()

    w = MainWindow()
    w.resize(WIDTH, HEIGHT)
    w.show()
    pump(1.0)
    w.open_path(str(FIXTURE), check_session=False)
    pump(0.5)
    rec = w.recording
    a = rec.analysis
    a.annotations[:] = [dict(s) for s in STIMULI]
    w.detection.annotations_changed()

    # ── Detect Peaks, Analyze ──
    w.detection.cb_method.setCurrentIndex(0)                    # Auto (NeuroKit2)
    w.detection.detect()
    wait(lambda: rec.detection.rpeaks_ok is not None, "Detect Peaks")
    w.analyze()
    wait(lambda: status() == "Analysis done", "Analyze")
    wait(lambda: not w._active, "Frequency / Non-linear")
    mo = w.morphology
    mo.measure()
    wait(lambda: status().startswith("Interval delineation done"), "Measure intervals")
    ev = w.events
    ev.btn_classify.click()
    wait(lambda: a.premature_summary is not None, "Classify events")
    wait(lambda: not w._active, "Classify events")

    # ── Signal & Detection ──
    det = w.detection

    def view(t0: float, t1: float, strip: bool) -> None:
        det.btn_strip.setChecked(strip)
        det.plot.set_window(t0, t1)
        det.sp_goto.setValue(t0)

    w.tabs.setCurrentWidget(det)
    view(44.0, 56.0, True)
    save(w, "01_detection_overview")
    view(60.0, 62.0, True)
    save(w, "02_detection_zoom_normal")

    # ── 03: zoom on the first premature beat ──
    events = a.arrhythmia_events
    print("   events:", sorted({e.kind for e in events}), len(events))
    premature = [e for e in events if e.kind == "premature_beat"]
    target = (premature or events)[0]
    t_ev = (target.t_start + target.t_end) / 2
    view(t_ev - 2.0, t_ev + 2.0, True)
    save(w, "03_detection_zoom_bradycardia")
    print(f"   03 centred on {t_ev:.2f} s ({target.kind})")

    # ── Rhythm & HRV ──
    rh = w.rhythm
    w.tabs.setCurrentWidget(rh)
    pump(0.5)
    rh.verticalScrollBar().setValue(0)
    save(w, "04_rhythm_hrv")
    scroll_to(rh, "Power spectral density")
    save(w, "05_rhythm_metrics")
    scroll_to(rh, "Heart rate")
    save(w, "06_rhythm_nonlinear")

    # ── Time course ──
    tc = w.time_course
    w.tabs.setCurrentWidget(tc)
    tc.setCurrentIndex(0)
    tc.sp_epoch.setValue(15.0)
    a.epoch_df = None
    tc.btn_epochs.click()
    wait(lambda: a.epoch_df is not None, "Epochs")
    save(w, "08_time_epochs")
    tc.setCurrentIndex(1)
    tc.btn_sliding.click()
    wait(lambda: a.rolling_hrv_df is not None, "Sliding")
    save(w, "09_time_sliding")
    tc.setCurrentIndex(2)
    tc.chk_lfhf.setChecked(True)
    tc.btn_auto.click()
    wait(lambda: a.autonomic_result is not None, "Autonomic")
    wait(lambda: not w._active, "Autonomic drawing")
    save(w, "14_time_autonomic")
    tc.setCurrentIndex(3)
    tc.btn_evt.click()
    wait(lambda: a.event_locked is not None, "Event-locked")
    save(w, "15_time_event_locked")

    # ── Morphology & Intervals ──
    w.tabs.setCurrentWidget(mo)
    mo.refresh_banner()
    mo.verticalScrollBar().setValue(0)
    save(w, "10_morphology_intervals")

    # ── Events ──
    w.tabs.setCurrentWidget(ev)
    if ev.table.rowCount():
        ev.table.selectRow(0)
    save(w, "11_events")

    # ── Report ──
    w.tabs.setCurrentWidget(w.report)
    w.report.refresh()
    save(w, "12_report")

    # ── Settings dialog ──
    dlg = w.settings
    dlg.show()
    dlg.adjustSize()
    save(dlg, "13_settings_dialog")
    dlg.hide()

    w.jobs.cancel_all()
    w.jobs.wait(5000)
    w.session_dirty = False
    w.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
