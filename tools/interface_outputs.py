# -*- coding: utf-8 -*-
"""
tools/interface_outputs.py -- run one scenario in one interface and write
what it exports, so the two interfaces can be compared file by file
(tools/compare_interfaces.py, phase 4 of the migration).

    python tools/interface_outputs.py --ui tk|qt --out DIR recording.mat

The scenario, the same in both, through each interface's own code paths:
open the recording, Method = Auto, Detect Peaks, add one noisy region and
one annotation, Analyze (core, then frequency and non-linear HRV),
Measure intervals, Epochs, Sliding, Autonomic, Classify events. Then the
outputs: the Excel workbook (ExportController.write_excel), the RR CSV
(export_rr_csv), the report text (report_text), the Report tab's
provenance rows (report_view), and the session state that Save Session
would write (collect_session_state).

Run each interface in its own process: both patch the controller modules
they share (message boxes, file dialogs).

The Tk interface's background threads are run synchronously at start()
(on the main thread): some Python builds ship a Tcl that tkinter treats as
non-threaded (Tcl 9 without tcl_platform(threaded)), where a worker's
after(0) never reaches the main loop. The worker code is the same; only
when it runs changes.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NOISE = {"t_start": 30.0, "t_end": 34.0, "note": "validation"}
ANNOTATION = {"t_start": 60.0, "t_end": 80.0, "label": "drug", "color": "#E53935"}
TIMEOUT_S = 600


def _session_json(state: dict) -> str:
    from ecg.io.session import _SessionEncoder
    return json.dumps(state, cls=_SessionEncoder, indent=1, sort_keys=True)


def _write(out: Path, export_ctrl, state: dict) -> None:
    import io
    out.mkdir(parents=True, exist_ok=True)
    b = io.BytesIO()
    print("export: workbook", flush=True)
    export_ctrl.write_excel(b)
    (out / "workbook.xlsx").write_bytes(b.getvalue())
    print("export: report", flush=True)
    (out / "report.txt").write_text(export_ctrl.report_text(), encoding="utf-8")
    print("export: provenance", flush=True)
    view = export_ctrl.report_view()
    (out / "provenance.json").write_text(json.dumps(
        [list(map(str, r)) if isinstance(r, (list, tuple)) else str(r)
         for r in view.provenance], indent=1), encoding="utf-8")
    print("export: session", flush=True)
    (out / "session.json").write_text(_session_json(state), encoding="utf-8")


def _rr_csv(export_ctrl, path: Path) -> None:
    import ecg.ui.export_controller as ec
    saved = ec.filedialog

    class _Dialog:
        @staticmethod
        def asksaveasfilename(**_k):
            return str(path)
    ec.filedialog = _Dialog
    print("export: rr.csv", flush=True)
    try:
        export_ctrl.export_rr_csv()
    finally:
        ec.filedialog = saved


# ── Tk ─────────────────────────────────────────────────────────────────
class _SyncThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=None, **_k):
        self._t, self._a, self._k = target, args, kwargs or {}

    def start(self):
        self._t(*self._a, **self._k)

    def join(self, *_a):
        pass

    def is_alive(self):
        return False


class _ThreadingSync:
    """The threading module as the Tk modules see it, Thread synchronous."""
    Thread = _SyncThread

    def __getattr__(self, name):
        return getattr(threading, name)


def _load_state(path: "str | None") -> "dict | None":
    if not path:
        return None
    from ecg.io.session import _session_decode_hook
    return json.loads(Path(path).read_text(encoding="utf-8"), object_hook=_session_decode_hook)


def run_tk(recording: str, out: Path, restore: "str | None" = None) -> None:
    from ecg.ui.app import ECGApp
    import ecg.ui.app_async as app_async
    import ecg.ui.tab_detection as tab_detection
    for mod in (app_async, tab_detection):            # see the module docstring
        mod.threading = _ThreadingSync()
    app = ECGApp()
    steps = []
    done_steps: list = []
    failed: list = []

    def idle() -> bool:
        return not app._async_busy and not app.ui.running_modules

    def chain(*items):
        steps.extend(items)

    def finish(code: int) -> None:
        """End the process once the outputs are written: on the Windows CI
        runner the Tk main loop did not return after quit()."""
        import os
        import traceback
        if failed:
            traceback.print_exception(failed[0])
        print(f"tk: {'outputs written' if code == 0 else 'failed'}", flush=True)
        sys.stderr.flush()
        os._exit(code)

    def run_next(t0=None):
        if not steps:
            finish(0)
            return
        cond, action = steps[0]
        if cond is None or cond():
            steps.pop(0)
            print(f"tk step {len(done_steps) + 1}", flush=True)
            done_steps.append(1)
            try:
                action()
            except Exception as exc:          # stop, reported by finish()
                failed.append(exc)
                finish(1)
                return
            app.after(10, run_next)
            return
        if time.time() - (t0 or time.time()) > TIMEOUT_S:
            failed.append(TimeoutError(getattr(action, "__name__", "step")))
            finish(1)
            return
        app.after(50, lambda: run_next(t0 or time.time()))

    r = lambda: app.analysis.results or {}

    def detect():
        app.cb_det_method.set("Auto (NeuroKit2)")
        app._on_det_method_change("Auto (NeuroKit2)")
        app.signal_ctrl.preview()

    def regions():
        app.analysis.noise_regions.append(dict(NOISE))
        app._noise_regions_changed()
        app.analysis.annotations.append(dict(ANNOTATION))
        app._update_ann_count()

    def export():
        _write(out, app.export_ctrl, app.session_ctrl.collect_session_state())
        _rr_csv(app.export_ctrl, out / "rr.csv")

    state = _load_state(restore)
    if state is not None:
        def restore_session():
            sc = app.session_ctrl
            sc.on_restore_session_done(sc.restore_session_worker(state, recording),
                                       state.get("saved_at", ""))
        chain((None, lambda: app._load_path(recording)),
              (lambda: idle() and app.signal.time is not None, restore_session),
              (lambda: idle() and app.detection.rpeaks_ok is not None, export))
        app.after(200, run_next)
        app.mainloop()
        if failed:
            raise failed[0]
        return
    chain((None, lambda: app._load_path(recording)),
          (lambda: idle() and app.signal.time is not None, detect),
          (lambda: idle() and app.detection.rpeaks_ok is not None, regions),
          (idle, app._run_analysis),
          (lambda: idle() and r().get("hrv_nonlin") is not None
           and not r()["hrv_nonlin"].empty, app.analysis_ctrl.run_intervals),
          (lambda: idle() and r().get("intervals") is not None
           and not r()["intervals"].empty, app.analysis_ctrl.compute_epochs),
          (lambda: idle() and app.analysis.epoch_df is not None,
           app.analysis_ctrl.compute_rolling_hrv),
          (lambda: idle() and app.analysis.rolling_hrv_df is not None,
           lambda: app.analysis_ctrl.compute_autonomic()),
          (lambda: idle() and app.analysis.autonomic_result is not None,
           app.analysis_ctrl.run_arrhythmia_analysis),
          (lambda: idle() and getattr(app.analysis, "arrhythmia_run", None) is not None, export))
    app.after(200, run_next)
    app.mainloop()
    print("tk: main loop ended", flush=True)
    if failed:
        raise failed[0]


# ── Qt ─────────────────────────────────────────────────────────────────
def run_qt(recording: str, out: Path, restore: "str | None" = None) -> None:
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    from ecg.qt.main_window import MainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow()
    w.confirm_discard = lambda action: True           # no prompt on close
    w.resize(1400, 900)
    w.show()

    def wait(cond, what):
        print(f"qt waiting: {what}", flush=True)
        end = time.time() + TIMEOUT_S
        while not (cond() and not w._active):
            if time.time() > end:
                raise TimeoutError(what)
            app.processEvents()
            time.sleep(0.01)
        if w.errors.shown:
            raise RuntimeError(f"{what}: {w.errors.shown}")

    r = lambda: w.recording.analysis.results or {}
    w.open_path(recording, check_session=False)
    state = _load_state(restore)
    if state is not None:
        w.session.saved_at = None
        w.session.restore(state, recording)
        wait(lambda: w.session.saved_at is not None, "Restore")
        for _ in range(50):                         # the restored figures' chains
            app.processEvents()
            time.sleep(0.01)
        _write(out, w.figures.export_ctrl, w.session.ctrl.collect_session_state())
        _rr_csv(w.figures.export_ctrl, out / "rr.csv")
        return
    w.detection.cb_method.setCurrentIndex(0)          # Auto (NeuroKit2)
    w.detection.detect()
    wait(lambda: w.recording.detection.rpeaks_ok is not None, "Detect Peaks")
    a = w.recording.analysis
    a.noise_regions.append(dict(NOISE))
    w.detection.noise_changed()
    a.annotations.append(dict(ANNOTATION))
    w.detection.annotations_changed()
    w.analyze()
    wait(lambda: w.statusBar().currentMessage() == "Analysis done", "Analyze")
    w.morphology.measure()
    wait(lambda: r().get("intervals") is not None and not r()["intervals"].empty, "Intervals")
    ctl = w.figures.analysis_ctrl
    ctl.compute_epochs()
    wait(lambda: a.epoch_df is not None, "Epochs")
    ctl.compute_rolling_hrv()
    wait(lambda: a.rolling_hrv_df is not None, "Sliding")
    ctl.compute_autonomic()
    wait(lambda: a.autonomic_result is not None, "Autonomic")
    ctl.run_arrhythmia_analysis()
    wait(lambda: getattr(a, "arrhythmia_run", None) is not None, "Events")
    _write(out, w.figures.export_ctrl, w.session.ctrl.collect_session_state())
    _rr_csv(w.figures.export_ctrl, out / "rr.csv")
    w.jobs.cancel_all()
    w.jobs.wait(5000)


def main(argv=None) -> int:
    # A run that stalls prints every thread's stack (then the parent's time
    # limit stops it): where it blocks, not only that it does.
    import faulthandler
    faulthandler.dump_traceback_later(TIMEOUT_S + 300, repeat=False, exit=False)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ui", choices=("tk", "qt"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--restore", help="a session.json to restore instead of the scenario")
    ap.add_argument("recording")
    a = ap.parse_args(argv)
    (run_tk if a.ui == "tk" else run_qt)(str(Path(a.recording).resolve()), Path(a.out),
                                         a.restore)
    print(f"{a.ui}: outputs in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
