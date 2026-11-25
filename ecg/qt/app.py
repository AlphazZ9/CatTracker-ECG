# -*- coding: utf-8 -*-
"""
ecg.qt.app -- start the Qt interface.

    python -m ecg.qt [recording.mat]
"""
from __future__ import annotations

import multiprocessing
import sys
from typing import Optional


def main(argv: "Optional[list[str]]" = None) -> int:
    # First thing: in a frozen build each 'spawn' worker of
    # ecg.core.delineation_pool re-runs the executable; this turns that run
    # into the worker. No-op under plain Python.
    multiprocessing.freeze_support()
    argv = list(sys.argv if argv is None else argv)
    self_test = None
    if "--self-test" in argv:
        # A packaging check: start the window, draw, open no file, exit 0,
        # and write "ok" (or the error) to the file named after the flag.
        i = argv.index("--self-test")
        self_test = argv[i + 1] if i + 1 < len(argv) else "ecg_qt_self_test.txt"
        del argv[i:i + 2]

    import pyqtgraph as pg
    from PySide6 import QtWidgets

    from ecg.qt import theme
    from ecg.qt.main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setApplicationName("ECG Analysis")
    pg.setConfigOptions(antialias=False, useOpenGL=False)
    prefs = theme.load_prefs()
    theme.apply(app, prefs["theme"], prefs["font_scale"])
    from ecg.qt.figure_host import sync_figure_theme
    sync_figure_theme(theme.current().is_dark)
    from ecg.qt.errors import install_excepthook, setup_logging
    try:
        log_path = setup_logging()
    except OSError:                      # no writable session folder: console only
        log_path = None
    win = MainWindow()
    install_excepthook(win.errors)
    import logging
    logging.getLogger("ecg").info("Qt interface started (log: %s)", log_path)
    win.btn_theme.setText("Light mode" if theme.current().is_dark else "Dark mode")
    win.show()
    if self_test is not None:
        return _self_test(app, win, self_test, argv[1] if len(argv) > 1 else None)
    if len(argv) > 1:
        win.open_path(argv[1])
    return app.exec()


def _self_test(app, win, out: str, recording: "Optional[str]" = None) -> int:
    """Draw every tab and a figure, then quit; the result goes to *out*.
    With *recording*: also Detect Peaks, Analyze and Measure intervals on it
    (the intervals run in the worker processes of a frozen build)."""
    from pathlib import Path
    from PySide6 import QtCore
    result = {"text": "not run"}

    def check() -> None:
        try:
            import numpy as np
            import ecg
            for i in range(win.tabs.count()):
                win.tabs.setCurrentIndex(i)
                app.processEvents()
            win.figures.slot("rr").update(lambda fig: fig.add_subplot(111).plot(np.arange(10)))
            win.detection.plot.set_signal(np.sin(np.arange(4000) / 50.0), 2000.0)
            app.processEvents()
            extra = _self_test_recording(app, win, recording) if recording else ""
            result["text"] = (f"ok {ecg.__version__} Qt {QtCore.qVersion()} "
                              f"platform {app.platformName()} tabs {win.tabs.count()}{extra}")
        except Exception as exc:                     # reported, not raised
            import traceback
            result["text"] = "error " + "".join(traceback.format_exception(exc))
        finally:
            win.session_dirty = False
            win.close()
            app.quit()
    QtCore.QTimer.singleShot(500, check)
    app.exec()
    Path(out).write_text(result["text"] + "\n", encoding="utf-8")
    return 0 if result["text"].startswith("ok") else 1


if __name__ == "__main__":
    sys.exit(main())


def _self_test_recording(app, win, path: str) -> str:
    import time

    def wait(cond, what: str, timeout: float = 600.0) -> None:
        end = time.time() + timeout
        while not (cond() and not win._active):
            if time.time() > end:
                raise TimeoutError(what)
            if win.errors.shown:
                raise RuntimeError(f"{what}: {win.errors.shown[0]}")
            app.processEvents()
            time.sleep(0.01)

    win.open_path(path, check_session=False)
    win.detection.detect()
    wait(lambda: win.recording.detection.rpeaks_ok is not None, "Detect Peaks")
    win.analyze()
    wait(lambda: win.statusBar().currentMessage() == "Analysis done", "Analyze")
    win.morphology.measure()
    r = lambda: win.recording.analysis.results or {}
    wait(lambda: r().get("intervals") is not None and not r()["intervals"].empty, "Intervals")
    return (f" peaks {len(win.recording.detection.rpeaks_ok)}"
            f" intervals {len(r()['intervals'])}")
