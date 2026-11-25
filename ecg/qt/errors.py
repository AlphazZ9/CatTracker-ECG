# -*- coding: utf-8 -*-
"""
ecg.qt.errors
-------------
What happens when something goes wrong in the Qt interface.

* Every error is written to the log file (setup_logging: a rotating file
  in the session folder, ``logs/ecg_qt.log``), with its traceback.
* An exception nobody caught -- in the main thread (a slot, a timer) or in
  a worker thread -- never closes the window: install_excepthook routes it
  to ErrorReporter, which logs it and shows it in a non-blocking dialog
  (the details hold the traceback). While one such dialog is open, the
  next errors are only logged, so an error repeated on every repaint
  cannot bury the window under dialogs.
* A background job that fails is reported the same way (MainWindow.run_job).
"""
from __future__ import annotations

import logging
import sys
import threading
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

from PySide6 import QtCore, QtWidgets

log = logging.getLogger("ecg")

LOG_NAME = "ecg_qt.log"
LOG_MAX_BYTES = 1_000_000
LOG_BACKUPS = 3


def default_log_dir() -> Path:
    from ecg.core.models import SESSION_DIR
    return Path(SESSION_DIR) / "logs"


def setup_logging(log_dir: "Optional[Path]" = None, level: int = logging.INFO) -> Path:
    """The "ecg" logger also writes to *log_dir*/ecg_qt.log (rotated at
    1 MB, 3 kept). Idempotent. Returns the log file's path."""
    path = Path(log_dir or default_log_dir()) / LOG_NAME
    for h in log.handlers:
        if isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == path.resolve():
            return path
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS,
                                  encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s"))
    log.addHandler(handler)
    if log.level == logging.NOTSET or log.level > level:
        log.setLevel(level)
    return path


class ErrorReporter(QtCore.QObject):
    """Logs an error and shows it, from any thread (the dialog always opens
    on the main thread)."""

    _report = QtCore.Signal(str, str, str)          # title, message, details

    def __init__(self, parent: "Optional[QtWidgets.QWidget]" = None) -> None:
        super().__init__(parent)
        self._parent = parent
        self._box: "Optional[QtWidgets.QMessageBox]" = None
        self.n_suppressed = 0
        self.shown: "list[tuple[str, str]]" = []     # (title, message), for tests
        self._report.connect(self._show, QtCore.Qt.ConnectionType.QueuedConnection)

    def report(self, title: str, message: str, details: str = "") -> None:
        log.error("%s: %s%s", title, message, f"\n{details}" if details else "")
        self._report.emit(title, message, details)

    def report_exception(self, title: str, exc_type: Any, exc: BaseException,
                         tb: Any) -> None:
        details = "".join(traceback.format_exception(exc_type, exc, tb))
        self.report(title, str(exc) or exc_type.__name__, details)

    def _show(self, title: str, message: str, details: str) -> None:
        if self._box is not None and self._box.isVisible():
            self.n_suppressed += 1
            return
        self.shown.append((title, message))
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Warning, title,
                                    f"{message}\n\nThe analysis state is unchanged; the "
                                    "details are also in the log file (Help > Open log "
                                    "folder).", parent=self._parent)
        if details:
            box.setDetailedText(details)
        box.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        box.finished.connect(lambda *_a: setattr(self, "_box", None))
        self._box = box
        box.open()


def install_excepthook(reporter: ErrorReporter) -> None:
    """Uncaught exceptions of the main thread and of worker threads go to
    *reporter* (KeyboardInterrupt keeps its default behaviour)."""
    previous = sys.excepthook

    def _hook(exc_type: Any, exc: BaseException, tb: Any) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            previous(exc_type, exc, tb)
            return
        try:
            reporter.report_exception("Unexpected error", exc_type, exc, tb)
        except Exception:                       # never raise from the hook
            previous(exc_type, exc, tb)

    def _thread_hook(args: Any) -> None:
        if args.exc_type is SystemExit:
            return
        try:
            reporter.report_exception(f"Unexpected error in {getattr(args.thread, 'name', '?')}",
                                      args.exc_type, args.exc_value, args.exc_traceback)
        except Exception:
            pass

    sys.excepthook = _hook
    threading.excepthook = _thread_hook
