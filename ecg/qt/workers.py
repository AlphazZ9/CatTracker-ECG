# -*- coding: utf-8 -*-
"""
ecg.qt.workers
--------------
Background jobs of the Qt interface: a computation runs in a QThreadPool
thread, reports its progress, can be cancelled, and hands its result (or
its error) back to the main thread through Qt signals.

The contract, which fixes what the audit found in the Tk interface:

* a job receives its inputs when it is created (a snapshot); it never reads
  the interface's state while it runs, and never writes it;
* only the main thread writes the result into the model (the receiver of
  ``finished``);
* cancellation needs no change to the algorithms: the ecg.core / ecg.services
  functions already call a ``progress_cb(percent, message)`` at their
  checkpoints, and the job's callback raises ``Cancelled`` there once
  ``cancel()`` was called. The computation stops at its next checkpoint;
  nothing it computed is used.

Several jobs can run at once; the caller decides whether a result is still
wanted when it lands (e.g. the recording changed meanwhile).
"""
from __future__ import annotations

import threading
import traceback
from typing import Any, Callable

from PySide6 import QtCore


class Cancelled(Exception):
    """Raised inside a job's computation, at a progress checkpoint, once the
    job was cancelled."""


class JobSignals(QtCore.QObject):
    """Signals of a Job, delivered to receivers in the main thread."""
    progress = QtCore.Signal(int, str)          # percent, message
    finished = QtCore.Signal(object)            # the result
    failed = QtCore.Signal(str, str)            # message, traceback
    cancelled = QtCore.Signal()


class Job(QtCore.QRunnable):
    """``fn(*args, progress_cb=..., **kwargs)`` in a pool thread.

    *fn* must accept a ``progress_cb`` keyword (pass ``progress=False`` for
    a function that has none: it is then called without it and can only be
    cancelled before it starts)."""

    def __init__(self, fn: Callable[..., Any], *args: Any, progress: bool = True,
                 **kwargs: Any) -> None:
        super().__init__()
        self.setAutoDelete(False)              # the JobRunner keeps it until it ends
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self._uses_progress = progress
        self._cancel = threading.Event()
        self.signals = JobSignals()
        self.done = threading.Event()

    def cancel(self) -> None:
        """Stop at the next progress checkpoint (thread-safe)."""
        self._cancel.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def _progress(self, pct: int, msg: str) -> None:
        if self._cancel.is_set():
            raise Cancelled()
        self.signals.progress.emit(int(pct), str(msg))

    def run(self) -> None:                     # pool thread
        try:
            if self._cancel.is_set():
                raise Cancelled()
            if self._uses_progress:
                result = self.fn(*self.args, progress_cb=self._progress, **self.kwargs)
            else:
                result = self.fn(*self.args, **self.kwargs)
            if self._cancel.is_set():
                raise Cancelled()
        except Cancelled:
            self.signals.cancelled.emit()
        except Exception as exc:                # reported, never swallowed
            self.signals.failed.emit(str(exc) or type(exc).__name__, traceback.format_exc())
        else:
            self.signals.finished.emit(result)
        finally:
            self.done.set()


class JobRunner(QtCore.QObject):
    """Starts jobs on a thread pool and keeps them alive until they end."""

    def __init__(self, parent: "QtCore.QObject | None" = None,
                 max_threads: "int | None" = None) -> None:
        super().__init__(parent)
        self.pool = QtCore.QThreadPool(self)
        if max_threads:
            self.pool.setMaxThreadCount(max_threads)
        self._jobs: "set[Job]" = set()

    def start(self, job: Job) -> Job:
        self._jobs.add(job)
        for sig in (job.signals.finished, job.signals.failed, job.signals.cancelled):
            sig.connect(lambda *_a, j=job: self._jobs.discard(j))
        self.pool.start(job)
        return job

    @property
    def running(self) -> "list[Job]":
        return [j for j in self._jobs if not j.done.is_set()]

    def cancel_all(self) -> None:
        for j in list(self._jobs):
            j.cancel()

    def wait(self, msecs: int = -1) -> bool:
        """Wait for every job (tests, shutdown)."""
        return self.pool.waitForDone(msecs)
