"""Phase 3: errors never close the window and are logged; jobs share the
progress row; unsaved changes are offered for saving."""
import logging
import sys
import threading

import pytest
from PySide6 import QtWidgets

from ecg.qt import errors
from ecg.qt.main_window import MainWindow


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w.jobs.cancel_all()
    w.jobs.wait(5000)


def _boom(*_a, **_k):
    raise ValueError("bad input")


def test_log_file(tmp_path):
    path = errors.setup_logging(tmp_path)
    assert errors.setup_logging(tmp_path) == path            # idempotent
    logging.getLogger("ecg").error("written to the file")
    for h in logging.getLogger("ecg").handlers:
        h.flush()
    assert "written to the file" in path.read_text(encoding="utf-8")


def test_failed_job_is_reported(qtbot, window):
    window.run_job("Test job", _boom, progress=False, on_done=lambda r: None)
    qtbot.waitUntil(lambda: bool(window.errors.shown), timeout=5000)
    assert window.errors.shown[0] == ("Test job failed", "bad input")
    assert "failed" in window.statusBar().currentMessage()
    assert window.progress.isHidden() and not window._active


def test_error_in_on_done_is_reported(qtbot, window):
    window.run_job("Test job", lambda: 1, progress=False, on_done=_boom)
    qtbot.waitUntil(lambda: bool(window.errors.shown), timeout=5000)
    assert window.errors.shown[0] == ("Test job: showing the result failed", "bad input")


def test_one_dialog_at_a_time(qtbot, window):
    for _ in range(5):
        window.errors.report("Repeated", "again")
    qtbot.waitUntil(lambda: bool(window.errors.shown), timeout=5000)
    qtbot.wait(50)
    assert len(window.errors.shown) == 1 and window.errors.n_suppressed == 4


def test_uncaught_exceptions_go_to_the_reporter(qtbot, window, monkeypatch):
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    errors.install_excepthook(window.errors)
    try:
        raise RuntimeError("main thread")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    t = threading.Thread(target=lambda: _boom(), name="worker-x")
    t.start()
    t.join()
    qtbot.waitUntil(lambda: bool(window.errors.shown), timeout=5000)
    assert window.errors.shown[0] == ("Unexpected error", "main thread")
    assert window.errors.n_suppressed == 1          # the thread's, logged only


def test_concurrent_jobs_share_the_progress_row(qtbot, window):
    gate = threading.Event()
    window.run_job("First", lambda: gate.wait(5), progress=False, on_done=lambda r: None)
    window.run_job("Second", lambda: gate.wait(5), progress=False, on_done=lambda r: None)
    assert not window.progress.isHidden()
    assert window.lbl_progress.text() == "Second… (+1 running)"
    gate.set()
    qtbot.waitUntil(lambda: not window._active, timeout=5000)
    assert window.progress.isHidden()


@pytest.mark.unsaved_prompt
def test_unsaved_changes_prompt(qtbot, window, monkeypatch):
    import numpy as np
    B = QtWidgets.QMessageBox.StandardButton
    window.recording.signal.filtered = np.zeros(10)
    window.recording.signal.filepath = "x.mat"
    window.session_dirty = True
    answers = iter([B.Cancel, B.Discard, B.Save])
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: next(answers)))
    saved = []
    monkeypatch.setattr(window.session, "save_now", lambda: saved.append(1) or True)
    assert window.confirm_discard("Quit") is False
    assert window.confirm_discard("Quit") is True and not saved
    assert window.confirm_discard("Quit") is True and saved
    window.session_dirty = False                    # teardown closes without asking


def test_result_for_changed_peaks_is_dropped(qtbot, window):
    """A peak-bound job's result is not shown for peaks no longer there."""
    gate = threading.Event()
    got = []
    window.run_job("Bound", lambda: gate.wait(5) and 1, progress=False, peaks_bound=True,
                   on_done=got.append)
    window.peaks_changed()                         # e.g. a peak edit while it runs
    gate.set()
    qtbot.waitUntil(lambda: not window._active, timeout=5000)
    assert got == [] and "peaks changed" in window.statusBar().currentMessage()
    window.run_job("Bound", lambda: 2, progress=False, peaks_bound=True, on_done=got.append)
    qtbot.waitUntil(lambda: got == [2], timeout=5000)
    window.session_dirty = False
