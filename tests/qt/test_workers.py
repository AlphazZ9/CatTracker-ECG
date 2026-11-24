"""ecg.qt.workers: results and errors come back on the main thread, progress
is reported, and cancel() stops a computation at its next checkpoint."""
import threading
import time

from PySide6 import QtCore

from ecg.qt.workers import Job, JobRunner


def _slow(n, progress_cb, step_s=0.01):
    for i in range(n):
        progress_cb(int(100 * i / n), f"step {i}")
        time.sleep(step_s)
    return n * 2


def test_result_and_progress_arrive_on_the_main_thread(qtbot):
    runner = JobRunner()
    job = Job(_slow, 5)
    seen = {}
    job.signals.progress.connect(lambda p, m: seen.setdefault("progress", []).append(p))
    job.signals.finished.connect(lambda r: seen.update(
        result=r, main=QtCore.QThread.currentThread() is QtCore.QCoreApplication.instance().thread()))
    with qtbot.waitSignal(job.signals.finished, timeout=5000):
        runner.start(job)
    assert seen["result"] == 10 and seen["main"]
    assert seen["progress"][0] == 0 and len(seen["progress"]) == 5


def test_cancel_stops_at_the_next_checkpoint(qtbot):
    runner = JobRunner()
    reached = []
    job = Job(lambda progress_cb: [progress_cb(i, "") or reached.append(i) or time.sleep(0.02)
                                   for i in range(500)])
    job.signals.progress.connect(lambda p, m: job.cancel() if p == 3 else None)
    with qtbot.waitSignal(job.signals.cancelled, timeout=5000):
        runner.start(job)
    assert runner.wait(2000)
    assert len(reached) < 50                 # stopped long before the end


def test_an_error_is_reported_with_its_traceback(qtbot):
    runner = JobRunner()

    def boom(progress_cb):
        raise ValueError("Signal is flat — wrong channel.")
    job = Job(boom)
    with qtbot.waitSignal(job.signals.failed, timeout=5000) as blocker:
        runner.start(job)
    msg, tb = blocker.args
    assert "flat" in msg and "ValueError" in tb


def test_several_jobs_run_at_once(qtbot):
    runner = JobRunner(max_threads=4)
    gate = threading.Barrier(3, timeout=5)
    jobs = [Job(lambda progress_cb: gate.wait() or "ok") for _ in range(3)]
    with qtbot.waitSignals([j.signals.finished for j in jobs], timeout=8000):
        for j in jobs:
            runner.start(j)
