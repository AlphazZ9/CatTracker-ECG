"""
Background jobs landing on a different recording (audit 2026-11-13, bugs
1, 8, 9, 17, 18), driven through stub apps (no Tk window):

* opening a file while a job runs is refused before anything is reset (1);
* a load / Detect Peaks / Analyze / session check / restore whose result
  lands after another file was opened is dropped (1, 8, 17);
* an Analyze that lands after a peak edit does not bring the edited peaks
  back nor stamp core as current (9);
* Save Session writes the registry row and the training sample of the file
  it was started on, even when another one is opened mid-save (18).
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ecg.core import analysis
from ecg.ui import session_controller as sc_module
from ecg.services import session_service  # where restore reloads the signal
from ecg.ui.app_async import recording_changed
from ecg.ui.freshness import UP_TO_DATE
from ecg.ui.session_controller import SessionController
from ecg.ui.signal_controller import SignalController
from ecg.ui.state import AnalysisState, DetectionState, SessionState, SignalState
from ecg.ui.theme import ORANGE


class _Recorder:
    """Records _set_status calls."""
    def __init__(self) -> None:
        self.calls: "list[tuple[str, object]]" = []

    def __call__(self, text, color=None) -> None:
        self.calls.append((text, color))

    def last(self) -> str:
        return self.calls[-1][0] if self.calls else ""


def _new_file(app, path="C:/data/B.mat") -> None:
    """What load_path / reset_for_new_file does to the state a job reads."""
    app._generation += 1
    app.signal.filepath = path


# ── the shared guard ─────────────────────────────────────────────────────

def test_recording_changed_reports_only_a_new_generation():
    status = _Recorder()
    app = SimpleNamespace(_generation=3, _set_status=status)
    assert recording_changed(app, 3, "Job") is False
    assert status.calls == []
    assert recording_changed(app, 2, "Job") is True
    assert "discarded" in status.last() and status.calls[-1][1] == ORANGE


# ── 1. load_path while a job runs ─────────────────────────────────────────

def test_load_path_is_refused_while_a_job_runs(tmp_path):
    other = tmp_path / "B.mat"
    other.write_bytes(b"")
    status = _Recorder()
    sig = SignalState(filepath="C:/data/A.mat", raw=np.arange(10.0))
    # Nothing else is defined: any reset / load step would raise here.
    app = SimpleNamespace(_async_busy=True, _generation=7, signal=sig, _set_status=status)
    SignalController(app).load_path(str(other))
    assert app.signal.filepath == "C:/data/A.mat"
    assert app.signal.raw is sig.raw
    assert app._generation == 7
    assert "B.mat" in status.last()


def _signal_app():
    started: dict = {}
    app = SimpleNamespace(
        _generation=0, _async_busy=False, _set_status=_Recorder(),
        signal=SignalState(filepath="C:/data/A.mat"),
        detection=DetectionState(),
        analysis=AnalysisState(),
        btn_preview=None,
        _snapshot_params=lambda: {},
    )

    def _start_async(btn, label, status, worker, on_done, pass_result=False):
        started["on_done"] = on_done
    app._start_async = _start_async
    return app, started


def test_raw_load_landing_on_another_file_is_dropped():
    app, started = _signal_app()
    SignalController(app).load_raw_only()
    _new_file(app)
    started["on_done"]({"fs": 1000, "signal_raw": np.ones(50),
                        "signal_raw_norm": np.ones(50)})
    assert app.signal.raw is None and app.signal.fs != 1000
    assert "discarded" in app._set_status.last()


def test_detect_peaks_landing_on_another_file_is_dropped():
    app, started = _signal_app()
    SignalController(app).preview()
    _new_file(app)
    started["on_done"]({"fs": 1000, "signal_raw": np.ones(50),
                        "rpeaks_ok": np.array([1, 2, 3])})
    assert app.signal.raw is None and app.detection.rpeaks_ok is None


# ── 8 / 9. Analyze ────────────────────────────────────────────────────────

requires_nk = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")


@pytest.fixture
def ana_app(monkeypatch):
    from tests.ui import test_freshness as tf
    from tests.helpers import make_synth_ecg
    from ecg.ui import analysis_controller as ac
    box = SimpleNamespace(showwarning=lambda *a, **k: None, showerror=lambda *a, **k: None,
                          showinfo=lambda *a, **k: None)
    monkeypatch.setattr(ac, "messagebox", box)
    sig, fs, rp = make_synth_ecg(n_beats=60, seed=5)
    app = tf._App(sig, fs, rp)
    app._set_status = _Recorder()
    return app


def _run_analysis_with(app, between):
    """run_analysis with a job that calls *between* after the worker, before
    the result lands (what the user does while it runs)."""
    def _start_async(btn, label, status, worker, on_done, pass_result=False):
        out = worker()
        between()
        on_done(out)
    app._start_async = _start_async
    app.analysis_ctrl.run_analysis()


@requires_nk
def test_analyze_landing_on_another_file_is_dropped(ana_app):
    a = ana_app

    def _open_other():
        a._generation += 1
        a.detection.rpeaks_ok = None      # reset_for_new_file
        a.analysis.results = None
    _run_analysis_with(a, _open_other)
    assert a.analysis.results is None
    assert a.detection.rpeaks_ok is None
    assert a.analysis.artifact_report is None
    assert "discarded" in a._set_status.last()


@requires_nk
def test_analyze_after_a_peak_edit_keeps_the_edit(ana_app):
    a = ana_app
    edited: dict = {}

    def _edit():
        # What on_detail_click does: drop the results, commit the new peaks.
        a.detection_ctrl._invalidate_stale_results()
        a.detection.rpeaks_ok = np.delete(a.detection.rpeaks_ok, 20)
        edited["rp"] = a.detection.rpeaks_ok
    _run_analysis_with(a, _edit)
    assert a.detection.rpeaks_ok is edited["rp"]
    assert a.analysis.results is None
    assert a.freshness.status("core").state != UP_TO_DATE
    assert "peaks changed" in a._set_status.last()


@requires_nk
def test_analyze_without_changes_still_lands(ana_app):
    a = ana_app
    _run_analysis_with(a, lambda: None)
    assert a.analysis.results is not None


# ── 17. session check / restore ───────────────────────────────────────────

def _session_app():
    jobs: list = []
    app = SimpleNamespace(
        _generation=0, _async_busy=False, _set_status=_Recorder(),
        signal=SignalState(filepath="C:/data/A.mat"),
        session=SessionState(), btn_preview=None,
        loads=[],
    )
    app._load_raw_only = lambda: app.loads.append(app.signal.filepath)

    def _start_async(btn, label, status, worker, on_done, pass_result=False):
        jobs.append((worker, on_done))
    app._start_async = _start_async
    return app, jobs


def test_session_check_landing_on_another_file_is_dropped(monkeypatch):
    asked: list = []
    monkeypatch.setattr(sc_module, "load_session", lambda p: {"saved_at": "x", "n_beats": 3})
    monkeypatch.setattr(sc_module, "messagebox",
                        SimpleNamespace(askyesno=lambda *a, **k: asked.append(a) or True))
    app, jobs = _session_app()
    ctrl = SessionController(app)
    ctrl.try_restore_session("C:/data/A.mat")
    worker, on_done = jobs.pop()
    state = worker()
    _new_file(app)
    on_done(state)
    assert asked == [] and jobs == [] and app.loads == []


def test_session_restore_landing_on_another_file_is_dropped(monkeypatch):
    monkeypatch.setattr(sc_module, "load_session", lambda p: {"saved_at": "x", "n_beats": 3})
    monkeypatch.setattr(sc_module, "messagebox",
                        SimpleNamespace(askyesno=lambda *a, **k: True))
    app, jobs = _session_app()
    ctrl = SessionController(app)
    ctrl.try_restore_session("C:/data/A.mat")
    worker, on_done = jobs.pop()
    on_done(worker())                     # Yes: the restore job starts
    _restore_worker, restore_done = jobs.pop()
    _new_file(app)
    # An empty bundle: any write would raise KeyError.
    restore_done({})
    assert app.signal.filepath == "C:/data/B.mat"


def test_restore_worker_reads_the_file_it_was_started_for(monkeypatch):
    read: list = []

    def _load(path, channel):
        read.append(path)
        raise IOError("stop here")
    monkeypatch.setattr(session_service, "load_mat_signal", _load)
    app = SimpleNamespace(signal=SignalState(filepath="C:/data/B.mat"))
    with pytest.raises(RuntimeError):
        SessionController(app).restore_session_worker({"fs": 1000}, "C:/data/A.mat")
    assert read == ["C:/data/A.mat"]


# ── 18. Save Session while another file is opened ─────────────────────────

def test_save_session_uses_the_state_of_the_file_it_saves(monkeypatch):
    upserts: list = []
    samples: list = []
    deleted: list = []
    monkeypatch.setattr(sc_module, "_DB_AVAILABLE", True)
    monkeypatch.setattr(sc_module, "save_session", lambda fp, state: "C:/data/A.ecgsession")
    monkeypatch.setattr(sc_module, "upsert_recording", lambda **kw: upserts.append(kw))
    monkeypatch.setattr(sc_module, "save_training_sample",
                        lambda fp, flt, fs, rp, progress_cb=None, source_path="":
                        samples.append((fs, rp)) or 7)
    monkeypatch.setattr(sc_module, "delete_training_sample", lambda fp: deleted.append(fp))
    # A is a training example whose peaks were edited since it was added:
    # Save Session refreshes it (and only then touches the training cache).
    monkeypatch.setattr(sc_module, "training_example_state", lambda fp, rp: "changed")
    import ecg.io.session as io_session
    monkeypatch.setattr(io_session, "_file_fingerprint", lambda p: "fp-" + p)

    fs = 1000
    rp_a = np.array([100, 300, 500, 700, 900])
    rr_df = pd.DataFrame({"RR_ms": [200.0, 200.0, 200.0, 200.0]})
    hrv = pd.DataFrame({"HRV_SDNN": [1.5], "HRV_RMSSD": [2.5]})
    ui_updates: list = []
    app = SimpleNamespace(
        _generation=0, _async_busy=False, _set_status=_Recorder(),
        signal=SignalState(filepath="C:/data/A.mat", filtered=np.zeros(1000), fs=fs,
                           time=np.arange(1000) / fs),
        detection=DetectionState(rpeaks_ok=rp_a),
        analysis=AnalysisState(results={"rr_df": rr_df, "hrv_time": hrv}),
        session=SessionState(recording_notes="notes of A", verified_for_training=True,
                             dirty=True),
        ent_project_name=None, btn_save_session=None,
        refresh_ml_status=lambda: None,
        after=lambda ms, fn=None: None,
    )

    def _start_async(btn, label, status, worker, on_done, pass_result=False):
        # File B is opened before the worker reads anything.
        _new_file(app)
        app.signal.time, app.signal.fs = None, 500
        app.detection.rpeaks_ok = None
        app.analysis.results = None
        app.session = SessionState(recording_notes="notes of B",
                                   verified_for_training=False, dirty=True)
        on_done(worker())
    app._start_async = _start_async
    ctrl = SessionController(app)
    ctrl.collect_session_state = lambda: {"saved_at": "now"}
    ctrl.update_session_ui = lambda *a, **k: ui_updates.append((a, k))

    ctrl.save_session()

    assert len(upserts) == 1
    row = upserts[0]
    assert row["filepath"] == "C:/data/A.mat"
    assert row["notes"] == "notes of A"
    assert row["verified_for_training"] is True
    assert row["stats"]["hr_mean"] == pytest.approx(300.0)
    assert row["stats"]["sdnn"] == 1.5 and row["stats"]["rmssd"] == 2.5
    assert row["stats"]["duration_s"] == pytest.approx(1.0)
    assert row["stats"]["n_peaks"] == 5
    assert deleted == [] and len(samples) == 1
    assert samples[0][0] == fs and samples[0][1] is rp_a
    # B's sidebar is not told it was saved, nor its dirty flag cleared.
    assert ui_updates == []
    assert app.session.dirty is True
    assert "before this one" in app._set_status.last()
