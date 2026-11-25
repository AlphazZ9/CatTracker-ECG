# -*- coding: utf-8 -*-
"""The ML detector's training examples: one action adds / removes / updates
the open recording as an example, right away; Save Session only refreshes an
existing example whose peaks changed; the status is in plain words; the
training dialog lists the examples (each removable) and gates Train; Method =
ML Detector without a model says what to do."""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from ecg.core import ml_detector
from ecg.app import ml_examples
from ecg.app import session_controller as sc_module
from ecg.app.session_controller import SessionController
from ecg.app.state import DetectionState, SessionState, SignalState
from tests.helpers import add_qrs_pulse, rpeaks_from_rr

FS = 1000


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Training cache and model files in tmp_path (never the user's)."""
    d = tmp_path / "ml_training"
    monkeypatch.setattr(ml_detector, "TRAINING_DATA_DIR", d)
    for name in ("MODEL_PATH", "META_PATH", "MODEL_BACKUP_PATH", "META_BACKUP_PATH",
                 "LEGACY_MODEL_PATH"):
        monkeypatch.setattr(ml_detector, name, tmp_path / f"model_{name}")
    monkeypatch.setitem(ml_detector._model_cache, "mtime", None)
    monkeypatch.setitem(ml_detector._model_cache, "model", None)
    return d


def _ecg(seed: int = 0, n_beats: int = 60):
    rng = np.random.default_rng(seed)
    rp = rpeaks_from_rr(list(110 + rng.normal(0, 4, n_beats)), FS, start=200)
    sig = rng.normal(0, 0.02, rp[-1] + 400)
    for c in rp:
        add_qrs_pulse(sig, int(c), FS)
    return sig, rp


def _recording(tmp_path, name: str = "rec.mat"):
    p = tmp_path / name
    p.write_bytes(name.encode())
    return str(p)


def _fp(path: str) -> str:
    from ecg.io.session import _file_fingerprint
    return _file_fingerprint(path)


# ── core: what an example keeps ──────────────────────────────────────────

def test_example_keeps_its_peaks_and_tells_when_they_change(tmp_path, cache):
    sig, rp = _ecg()
    path = _recording(tmp_path)
    assert ml_detector.training_example_state(_fp(path), rp) == "absent"
    ml_detector.save_training_sample(_fp(path), sig, FS, rp, source_path=path)
    assert ml_detector.training_example_state(_fp(path), rp) == "same"
    assert ml_detector.training_example_state(_fp(path), rp[::-1]) == "same"   # order-free
    assert ml_detector.training_example_state(_fp(path), rp[:-1]) == "changed"
    [entry] = ml_detector.list_training_files()
    assert entry["n_peaks"] == len(rp) and entry["source_path"] == path
    assert entry["rate_bpm"] == pytest.approx(60000 / 110, rel=0.05)
    assert entry["added_at"][:10] == datetime.now().date().isoformat()


def test_older_cache_without_peaks_still_lists_and_counts(tmp_path, cache):
    cache.mkdir(parents=True)
    labels = np.array([0, 1, 1, 0, 1])
    np.savez(cache / "legacyfp.npz", features=np.zeros((5, 12)), labels=labels)
    assert ml_detector.training_example_state("legacyfp", np.array([1, 2])) == "unknown"
    [entry] = ml_detector.list_training_files()
    assert entry["n_peaks"] == 3 and entry["rate_bpm"] is None
    assert entry["source_path"] == "" and entry["added_at"]


def test_detect_without_model_says_what_to_do(cache):
    with pytest.raises(ml_detector.NoModelError) as exc:
        ml_detector.detect_peaks_ml(np.zeros(1000), FS)
    assert isinstance(exc.value, RuntimeError)          # old callers still catch it
    msg = str(exc.value)
    assert "no model is trained yet" in msg and "training examples" in msg
    assert "Train model" in msg


# ── plain status ─────────────────────────────────────────────────────────

def _examples(n: int, mtime: float = 0.0) -> "list[dict]":
    return [{"fingerprint": f"fp{i}", "mtime": mtime} for i in range(n)]


def test_status_without_model():
    short, full, tone = ml_examples.status_text(_examples(2), None)
    assert full == "No model yet · 2 training examples — add at least 5, then Train"
    assert short == "No model yet · 2 examples" and tone == ml_examples.TONE_MUTED
    _s, full, _t = ml_examples.status_text(_examples(1), None)
    assert full.startswith("No model yet · 1 training example —")
    _s, full, _t = ml_examples.status_text(_examples(6), None)
    assert full == "No model yet · 6 training examples — ready to Train"


def test_status_with_model_and_retrain_suggestion():
    trained = datetime(2026, 11, 1, 12, 0)
    meta = {"trained_at": trained.isoformat(), "n_training_files": 6,
            "holdout_accuracy": 0.9812,
            "training_fingerprints": [f"fp{i}" for i in range(6)]}
    old = (trained - timedelta(days=1)).timestamp()
    short, full, tone = ml_examples.status_text(_examples(6, old), meta)
    assert full == ("6 training examples · model trained on 6 (98 % of candidate peaks "
                    "classified correctly on held-out data)")
    assert tone == ml_examples.TONE_OK and short == "Model ready · 6 examples"
    examples = _examples(6, old) + [{"fingerprint": "new", "mtime": old + 3 * 86400}]
    short, full, tone = ml_examples.status_text(examples, meta)
    assert full.endswith("· retrain suggested: 1 new or updated example")
    assert full.startswith("7 training examples · model trained on 6 (98 %")
    assert tone == ml_examples.TONE_WARN and short.startswith("Retrain suggested")
    _s, full, _t = ml_examples.status_text(_examples(5, old), meta)
    assert full.endswith("retrain suggested: 1 removed")
    # A model saved before training_fingerprints existed: nothing invented.
    legacy = {k: v for k, v in meta.items() if k != "training_fingerprints"}
    _s, full, _t = ml_examples.status_text(_examples(5, old), legacy)
    assert "retrain" not in full
    # No jargon left in what the user reads.
    for text in (full, ml_examples.TIP_STATUS, ml_examples.TIP_ADD):
        assert "acc=" not in text and "f1" not in text.lower() and "verified" not in text.lower()


def test_one_example_button_per_state():
    b = ml_examples.example_button(True, 120, "absent")
    assert (b.mode, b.text, b.enabled, b.hint) == ("add", ml_examples.BTN_ADD, True, "")
    b = ml_examples.example_button(True, 0, "absent")
    assert not b.enabled and "Detect the peaks" in b.hint
    b = ml_examples.example_button(False, 0, "absent")
    assert not b.enabled and "Open a recording" in b.hint
    for state in ("same", "unknown"):
        b = ml_examples.example_button(True, 120, state)
        assert (b.mode, b.text, b.show_remove) == ("remove", ml_examples.BTN_REMOVE, False)
    b = ml_examples.example_button(True, 119, "changed")
    assert (b.mode, b.text, b.show_remove) == ("update", ml_examples.BTN_UPDATE, True)


def test_train_gate():
    assert ml_examples.train_gate(1)[0] is False and "at least 2" in ml_examples.train_gate(1)[1]
    ok, warn = ml_examples.train_gate(3)
    assert ok and "At least 5" in warn
    assert ml_examples.train_gate(5) == (True, "")


def test_train_result_in_plain_words():
    text = ml_examples.train_result_text(
        {"ok": True, "n_files": 6, "accuracy": 0.97, "recall": 0.95, "precision": 0.99,
         "previous_accuracy": 0.95})
    assert text.startswith("✓ Model trained on 6 training examples.")
    assert "97 % of candidate peaks were classified correctly" in text
    assert "found 95 % of the true R-peaks" in text and "+2 points" in text
    assert ml_examples.train_result_text({"ok": False, "message": "No."}) == "✗ No."


# ── the one action, on the session controller ────────────────────────────

class _Rec:
    def __init__(self):
        self.calls = []

    def __call__(self, *a, **k):
        self.calls.append(a)

    def last(self):
        return self.calls[-1][0] if self.calls else ""


def _app(path, sig, rp):
    app = SimpleNamespace(
        _generation=0, _set_status=_Rec(), refresh_ml_status=_Rec(),
        signal=SignalState(filepath=path, filtered=sig, fs=FS, time=np.arange(len(sig)) / FS),
        detection=DetectionState(rpeaks_ok=rp),
        analysis=SimpleNamespace(results=None),
        session=SessionState(), btn_ml_example=None, ent_project_name=None,
        after=lambda ms, fn=None: None, _set_progress=lambda *a: None,
    )
    app.example_refreshes = []
    app._refresh_ml_example = lambda force=False: app.example_refreshes.append(force)
    app._start_async = (lambda btn, label, status, worker, on_done, pass_result=False:
                        on_done(worker()))
    return app


def test_add_then_remove_the_open_recording(tmp_path, cache, monkeypatch):
    monkeypatch.setattr(sc_module, "_DB_AVAILABLE", True)
    flags = []
    monkeypatch.setattr(sc_module, "set_verified", lambda p, v: flags.append((p, v)))
    sig, rp = _ecg()
    path = _recording(tmp_path)
    app = _app(path, sig, rp)
    ctrl = SessionController(app)

    ctrl.add_training_example()
    assert ml_detector.training_example_state(_fp(path), rp) == "same"
    assert app.session.verified_for_training is True and flags == [(path, True)]
    assert "Added to the training examples" in app._set_status.last()
    assert app.refresh_ml_status.calls and app.example_refreshes == [True]
    assert ctrl.is_training_example(path)

    # Declined confirmation: nothing changes.
    monkeypatch.setattr(sc_module.messagebox, "askyesno", lambda *a, **k: False)
    assert ctrl.remove_training_example() is False
    assert ml_detector.training_example_state(_fp(path), rp) == "same"

    monkeypatch.setattr(sc_module.messagebox, "askyesno", lambda *a, **k: True)
    assert ctrl.remove_training_example() is True
    assert ml_detector.training_example_state(_fp(path), rp) == "absent"
    assert app.session.verified_for_training is False and flags[-1] == (path, False)
    assert "removed from the training examples" in app._set_status.last()


def test_update_replaces_the_peaks_of_an_example(tmp_path, cache, monkeypatch):
    monkeypatch.setattr(sc_module, "_DB_AVAILABLE", False)
    sig, rp = _ecg()
    path = _recording(tmp_path)
    app = _app(path, sig, rp)
    ctrl = SessionController(app)
    ctrl.add_training_example()
    edited = rp[:-2]
    app.detection.rpeaks_ok = edited
    assert ml_detector.training_example_state(_fp(path), edited) == "changed"
    ctrl.add_training_example()
    assert ml_detector.training_example_state(_fp(path), edited) == "same"
    assert "Training example updated" in app._set_status.last()
    assert len(ml_detector.list_training_files()) == 1


def test_add_needs_peaks(tmp_path, cache, monkeypatch):
    warned = []
    monkeypatch.setattr(sc_module.messagebox, "showwarning", lambda *a, **k: warned.append(a))
    sig, _rp = _ecg()
    app = _app(_recording(tmp_path), sig, np.array([], dtype=int))
    SessionController(app).add_training_example()
    assert warned and ml_detector.list_training_files() == []


def _save(ctrl, app, monkeypatch, calls):
    real = ml_detector.save_training_sample
    monkeypatch.setattr(sc_module, "save_training_sample",
                        lambda *a, **k: calls.append(a) or real(*a, **k))
    monkeypatch.setattr(sc_module, "save_session", lambda fp, state: "x.ecgsession")
    ctrl.save_session_worker({"saved_at": "now"}, app.signal.filepath, app.signal.filtered)


def test_save_session_does_not_add_or_remove_examples(tmp_path, cache, monkeypatch):
    monkeypatch.setattr(sc_module, "_DB_AVAILABLE", False)
    sig, rp = _ecg()
    path = _recording(tmp_path)
    app = _app(path, sig, rp)
    ctrl = SessionController(app)
    calls: list = []
    # Not an example: Save Session leaves the training set alone.
    _save(ctrl, app, monkeypatch, calls)
    assert calls == [] and ml_detector.list_training_files() == []
    # An example, peaks unchanged: no rewrite.
    ml_detector.save_training_sample(_fp(path), sig, FS, rp, source_path=path)
    app.session.verified_for_training = True
    _save(ctrl, app, monkeypatch, calls)
    assert calls == []
    # Peaks edited since it was added: the example follows them.
    app.detection.rpeaks_ok = rp[1:]
    _save(ctrl, app, monkeypatch, calls)
    assert len(calls) == 1
    assert ml_detector.training_example_state(_fp(path), rp[1:]) == "same"
    # The flag off (e.g. a stale session) never deletes an example.
    app.session.verified_for_training = False
    _save(ctrl, app, monkeypatch, calls)
    assert len(ml_detector.list_training_files()) == 1


# ── Method = ML Detector without a model ─────────────────────────────────

def test_choosing_ml_without_a_model_opens_the_section_and_says_why(cache, monkeypatch):
    from ecg.app.detection_controller import DetectionController
    expanded = []
    app = SimpleNamespace(_sg_frame=None, lbl_ml_status=object(), refresh_ml_status=_Rec(),
                          _set_ml_expanded=lambda v: expanded.append(v), _set_status=_Rec())
    ctrl = DetectionController(app)
    ctrl.on_det_method_change("Wavelet (CWT)")
    assert expanded == [] and app._set_status.calls == []
    assert app.refresh_ml_status.calls      # the no-model note leaves on another Method
    app.refresh_ml_status.calls.clear()
    ctrl.on_det_method_change("ML Detector")
    assert expanded == [True] and app.refresh_ml_status.calls
    # The status bar only points at the section (too narrow for the how-to).
    assert app._set_status.last() == ml_examples.NO_MODEL_STATUS
    assert len(ml_examples.NO_MODEL_STATUS) < 80
    monkeypatch.setattr(ml_detector.MLPeakModel, "exists", staticmethod(lambda: True))
    app._set_status.calls.clear()
    ctrl.on_det_method_change("ML Detector")
    assert expanded == [True] and app._set_status.calls == []


# ── the training dialog ──────────────────────────────────────────────────



def test_training_records_its_examples_and_status_suggests_retrain(tmp_path, cache):
    for seed in range(3):
        sig, rp = _ecg(seed=seed)
        path = _recording(tmp_path, f"m{seed}.mat")
        ml_detector.save_training_sample(_fp(path), sig, FS, rp, source_path=path)
    result = ml_detector.train_model(min_samples=10)
    assert result["ok"], result
    meta = ml_detector.MLPeakModel.load().meta
    assert len(meta["training_fingerprints"]) == 3
    _s, full, tone = ml_examples.status_text(ml_detector.list_training_files(), meta)
    assert "retrain" not in full and tone == ml_examples.TONE_OK
    ml_detector.delete_training_sample(meta["training_fingerprints"][0])
    _s, full, tone = ml_examples.status_text(ml_detector.list_training_files(), meta)
    assert full.endswith("retrain suggested: 1 removed") and tone == ml_examples.TONE_WARN


def test_status_line_says_what_to_do_when_ml_is_chosen_without_a_model():
    examples = [{"fingerprint": "a", "mtime": 0.0}] * 2
    short, full, tone = ml_examples.status_text(examples, None, ml_selected=True)
    assert short == "No model yet · 2 examples" and tone == ml_examples.TONE_WARN
    assert full.startswith("ML Detector needs a trained model:")
    assert "another Method" in full and "5 or more examples" in full and "click Train" in full
    # The panel keeps the whole how-to in that state (no peaks, no plots then).
    assert ml_examples.panel_text(examples, None, ml_selected=True) == full
    assert ml_examples.panel_text(examples, None) == "Add at least 5 training examples, then Train."
    assert ml_examples.panel_text(examples * 3, None) == "Enough training examples: ready to Train."
    assert ml_examples.panel_text(examples, {"n_training_files": 6, "holdout_accuracy": 0.976})         == "Trained on 6 · 98 % correct on held-out data"
    # With a model, or on another Method: the usual line.
    _s, full, _t = ml_examples.status_text(examples, None)
    assert full.startswith("No model yet · 2 training examples")
    meta = {"n_training_files": 2, "holdout_accuracy": 0.9, "trained_at": "2999-01-01T00:00:00"}
    _s, full, _t = ml_examples.status_text(examples, meta, ml_selected=True)
    assert full.startswith("2 training examples · model trained on 2")


def test_remove_waits_while_a_job_may_write_the_example(tmp_path, cache, monkeypatch):
    """A running Add / Update / Save Session writes the .npz (and the flag)
    at its end: a removal meanwhile would be undone, so it is refused."""
    monkeypatch.setattr(sc_module, "_DB_AVAILABLE", False)
    sig, rp = _ecg()
    path = _recording(tmp_path)
    app = _app(path, sig, rp)
    ctrl = SessionController(app)
    ctrl.add_training_example()
    asked = []
    monkeypatch.setattr(sc_module.messagebox, "askyesno", lambda *a, **k: asked.append(a) or True)
    app._async_busy = True
    assert ctrl.remove_training_example() is False
    assert asked == [] and app._set_status.last() == ml_examples.BUSY_MESSAGE
    assert ml_detector.training_example_state(_fp(path), rp) == "same"
    assert app.session.verified_for_training is True
    app._async_busy = False
    assert ctrl.remove_training_example() is True
    assert ml_detector.training_example_state(_fp(path), rp) == "absent"




