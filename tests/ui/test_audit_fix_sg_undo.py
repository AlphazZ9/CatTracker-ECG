"""
Audit 2026-11-13, bugs 20 and 21, driven through stub apps (no Tk window):

* 20 -- the 'SG window (ms)' entry reaches the SG + Derivative detector
  (it used to always run with its 20 ms default);
* 21 -- Detect Peaks / a raw load clear the undo/redo stacks along with the
  manual edits they snapshot, so Ctrl+Z cannot put the previous run's
  exclusions/additions back onto the new detection.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from ecg.core.models import FilterParams
from ecg.ui import session_controller as sc_module
from ecg.ui.app_helpers import HelpersMixin
from ecg.ui.detection_controller import DetectionController
from ecg.ui.session_controller import SessionController
from ecg.ui.signal_controller import SignalController
from ecg.ui.state import (AnalysisState, DetectionState, SessionState,
                          SignalState, UIState)

FS = 2000


class _Entry:
    def __init__(self, text: str) -> None:
        self.text = text

    def get(self) -> str:
        return self.text


def _snapshot(monkeypatch, sg_text):
    monkeypatch.setattr(sc_module.FilterParams, "from_widgets",
                        classmethod(lambda cls, app: FilterParams(
                            detection_method="SG + Derivative", invert_signal=True,
                            no_filter=True)))
    app = SimpleNamespace(analysis=AnalysisState())
    app._safe_float = lambda w, d=0.0: HelpersMixin._safe_float(app, w, d)
    if sg_text is not None:
        app.ent_sg_window_ms = _Entry(sg_text)
    return SessionController(app).snapshot_params()


# ── 20. SG window ────────────────────────────────────────────────────────

def test_snapshot_holds_the_sg_window(monkeypatch):
    assert _snapshot(monkeypatch, "6")["sg_window_ms"] == 6.0


def test_snapshot_sg_window_falls_back_to_default(monkeypatch):
    assert _snapshot(monkeypatch, "abc")["sg_window_ms"] == 20.0
    assert _snapshot(monkeypatch, None)["sg_window_ms"] == 20.0


def test_sg_detector_receives_the_typed_window(monkeypatch):
    seen: dict = {}

    def _fake_sg(sig, fs, sg_window_ms, **kw):
        seen["sg_window_ms"] = sg_window_ms
        return np.array([10, 20], dtype=int), np.ones(2), 0.5
    from ecg.core import pipeline
    monkeypatch.setattr(pipeline, "detect_peaks_sg_derivative", _fake_sg)
    params = _snapshot(monkeypatch, "6")
    SignalController._preview_bundle(np.sin(np.arange(4000) / 10.0), FS, params)
    assert seen["sg_window_ms"] == 6.0


# ── 21. undo/redo after re-detect ────────────────────────────────────────

def _edited_app():
    app = MagicMock()
    app.analysis, app.detection = AnalysisState(), DetectionState()
    app.signal, app.ui, app.session = SignalState(fs=FS), UIState(), SessionState()
    app._slots, app._kpi, app._generation = {}, {}, 0
    app._EDIT_UNDO_LIMIT = 50
    app.detection_ctrl = DetectionController(app)
    det = app.detection
    # Two edits on the previous run, the second one undone.
    det.manual_excluded = {600}
    det.manual_added = {777}
    det.edit_undo = [(frozenset(), frozenset())]
    det.edit_redo = [(frozenset({600, 1000}), frozenset({777}))]
    return app


def _bundle():
    sig = np.zeros(4000)
    rp = np.arange(200, 3800, 400)
    return {"fs": FS, "signal_raw": sig, "signal_raw_norm": sig,
            "signal_flt": sig, "all_cands": rp, "all_proms": np.ones(len(rp)),
            "no_filter_mode": False, "rpeaks_ok": rp, "rpeaks_rej": np.array([], int),
            "thresh_amp": 0.5, "detected_ch": "ECG", "requested_ch": "ECG",
            "fs_from_file": True, "detected_fs": FS, "dur_s": 2.0}


def _assert_history_gone(app):
    assert app.detection.edit_undo == [] and app.detection.edit_redo == []
    app.detection_ctrl.undo_edit()
    app.detection_ctrl.redo_edit()
    assert app.detection.manual_excluded == set()
    assert app.detection.manual_added == set()


def test_re_detect_clears_undo_and_redo():
    app = _edited_app()
    SignalController(app).on_preview_done(_bundle())
    _assert_history_gone(app)


def test_raw_load_clears_undo_and_redo():
    app = _edited_app()
    SignalController(app).on_raw_load_done(_bundle())
    _assert_history_gone(app)


def test_re_detect_refreshes_the_undo_buttons():
    app = _edited_app()
    app.detection_ctrl = MagicMock()
    SignalController(app).on_preview_done(_bundle())
    app.detection_ctrl.update_undo_btns.assert_called()
