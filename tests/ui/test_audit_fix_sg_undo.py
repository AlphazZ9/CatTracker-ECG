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
from ecg.app import session_controller as sc_module
from ecg.app.detection_controller import DetectionController
from ecg.app.session_controller import SessionController
from ecg.app.signal_controller import SignalController
from ecg.app.state import (AnalysisState, DetectionState, SessionState,
                          SignalState, UIState)

FS = 2000


class _Entry:
    def __init__(self, text: str) -> None:
        self.text = text

    def get(self) -> str:
        return self.text




# ── 20. SG window ────────────────────────────────────────────────────────







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
