"""Compare segments setup kept by Save Session and a UI rebuild: the rows
(label, from, to), Block (s), the Δ % reference, the view and the jitter
flag, and the comparison shown -- re-run on restore (no statistics in the
session file), or re-shown from the kept result after a UI rebuild."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest


from ecg.core.segment_plan import PAIRS, PAIRWISE
from ecg.io.session import _SessionEncoder, _session_decode_hook
from ecg.app.session_controller import SessionController

from ecg.app.compare_helpers import normalize_compare_setup

WINS = [("Pre 1", 0, 150), ("Post 1", 150, 300), ("Pre 2", 300, 450), ("Post 2", 450, 600)]






def _wait(app, frame):
    t0 = time.time()
    while app.analysis.last_comparison is None and time.time() - t0 < 60:
        frame.update()
        time.sleep(0.02)
    frame.update()








def _session_json(setup):
    text = json.dumps({"compare_segments": setup}, cls=_SessionEncoder)
    return json.loads(text, object_hook=_session_decode_hook)["compare_segments"]


# ── normalisation ─────────────────────────────────────────────────────────

def test_normalize_rejects_old_or_broken_blocks():
    assert normalize_compare_setup(None) is None
    assert normalize_compare_setup({}) is None
    assert normalize_compare_setup({"rows": "x"}) is None
    s = normalize_compare_setup({"rows": [["A", 0, 10], ["B", "1x", None], "bad"],
                                 "ref": 9, "view": "3d", "block_s": None,
                                 "run": {"rows": [["A", 5, 1]], "block_s": 10}})
    assert s["rows"] == [["A", 0.0, 10.0], ["B", "1x", ""]]        # typed text kept
    assert (s["ref"], s["view"], s["run"], s["ran"]) == (0, "rr", None, False)
    assert s["block_s"] == "10"
    for ref in (0, 3, PAIRWISE, PAIRS):
        assert normalize_compare_setup({"rows": [["A", 0, 1]], "ref": ref})["ref"] == ref
    assert normalize_compare_setup({"rows": [["A", 0, 1]], "ref": True})["ref"] == 0
    # A run alone gives the rows.
    s = normalize_compare_setup({"run": {"rows": [["A", 0, 5]], "block_s": "10"}})
    assert s["rows"] == [["A", 0.0, 5.0]] and s["run"]["block_s"] == 10.0 and s["ran"]


# ── round trip collect -> restore ─────────────────────────────────────────















def test_session_state_holds_the_setup_and_the_worker_passes_it(monkeypatch):
    from tests.ui.test_event_locked_controller import _computed, FS
    from ecg.services import session_service  # where restore reloads the signal
    app, _ = _computed()
    app.sl_thr = None
    app._qtc_formula = lambda: "mitchell"
    setup = normalize_compare_setup({"rows": [["A", 0, 5], ["B", 5, 9]], "ref": 1,
                                     "run": {"rows": [["A", 0, 5], ["B", 5, 9]],
                                             "block_s": 10}})
    app._compare_setup_snapshot = lambda: setup
    state = SessionController(app).collect_session_state()
    assert state["compare_segments"] == setup
    state = json.loads(json.dumps(state, cls=_SessionEncoder), object_hook=_session_decode_hook)
    monkeypatch.setattr(session_service, "load_mat_signal",
                        lambda path, ch=None: (np.zeros(1000), "ECG", None, FS))
    app._compute_preview_bundle = lambda sig, fs, params: {}
    bundle = SessionController(app).restore_session_worker(state)
    assert normalize_compare_setup(bundle["compare_segments"]) == setup
    del state["compare_segments"]
    assert SessionController(app).restore_session_worker(state)["compare_segments"] is None
    # nothing for this recording: no block written
    app._compare_setup_snapshot = lambda: None
    assert "compare_segments" not in SessionController(app).collect_session_state()


# ── review fixes: recording key, comparison in flight, Poincaré columns ──



def test_opening_a_file_drops_the_previous_compare_setup():
    from unittest.mock import MagicMock
    from ecg.app.signal_controller import SignalController
    from ecg.app.freshness import FreshnessService
    from ecg.app.state import AnalysisState, UIState
    app = MagicMock()
    app.analysis = AnalysisState()
    app.ui = UIState()
    app.ui.hover_after_id = None
    app.freshness = FreshnessService(app)
    app._slots, app._kpi, app._arr_card_widgets = {}, {}, []
    app._compare_pending_setup = (("a.mat", "ECG", 1, 10), {"rows": [["A", 0, 5]]})
    app._compare_setup_getter = lambda: {"rows": [["A", 0, 5]]}
    app._compare_setup_key = ("a.mat", "ECG", 1, 10)
    SignalController(app).reset_for_new_file()
    assert app._compare_pending_setup is None
    assert app._compare_setup_getter is None and app._compare_setup_key is None






