"""Compare segments setup kept by Save Session and a UI rebuild: the rows
(label, from, to), Block (s), the Δ % reference, the view and the jitter
flag, and the comparison shown -- re-run on restore (no statistics in the
session file), or re-shown from the kept result after a UI rebuild."""
from __future__ import annotations

import json
import time
import tkinter as tk
from types import SimpleNamespace

import numpy as np
import pytest

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.core.segment_plan import PAIRS, PAIRWISE
from ecg.io.session import _SessionEncoder, _session_decode_hook
from ecg.ui import compare_segments as cs
from ecg.ui.compare_segments import CompareSegmentsMixin, normalize_compare_setup
from ecg.ui.session_controller import SessionController

from tests.ui.test_compare_reference import _app, _entries, _ref_combo, _run, _table_text, _walk

M = CompareSegmentsMixin
WINS = [("Pre 1", 0, 150), ("Post 1", 150, 300), ("Pre 2", 300, 450), ("Post 2", 450, 600)]


@pytest.fixture
def root(monkeypatch):
    class _Inline:
        def __init__(self, target=None, daemon=None, **_kw):
            self._target = target

        def start(self):
            self._target()
    import threading
    monkeypatch.setattr(threading, "Thread", _Inline)
    try:
        r = ctk.CTk()
    except tk.TclError:
        pytest.skip("no display")
    r.withdraw()
    yield r
    for after_id in r.tk.splitlist(r.tk.call("after", "info")):
        r.tk.call("after", "cancel", after_id)
    r.destroy()


def _build(app, root):
    frame = ctk.CTkFrame(root)
    frame.pack(fill="both", expand=True)
    M._build_compare_segments(app, frame)
    frame.update()
    return frame


def _wait(app, frame):
    t0 = time.time()
    while app.analysis.last_comparison is None and time.time() - t0 < 60:
        frame.update()
        time.sleep(0.02)
    frame.update()


def _rows(frame):
    out = []
    for lbl_e in (e for e in _entries(frame) if e.cget("placeholder_text") == "Label"):
        kids = [c for c in lbl_e.master.winfo_children() if isinstance(c, ctk.CTkEntry)]
        out.append(tuple(k.get() for k in kids))
    return out


def _view_button(frame, text):
    return next(w for w in _walk(frame) if isinstance(w, ctk.CTkButton)
                and w.cget("text") == text)


def _jitter(frame):
    return next(w for w in _walk(frame) if isinstance(w, ctk.CTkCheckBox))


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

def test_setup_round_trip_and_rerun_reproduces_the_comparison(root):
    app = _app()
    frame = _build(app, root)
    assert M._compare_setup_snapshot(app)["run"] is None           # nothing run yet
    _run(app, frame, WINS)
    _ref_combo(frame)._dropdown_callback("Pairs (2 vs 1, 4 vs 3, 6 vs 5)")
    _view_button(frame, "Poincaré").invoke()
    assert _jitter(frame).get()                                      # on by default
    _jitter(frame).toggle()                                          # -> off
    frame.update()
    res = app.analysis.last_comparison
    assert res.comps == [None, 0, None, 2]
    table = _table_text(frame)
    # A row typed after the run (not run): kept as typed, the shown result
    # still the run's windows.
    rows_typed = _rows(frame)
    lbl_e = next(e for e in _entries(frame) if e.cget("placeholder_text") == "Label")
    kids = [c for c in lbl_e.master.winfo_children() if isinstance(c, ctk.CTkEntry)]
    kids[1].delete(0, "end"); kids[1].insert(0, "12x")
    setup = M._compare_setup_snapshot(app)
    assert setup["rows"][0] == ["Pre 1", "12x", 150.0]
    assert (setup["ref"], setup["view"], setup["jitter"], setup["ran"]) == (PAIRS, "poincare",
                                                                            False, True)
    assert setup["run"]["rows"] == [[l, float(a), float(b)] for l, a, b in WINS]
    assert setup["block_s"] == "30" or float(setup["block_s"]) > 0

    # Restore on a fresh app for the same recording (no stats in the JSON).
    back = _session_json(setup)
    app2 = _app()
    app2._compare_jitter = True
    M._restore_compare_setup(app2, back)          # tab not open: applied when built
    assert app2._compare_pending_setup is not None and app2._compare_built_for is None
    frame2 = _build(app2, root)
    _wait(app2, frame2)
    assert app2._compare_pending_setup is None
    res2 = app2.analysis.last_comparison
    assert res2 is not res and res2.comps == res.comps
    assert [s["label"] for s in res2.segs] == [w[0] for w in WINS]
    for a, b in zip(res.segs, res2.segs):
        for k in ("hr_mean", "rmssd", "sdnn", "n_beats", "sd1", "lf_hf"):
            assert a.get(k) == b.get(k) or (np.isnan(a.get(k)) and np.isnan(b.get(k)))
    assert _table_text(frame2) == table
    assert _rows(frame2)[0] == ("Pre 1", "12x", "150") and _rows(frame2)[1:] == rows_typed[1:]
    assert _ref_combo(frame2).get() == "Pairs (2 vs 1, 4 vs 3, 6 vs 5)"
    assert app2._compare_jitter is False and not _jitter(frame2).get()
    assert M._compare_setup_snapshot(app2)["view"] == "poincare"


def test_typed_rows_without_a_run_restore_without_running(root):
    app = _app()
    frame = _build(app, root)
    add = next(w for w in _walk(frame) if isinstance(w, ctk.CTkButton)
               and "Add segment" in w.cget("text"))
    add.invoke()
    setup = M._compare_setup_snapshot(app)
    assert len(setup["rows"]) == 3 and setup["run"] is None and not setup["ran"]
    app2 = _app()
    M._restore_compare_setup(app2, _session_json(setup))
    frame2 = _build(app2, root)
    assert len(_rows(frame2)) == 3
    assert app2.analysis.last_comparison is None


def test_old_session_without_the_block_keeps_the_defaults(root):
    app = _app()
    app._compare_pending_setup = None
    M._restore_compare_setup(app, None)
    assert app._compare_pending_setup is None
    frame = _build(app, root)
    assert [r[0] for r in _rows(frame)] == ["Segment A", "Segment B"]
    assert app.analysis.last_comparison is None


def test_setup_of_another_recording_is_not_applied(root):
    app = _app()
    app._compare_pending_setup = ((12345, 1), normalize_compare_setup(
        {"rows": [["X", 0, 10]], "run": None}))
    frame = _build(app, root)
    assert [r[0] for r in _rows(frame)] == ["Segment A", "Segment B"]


def test_restore_while_the_tab_is_open_rebuilds_it_now():
    calls = []
    app = SimpleNamespace(_signal_flt=np.zeros(10), _compare_built_for=("sig",),
                          tabs=SimpleNamespace(get=lambda: cs.TAB_COMPARE),
                          _refresh_compare_tab=lambda: calls.append(1))
    M._restore_compare_setup(app, {"rows": [["A", 0, 5]]})
    assert calls == [1] and app._compare_built_for is None
    assert app._compare_pending_setup[0] == cs._recording_key(app)
    assert cs._recording_key(app)[2:] == (id(app._signal_flt), 10)
    # pending wins over a stale tab in a snapshot (the session saved again)
    app._compare_setup_getter = lambda: {"rows": [["stale", 0, 1]]}
    app._compare_setup_key = app._compare_pending_setup[0]
    assert M._compare_setup_snapshot(app)["rows"] == [["A", 0.0, 5.0]]


def test_ui_rebuild_reshows_the_kept_result_without_recomputing(root, monkeypatch):
    app = _app()
    frame = _build(app, root)
    _run(app, frame, WINS[:3])
    res = app.analysis.last_comparison
    snap = M._compare_setup_snapshot(app)
    frame.destroy()
    import ecg.core.analysis as an
    monkeypatch.setattr(an, "compute_segment_stats",
                        lambda *a, **k: pytest.fail("recomputed after a UI rebuild"))
    app._compare_pending_setup = (cs._recording_key(app), snap)
    frame2 = _build(app, root)
    res2 = app.analysis.last_comparison
    assert res2.segs is res.segs and res2.comps == res.comps
    assert [r[0] for r in _rows(frame2)] == [w[0] for w in WINS[:3]]


def test_rows_survive_a_rebuild_for_new_peaks(root):
    """_refresh_compare_tab on the same recording (peaks edited): the rows
    typed stay, the comparison (of the old peaks) is not re-run."""
    app = _app()
    holder = ctk.CTkFrame(root)
    holder.pack()
    app._compare_holder = holder
    app._build_compare_segments = lambda win: M._build_compare_segments(app, win)
    app._build_group_card = lambda win: None
    app._compare_built_for = None
    app._compare_signature = lambda: M._compare_signature(app)
    M._refresh_compare_tab(app)
    _run(app, holder, WINS[:3])
    app._rpeaks_ok = app._rpeaks_ok[:-5]                 # peaks edited
    app.analysis.last_comparison = None
    M._refresh_compare_tab(app)
    holder.update()
    assert [r[0] for r in _rows(holder)] == [w[0] for w in WINS[:3]]
    assert app.analysis.last_comparison is None


def test_session_state_holds_the_setup_and_the_worker_passes_it(monkeypatch):
    from tests.ui.test_event_locked_controller import _computed, FS
    from ecg.ui import session_controller as sc_module
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
    monkeypatch.setattr(sc_module, "load_mat_signal",
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

def test_recording_key_includes_the_file_and_channel():
    """A freed array's id is reused by CPython: two files of the same length
    must still have different keys."""
    sig = np.zeros(100)
    a = SimpleNamespace(_signal_flt=sig, signal=SimpleNamespace(filepath="a.mat", channel="ECG"))
    b = SimpleNamespace(_signal_flt=sig, signal=SimpleNamespace(filepath="b.mat", channel="ECG"))
    c = SimpleNamespace(_signal_flt=sig, signal=SimpleNamespace(filepath="a.mat", channel="EMG"))
    keys = {cs._recording_key(x) for x in (a, b, c)}
    assert len(keys) == 3
    assert cs._recording_key(SimpleNamespace(_signal_flt=None)) is None


def test_opening_a_file_drops_the_previous_compare_setup():
    from unittest.mock import MagicMock
    from ecg.ui.signal_controller import SignalController
    from ecg.ui.freshness import FreshnessService
    from ecg.ui.state import AnalysisState, UIState
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


def test_a_comparison_in_flight_is_kept_by_a_snapshot(root, monkeypatch):
    """Save Session (or a UI rebuild) before the worker's result lands still
    holds the comparison, so it is re-run on restore."""
    pending = []

    class _Deferred:
        def __init__(self, target=None, daemon=None, **_kw):
            self._target = target

        def start(self):
            pending.append(self._target)
    import threading
    monkeypatch.setattr(threading, "Thread", _Deferred)
    app = _app()
    frame = _build(app, root)
    add = next(w for w in _walk(frame) if isinstance(w, ctk.CTkButton)
               and "Add segment" in w.cget("text"))
    add.invoke()
    rows = [e for e in _entries(frame) if e.cget("placeholder_text") == "Label"]
    for lbl_e, (label, lo, hi) in zip(rows, WINS[:3]):
        kids = [c for c in lbl_e.master.winfo_children() if isinstance(c, ctk.CTkEntry)]
        for k, v in zip(kids, (label, f"{lo:g}", f"{hi:g}")):
            k.delete(0, "end"); k.insert(0, v)
    next(w for w in _walk(frame) if isinstance(w, ctk.CTkButton)
         and "Compare" in w.cget("text")).invoke()
    assert pending and app.analysis.last_comparison is None      # still computing
    snap = M._compare_setup_snapshot(app)
    assert snap["ran"] and snap["run"]["rows"] == [[l, float(a), float(b)]
                                                   for l, a, b in WINS[:3]]
    # The worker lands: the run is the shown one, nothing left in flight.
    pending.pop()()
    _wait(app, frame)
    assert app.analysis.last_comparison is not None
    snap2 = M._compare_setup_snapshot(app)
    assert snap2["run"] == snap["run"]


def test_poincare_redrawn_when_the_card_width_changes_its_columns(root, monkeypatch):
    """A Poincaré drawn while the freshly built card is still narrow is
    redrawn with the right number of columns once the card has its width."""
    calls = []
    real = cs.draw_poincare_set

    def _spy(fig, segs, colors, **kw):
        calls.append(kw.get("ncols"))
        return real(fig, segs, colors, **kw)
    monkeypatch.setattr(cs, "draw_poincare_set", _spy)
    from matplotlib.figure import Figure
    # Render at once, whatever the slot's own visibility rules.
    monkeypatch.setattr(cs.CanvasSlot, "update", lambda self, fn: fn(Figure()))
    root.deiconify()
    root.geometry("300x600")
    app = _app()
    frame = _build(app, root)
    _run(app, frame, WINS)
    _view_button(frame, "Poincaré").invoke()
    frame.update()
    narrow = calls[-1]
    root.geometry("2400x900")
    for _ in range(20):
        root.update()
        time.sleep(0.02)
    assert calls[-1] != narrow and calls[-1] > narrow
