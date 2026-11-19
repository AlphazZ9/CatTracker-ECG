"""Compare segments tab, built for real on a stand-in app: no "Windows
around an event" card (the windows are typed, the comparison runs with no
event), the "Δ % against:" selector re-points the table, the comparison
text and the kept ComparisonResult without recomputing the segment
statistics, and the RR tachograms draw one panel per pair in Pairs mode."""
from __future__ import annotations

import time
import tkinter as tk
from types import SimpleNamespace

import numpy as np
import pytest

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.ui import compare_segments as cs
from ecg.ui.compare_segments import CompareSegmentsMixin

FS = 1000.0


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def _app(nav_pos: float = 450.0) -> SimpleNamespace:
    rng = np.random.default_rng(5)
    rr = np.r_[130 + 3 * rng.standard_normal(2500), 115 + 1.5 * rng.standard_normal(2500),
               122 + 2 * rng.standard_normal(2500)]
    rp = (np.r_[0, np.cumsum(np.round(rr))] + 500).astype(int)
    app = SimpleNamespace(
        _signal_flt=np.zeros(rp[-1] + 1000), _rpeaks_ok=rp, _fs=FS,
        _get_freq_bands=lambda: ((0.0, 0.15), (0.15, 1.5), (1.5, 5.0)),
        ent_subject=None, _bind_hover_tip=lambda *a, **k: None,
        analysis=SimpleNamespace(annotations=[], arrhythmia_events=[], last_comparison=None,
                                 last_comparison_basis=None),
        ui=SimpleNamespace(nav_pos=nav_pos),
        analysis_ctrl=SimpleNamespace(hrv_rr_bounds=lambda: (40.0, 400.0),
                                      hrv_breaks=lambda rp_: None),
        signal=SimpleNamespace(filepath="x.mat"),
        _set_btn_enabled=lambda *a, **k: None,
        _SEGMENT_TABLE_METRICS=CompareSegmentsMixin._SEGMENT_TABLE_METRICS,
        _compare_built_for=("sig",),
    )
    return app


@pytest.fixture
def tab(monkeypatch):
    # The worker hands its result back with win.after(), which Tk refuses
    # from another thread without a running mainloop: run it inline here.
    class _Inline:
        def __init__(self, target=None, daemon=None, **_kw):
            self._target = target

        def start(self):
            self._target()
    import threading
    monkeypatch.setattr(threading, "Thread", _Inline)
    try:
        root = ctk.CTk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    frame = ctk.CTkFrame(root)
    frame.pack(fill="both", expand=True)
    app = _app()
    CompareSegmentsMixin._build_compare_segments(app, frame)
    yield app, frame
    # CTk's pending after() callbacks would fire on a destroyed root.
    for after_id in root.tk.splitlist(root.tk.call("after", "info")):
        root.tk.call("after", "cancel", after_id)
    root.destroy()


def _entries(frame):
    return [w for w in _walk(frame) if isinstance(w, ctk.CTkEntry)]


def _set(entry, text: str) -> None:
    entry.delete(0, "end")
    entry.insert(0, text)


def _run(app, frame, windows) -> None:
    """Type *windows* (label, lo, hi) into the segment rows and Compare."""
    add = next(w for w in _walk(frame) if isinstance(w, ctk.CTkButton)
               and "Add segment" in w.cget("text"))
    while sum(1 for e in _entries(frame) if e.cget("placeholder_text") == "Label") < len(windows):
        add.invoke()
    rows = [e for e in _entries(frame) if e.cget("placeholder_text") == "Label"]
    for lbl_e, (label, lo, hi) in zip(rows, windows):
        # The row's entries follow its label entry: label, from, to.
        kids = [c for c in lbl_e.master.winfo_children() if isinstance(c, ctk.CTkEntry)]
        _set(kids[0], label); _set(kids[1], f"{lo:g}"); _set(kids[2], f"{hi:g}")
    run = next(w for w in _walk(frame) if isinstance(w, ctk.CTkButton)
               and "Compare" in w.cget("text"))
    app.analysis.last_comparison = None
    run.invoke()
    t0 = time.time()
    while app.analysis.last_comparison is None and time.time() - t0 < 60:
        frame.update()
        time.sleep(0.02)
    frame.update()
    assert app.analysis.last_comparison is not None, "the comparison never finished"


def _ref_combo(frame):
    return next(w for w in _walk(frame) if isinstance(w, ctk.CTkComboBox)
                and any("(reference)" in str(v) for v in w.cget("values")))


def _table_text(frame) -> str:
    tbl = next(w for w in _walk(frame) if isinstance(w, tk.Text))
    return tbl.get("1.0", "end")


def test_no_event_or_template_card(tab):
    app, frame = tab
    texts = [str(w.cget("text")) for w in _walk(frame) if isinstance(w, ctk.CTkLabel)]
    assert not any("event" in t.lower() for t in texts)
    assert not any(t.strip() in ("Template:", "Artefact margin (s):", "Window (s):")
                   for t in texts)
    buttons = [str(w.cget("text")) for w in _walk(frame) if isinstance(w, ctk.CTkButton)]
    assert "Save as template…" not in buttons
    # The segment rows' entries, the block length and the Stim at entry of
    # the configurations row (relative to the stimulus, not to an event).
    assert all(e.cget("placeholder_text") in ("Label", None, "", "e.g. 1803")
               for e in _entries(frame))
    # Typed windows over the Detection view's position: no event, no template.
    # This drives the real _run_compare / _on_done path, real quality_flags
    # included (a wrong keyword there used to break every run).
    _run(app, frame, [("S1", 300, 600), ("S2", 600, 900)])
    res = app.analysis.last_comparison
    assert len(res.flags) == 2
    assert res.event is None and res.template == "" and res.artefact_margin_s is None
    assert not any("overlaps the event" in f for fl in res.flags for f in fl)
    from ecg.io.segment_export import summary_rows
    items = [k for k, _v in summary_rows(res)]
    assert "Template" not in items and "Event (s)" not in items
    assert "Artefact margin (s)" not in items


def test_reference_selector_repoints_without_recomputing(tab):
    app, frame = tab
    _run(app, frame, [("S1", 0, 300), ("S2", 300, 600), ("S3", 600, 900)])
    res = app.analysis.last_comparison
    assert res.comps == [None, 0, 0]                 # default: against the first
    head = _table_text(frame).split("\n")[0]
    assert "S1 (ref.)" in head and head.count("Δ %") == 2
    cb = _ref_combo(frame)
    assert list(cb.cget("values")) == ["S1 (reference)", "S2 (reference)", "S3 (reference)",
                                       "Previous segment (pairwise)"]
    segs_before = res.segs
    hr_s3 = res.segs[2]["hr_mean"]

    cb._dropdown_callback("S2 (reference)")
    frame.update()
    assert app.analysis.last_comparison is res and res.segs is segs_before
    assert res.comps == [1, None, 1] and res.segs[2]["hr_mean"] == hr_s3
    assert len(res.cmps) == 2
    assert "S2 (ref.)" in _table_text(frame).split("\n")[0]
    from ecg.io.segment_export import comparison_frame
    assert list(comparison_frame(res).columns)[-2:] == ["Δ% S2→S1", "Δ% S2→S3"]
    assert app._last_seg_a is res.segs[1] and app._last_seg_b is res.segs[0]

    cb._dropdown_callback("Previous segment (pairwise)")
    frame.update()
    assert res.comps == [None, 0, 1]
    head = _table_text(frame).split("\n")[0]
    assert "Δ % vs S1" in head and "Δ % vs S2" in head
    texts = [w.cget("text") for w in _walk(frame) if isinstance(w, ctk.CTkLabel)]
    assert any("against the previous segment" in t for t in texts)

    # The choice is kept for the next run.
    _run(app, frame, [("S1", 0, 300), ("S2", 300, 600), ("S3", 600, 900)])
    assert app.analysis.last_comparison.comps == [None, 0, 1]


def test_duplicate_labels_stay_distinct_choices(tab):
    # Two segments with the same label: the selector used to keep only one
    # choice and _on_done raised StopIteration (no table drawn at all).
    app, frame = tab
    _run(app, frame, [("A", 0, 300), ("A", 300, 600), ("B", 600, 900)])
    res = app.analysis.last_comparison
    cb = _ref_combo(frame)
    vals = list(cb.cget("values"))
    assert vals[:3] == ["A: A (reference)", "B: A (reference)", "C: B (reference)"]
    assert cb.get() == "A: A (reference)"
    assert res.comps == [None, 0, 0]
    head = _table_text(frame).split("\n")[0]
    assert "A: A (ref.)" in head and "B: A" in head      # the table was drawn
    # Screen and exports name the same segments (the export tags A, B, C).
    cb._dropdown_callback("B: A (reference)")
    frame.update()
    assert res.comps == [1, None, 1]
    texts = [w.cget("text") for w in _walk(frame) if isinstance(w, ctk.CTkLabel)]
    assert any("Δ % against B: A" in t for t in texts)
    from ecg.io.segment_export import comparison_frame
    assert list(comparison_frame(res).columns)[-2:] == ["Δ% B→A", "Δ% B→C"]


def test_reference_changed_while_running_is_followed(tab):
    # The selector stays usable while the worker runs: the finished run
    # must use the reference shown, not the one read when Compare was hit.
    app, frame = tab
    wins = [("S1", 0, 300), ("S2", 300, 600), ("S3", 600, 900)]
    _run(app, frame, wins)
    cb = _ref_combo(frame)
    run = next(w for w in _walk(frame) if isinstance(w, ctk.CTkButton)
               and "Compare" in w.cget("text"))
    app.analysis.last_comparison = None
    run.invoke()            # the inline worker queues _on_done with after(0)
    cb._dropdown_callback("S3 (reference)")
    t0 = time.time()
    while app.analysis.last_comparison is None and time.time() - t0 < 60:
        frame.update()
        time.sleep(0.02)
    frame.update()
    res = app.analysis.last_comparison
    assert res.comps == [2, 2, None]
    assert cb.get() == "S3 (reference)"
    assert "S3 (ref.)" in _table_text(frame).split("\n")[0]
    assert len(res.cmps) == 2


def test_pairs_mode_draws_one_tachogram_panel_per_pair(tab, monkeypatch):
    app, frame = tab
    drawn = []
    real = cs.draw_rr_tachograms

    def _spy(fig, segs, colors, comparators=None, names=None):
        axes = real(fig, segs, colors, comparators, names)
        drawn.append(axes)
        return axes
    monkeypatch.setattr(cs, "draw_rr_tachograms", _spy)
    # Inside the ~919 s recording (7500 beats of ~122 ms).
    wins = [("Pre 1", 0, 150), ("Post 1", 150, 300), ("Pre 2", 300, 450),
            ("Post 2", 450, 600), ("Pre 3", 600, 750)]
    _run(app, frame, wins)
    assert len(drawn[-1]) == 1                      # against the first: one plot
    cb = _ref_combo(frame)
    cb._dropdown_callback("Pairs (2 vs 1, 4 vs 3, 6 vs 5)")
    frame.update()
    axes = drawn[-1]                                # the view followed the selector
    assert [a.get_title(loc="left") for a in axes] == [
        "Post 1 vs Pre 1", "Post 2 vs Pre 2", "Pre 3"]
    # The card grew to give each panel its row.
    heights = {int(w.cget("height")) for w in _walk(frame) if isinstance(w, ctk.CTkFrame)}
    assert cs._YBAR_H + 3 * cs._TACHO_ROW_H in heights
    cb._dropdown_callback("Pre 1 (reference)")
    frame.update()
    assert len(drawn[-1]) == 1
