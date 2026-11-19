"""Compare segments tab, configurations row: Save as… stores the rows as
offsets from Stim at, Apply places them at another stim time (clamped to
the recording, flagged), Delete asks first; the Stim marker picker lists a
Spike2 file's markers and fills Stim at, and is absent without markers."""
from __future__ import annotations

import tkinter as tk

import pytest

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.core import segment_plan as sp
from ecg.ui import compare_segments as cs
from ecg.ui.compare_segments import CompareSegmentsMixin

from tests.ui.test_compare_reference import _app, _entries, _run, _set, _walk

ROWS = [("Pre Stim 1min", 340, 400), ("Post Stim 1min", 400, 460),
        ("Pre Stim 5min", 100, 400), ("Post Stim 5min", 400, 700)]


@pytest.fixture
def root(monkeypatch, tmp_path):
    class _Inline:
        def __init__(self, target=None, daemon=None, **_kw):
            self._target = target

        def start(self):
            self._target()
    import threading
    monkeypatch.setattr(threading, "Thread", _Inline)
    monkeypatch.setattr(sp, "TEMPLATES_PATH", tmp_path / "templates.json")
    try:
        r = ctk.CTk()
    except tk.TclError:
        pytest.skip("no display")
    r.withdraw()
    yield r
    for after_id in r.tk.splitlist(r.tk.call("after", "info")):
        r.tk.call("after", "cancel", after_id)
    r.destroy()


def _build(root, markers=None):
    app = _app()
    if markers is not None:
        app.signal.markers = markers
    frame = ctk.CTkFrame(root)
    frame.pack(fill="both", expand=True)
    CompareSegmentsMixin._build_compare_segments(app, frame)
    frame.update()
    return app, frame


def _button(frame, text):
    return next(w for w in _walk(frame) if isinstance(w, ctk.CTkButton)
                and w.cget("text") == text)


def _stim(frame):
    return next(e for e in _entries(frame) if e.cget("placeholder_text") == "e.g. 1803")


def _cfg_combo(frame):
    return next(w for w in _walk(frame) if isinstance(w, ctk.CTkComboBox)
                and getattr(w._command, "__name__", "") != "_on_ref_pick"
                and getattr(w._command, "__name__", "") != "_on_marker_pick")


def _marker_combo(frame):
    return next((w for w in _walk(frame) if isinstance(w, ctk.CTkComboBox)
                 and getattr(w._command, "__name__", "") == "_on_marker_pick"), None)


def _rows(frame):
    out = []
    for lbl_e in (e for e in _entries(frame) if e.cget("placeholder_text") == "Label"):
        kids = [c for c in lbl_e.master.winfo_children() if isinstance(c, ctk.CTkEntry)]
        out.append((kids[0].get(), float(kids[1].get()), float(kids[2].get())))
    return out


def _status(frame):
    lbl = _button(frame, "Delete").master.winfo_children()[-1]
    return str(lbl.cget("text"))


def _type_rows(app, frame, rows):
    _run(app, frame, rows)          # types the rows (and runs once: harmless)


def test_save_then_apply_at_another_stim(root, monkeypatch):
    app, frame = _build(root)
    assert _cfg_combo(frame).get() == cs._NO_CONFIG
    assert _button(frame, "Apply").cget("state") == "disabled"
    _type_rows(app, frame, ROWS)
    _set(_stim(frame), "400")
    monkeypatch.setattr(cs, "ask_config_name", lambda *_a, **_k: "Stim 1-5")
    _button(frame, "Save as…").invoke()
    assert "saved" in _status(frame)
    (t,) = sp.user_templates(sp.TEMPLATES_PATH)
    assert [(w.lo, w.hi) for w in t.windows] == [(-60, 0), (0, 60), (-300, 0), (0, 300)]
    assert _cfg_combo(frame).get() == "Stim 1-5"
    assert _cfg_combo(frame).cget("values") == ["Stim 1-5"]   # built-ins hidden
    # Rows cleared to one, another stim time, Apply.
    _set(_stim(frame), "500")
    _button(frame, "Apply").invoke()
    frame.update()
    assert _rows(frame) == [("Pre Stim 1min", 440, 500), ("Post Stim 1min", 500, 560),
                            ("Pre Stim 5min", 200, 500), ("Post Stim 5min", 500, 800)]
    assert "applied at 500" in _status(frame)
    # The applied rows run.
    run = _button(frame, "▶  Compare")
    app.analysis.last_comparison = None
    run.invoke()
    frame.update()
    res = app.analysis.last_comparison
    assert [(s["t_lo"], s["t_hi"]) for s in res.segs][3] == (500, 800)


def test_apply_clamps_and_flags(root, monkeypatch):
    app, frame = _build(root)
    sp.save_template(sp.config_from_rows("c", ROWS, 400), sp.TEMPLATES_PATH)
    _button(frame, "Delete").master  # built
    # A new tab picks up the saved configuration.
    app, frame = _build(root)
    assert _cfg_combo(frame).get() == "c"
    _set(_stim(frame), "200")
    _button(frame, "Apply").invoke()
    rows = _rows(frame)
    assert rows[2] == ("Pre Stim 5min", 0, 200)                 # cut at 0
    assert "cut to fit: Pre Stim 5min" in _status(frame)
    dur = len(app._signal_flt) / app._fs
    _set(_stim(frame), f"{dur + 10:g}")
    _button(frame, "Apply").invoke()
    assert "outside: Post Stim 1min, Post Stim 5min" in _status(frame)


def test_apply_needs_stim_and_save_offers_first_post(root, monkeypatch):
    app, frame = _build(root)
    _type_rows(app, frame, ROWS)
    asked = []
    monkeypatch.setattr(cs.messagebox, "askyesno",
                        lambda *a, **k: asked.append(a[1]) or True)
    monkeypatch.setattr(cs, "ask_config_name", lambda *_a, **_k: "auto")
    _button(frame, "Save as…").invoke()
    assert "400 s, the start of 'Post Stim 1min'" in asked[0]
    assert _stim(frame).get() == "400"
    assert sp.user_templates(sp.TEMPLATES_PATH)[0].windows[0].lo == -60
    _set(_stim(frame), "")
    _button(frame, "Apply").invoke()
    assert "Type the stim time" in _status(frame)


def test_delete_asks_first(root, monkeypatch):
    sp.save_template(sp.config_from_rows("A", ROWS[:2], 400), sp.TEMPLATES_PATH)
    sp.save_template(sp.config_from_rows("B", ROWS[2:], 400), sp.TEMPLATES_PATH)
    app, frame = _build(root)
    _cfg_combo(frame).set("A")
    monkeypatch.setattr(cs.messagebox, "askyesno", lambda *a, **k: False)
    _button(frame, "Delete").invoke()
    assert len(sp.user_templates(sp.TEMPLATES_PATH)) == 2       # declined
    monkeypatch.setattr(cs.messagebox, "askyesno", lambda *a, **k: True)
    _button(frame, "Delete").invoke()
    assert [t.name for t in sp.user_templates(sp.TEMPLATES_PATH)] == ["B"]
    assert _cfg_combo(frame).cget("values") == ["B"]


def test_marker_picker_fills_stim_at(root):
    app, frame = _build(root, markers={
        "STIM SUB": [300.18 + 0.02 * k for k in range(50)], "STIM COL": [],
        "Keyboard": [198.0, 396.0]})
    cb = _marker_combo(frame)
    assert cb is not None and cb.get() == cs._MARKER_PROMPT
    vals = list(cb.cget("values"))
    assert vals == ["STIM SUB  300.2 s  ×50", "Keyboard  198.0 s",
                    "Keyboard  396.0 s"]
    cb._dropdown_callback(vals[0])
    assert _stim(frame).get() == "300.18"


def test_no_marker_picker_without_markers(root):
    _app_, frame = _build(root)                     # a MATLAB file: no markers
    assert _marker_combo(frame) is None
    _app_, frame = _build(root, markers={"STIM COL": []})
    assert _marker_combo(frame) is None


def test_stim_at_is_kept_by_the_setup(root):
    app, frame = _build(root)
    _set(_stim(frame), "1803")
    snap = app._compare_setup_getter()
    assert snap["stim"] == "1803"
    assert cs.normalize_compare_setup(snap)["stim"] == "1803"
    assert cs.normalize_compare_setup({"rows": [["A", 0, 1]]})["stim"] == ""


def test_long_recording_keeps_millisecond_stim(root, monkeypatch):
    """A marker at 12345.678 s (a 3.5 h telemetry recording) fills Stim at
    with every digit, and Apply places the rows at 12345.678 + offsets:
    ``:g`` would have shown 12345.7 and shifted every window by 22 ms."""
    import numpy as np
    sp.save_template(sp.config_from_rows("c", ROWS, 400), sp.TEMPLATES_PATH)
    app = _app()
    app._signal_flt = np.zeros(int(14000 * 1000), dtype=np.int8)   # 14 000 s
    app.signal.markers = {"STIM SUB": [12345.678]}
    frame = ctk.CTkFrame(root)
    frame.pack(fill="both", expand=True)
    CompareSegmentsMixin._build_compare_segments(app, frame)
    frame.update()
    cb = _marker_combo(frame)
    cb._dropdown_callback(list(cb.cget("values"))[0])
    assert _stim(frame).get() == "12345.678"
    _cfg_combo(frame).set("c")
    _button(frame, "Apply").invoke()
    frame.update()
    s = 12345.678
    assert _rows(frame) == [("Pre Stim 1min", s - 60, s), ("Post Stim 1min", s, s + 60),
                            ("Pre Stim 5min", s - 300, s), ("Post Stim 5min", s, s + 300)]
    assert "applied at 12345.678 s" in _status(frame)


def test_fmt_s():
    assert cs._fmt_s(1808.183) == "1808.183"
    assert cs._fmt_s(2000.0) == "2000"
    assert cs._fmt_s(0.5) == "0.5"
    assert cs._fmt_s(123456.7891) == "123456.789"
    assert cs._fmt_s(-0.0001) == "0"
    assert cs._fmt_s(12945.678000000001) == "12945.678"
