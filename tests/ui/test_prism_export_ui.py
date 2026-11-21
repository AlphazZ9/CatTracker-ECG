"""Compare segments tab, GraphPad Prism: the Results card's Prism… writes a
new file with one table per metric (and the tables it held before) or fills
a Prism template (the group exports are the Group tab's:
tests/ui/test_group_tab.py). The last template is remembered (never in the real home: the
preference file is pointed at tmp_path)."""
from __future__ import annotations

import tkinter as tk

import pytest

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.io import prism_template as pt
from ecg.ui import compare_segments as cs
from ecg.ui.compare_segments import CompareSegmentsMixin

from tests.ui.test_compare_reference import _app, _run, _walk

BLOB = "eNrsfQtgVMW58Jyzm+wjmychgCgcHtUAAQIiBsSyiTx/ESNQAUUhkI0Ekw0kiyAqrBIeCgio\r\nUA=="


def _enable(btn, enabled: bool, kind: str = "primary") -> None:
    btn.configure(state="normal" if enabled else "disabled")


@pytest.fixture
def root(monkeypatch, tmp_path):
    class _Inline:
        def __init__(self, target=None, daemon=None, **_kw):
            self._target = target

        def start(self):
            self._target()
    import threading
    monkeypatch.setattr(threading, "Thread", _Inline)
    monkeypatch.setattr(pt, "PREFS_PATH", tmp_path / "prism_prefs.json")
    errors: "list" = []
    monkeypatch.setattr(cs.messagebox, "showerror", lambda *a, **k: errors.append(a))
    monkeypatch.setattr(cs.messagebox, "showwarning", lambda *a, **k: errors.append(a))
    menus: "list" = []
    monkeypatch.setattr(cs, "prism_menu", lambda parent, button, new, fill: menus.append((new, fill)))
    try:
        r = ctk.CTk()
    except tk.TclError:
        pytest.skip("no display")
    r.withdraw()
    r.menus, r.errors = menus, errors
    yield r
    for after_id in r.tk.splitlist(r.tk.call("after", "info")):
        r.tk.call("after", "cancel", after_id)
    r.destroy()


def _dialogs(monkeypatch, save=None, open_=None):
    monkeypatch.setattr(cs.filedialog, "asksaveasfilename", lambda *a, **k: save)
    monkeypatch.setattr(cs.filedialog, "askopenfilename", lambda *a, **k: open_)


def _button(frame, text):
    return next(w for w in _walk(frame) if isinstance(w, ctk.CTkButton)
                and w.cget("text") == text)


def _as_template(src, dst) -> bytes:
    """An exported file made a template, as Prism would save it (a blob)."""
    text = open(src, encoding="utf-8").read().replace(
        "</GraphPadPrismFile>",
        f'<Template xmlns:dt="urn:schemas-microsoft-com:datatypes" dt:dt="bin.base64">{BLOB}'
        "</Template></GraphPadPrismFile>")
    raw = text.encode("utf-8")
    with open(dst, "wb") as fh:
        fh.write(raw)
    return raw


def test_recording_prism_new_file_and_template(root, monkeypatch, tmp_path):
    app = _app()
    app._set_btn_enabled = _enable
    frame = ctk.CTkFrame(root)
    frame.pack(fill="both", expand=True)
    CompareSegmentsMixin._build_compare_segments(app, frame)
    _run(app, frame, [("Pre", 300, 600), ("Post", 600, 900)])
    _button(frame, "Prism…").invoke()
    new, fill = root.menus[-1]
    out = tmp_path / "rec.pzfx"
    _dialogs(monkeypatch, save=str(out))
    new()
    got = pt.read_tables(str(out))
    titles = list(got)
    assert titles[0] == "Comparison" and "RR Pre" in titles and "Poincaré Post" in titles
    sdnn = next(t for t in titles if t.startswith("SDNN"))
    assert list(got[sdnn]["columns"]) == ["Pre", "Post"] and len(got[sdnn]["rows"]) == 1

    tpl = tmp_path / "my graphs.pzfx"
    raw = _as_template(out, tpl)
    filled = tmp_path / "filled.pzfx"
    _run(app, frame, [("Pre", 100, 400), ("Post", 400, 700)])
    _dialogs(monkeypatch, save=str(filled), open_=str(tpl))
    fill()
    assert not root.errors
    data = filled.read_bytes()
    assert data.endswith(raw[raw.index(b"<Template"):])           # the blob as it was
    new_vals = pt.read_tables(str(filled))[sdnn]["columns"]["Post"]
    assert new_vals != got[sdnn]["columns"]["Post"]
    assert pt.load_last_template(pt.PREFS_PATH) == str(tpl)       # remembered
    status = [str(w.cget("text")) for w in _walk(frame) if isinstance(w, ctk.CTkLabel)
              and "Prism template" in str(w.cget("text"))]
    assert status and f"{len(got)} tables filled" in status[0]
    # The template itself as the output: refused, nothing written.
    _dialogs(monkeypatch, save=str(tpl), open_=str(tpl))
    calls = iter([str(tpl), None])
    monkeypatch.setattr(cs.filedialog, "asksaveasfilename", lambda *a, **k: next(calls))
    fill()
    assert open(tpl, "rb").read() == raw and root.errors
