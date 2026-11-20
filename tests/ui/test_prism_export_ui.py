"""Compare segments tab, GraphPad Prism: the Results card's Prism… writes a
new file with one table per metric (and the tables it held before) or fills
a Prism template; the Group card's Prism… does the same for the group
table, without the rows its analysis left out, and per pair in All pairs
mode. The last template is remembered (never in the real home: the
preference file is pointed at tmp_path)."""
from __future__ import annotations

import tkinter as tk
from types import SimpleNamespace

import pandas as pd
import pytest

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.io import prism_template as pt
from ecg.io.group_table import append_group_row, group_row
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


HR = "Mean HR (60000/mean NN) (bpm)"


def _group_table(path) -> None:
    metrics = [("hr_mean", "Mean HR (60000/mean NN)", "bpm"), ("sdnn", "SDNN", "ms"),
               ("sd1", "Poincaré SD1", "ms"), ("sd2", "Poincaré SD2", "ms"),
               ("rr_mean", "Mean RR", "ms")]
    labels = ["Pre 1", "Post 1", "Pre 5", "Post 5"]
    for i in range(4):
        segs = [{"label": lbl, "t_lo": 60.0 * k, "t_hi": 60.0 * k + 60, "duration_s": 60.0,
                 "hr_mean": 500 + i + 10 * k, "sdnn": 3 + 0.1 * i + k, "sd1": 2 + 0.1 * k,
                 "sd2": 5 + 0.1 * i, "rr_mean": 120 - k} for k, lbl in enumerate(labels)]
        flags = [[] for _ in labels]
        if i == 2:
            flags[3] = ["few beats"]                              # M3: Post 5 to check
        append_group_row(str(path), group_row(f"M{i + 1}", f"M{i + 1}.mat", segs, metrics,
                                              flags=flags))


def test_group_prism_leaves_out_the_rows_left_out_and_pairs(root, monkeypatch, tmp_path):
    gt = tmp_path / "group.xlsx"
    _group_table(gt)
    app = SimpleNamespace(_bind_hover_tip=lambda *a, **k: None, _set_btn_enabled=_enable,
                          _group_state={"path": str(gt), "a": "Pre 5", "b": "Post 5",
                                        "exclude": True, "name_a": "", "name_b": ""})
    frame = ctk.CTkFrame(root)
    frame.pack(fill="both", expand=True)
    CompareSegmentsMixin._build_group_card(app, frame)           # loads and runs
    frame.update()
    btn = _button(frame, "Prism…")
    assert btn.cget("state") == "normal"
    btn.invoke()
    new, fill = root.menus[-1]
    out = tmp_path / "group.pzfx"
    _dialogs(monkeypatch, save=str(out))
    new()
    got = pt.read_tables(str(out))
    assert list(got)[0] == HR
    assert list(got[HR]["columns"]) == ["Pre 1", "Post 1", "Pre 5", "Post 5"]
    assert got[HR]["rows"] == ["M1", "M2", "M4"]                  # M3 left out (Pre 5 vs Post 5)
    text = out.read_text(encoding="utf-8")
    assert "M3: Post 5: few beats" in text

    # All pairs: a table per pair x metric, M3 only out of the second pair.
    mode = next(w for w in _walk(frame) if isinstance(w, ctk.CTkComboBox)
                and cs.MODE_ALL in w.cget("values"))
    mode.set(cs.MODE_ALL)
    _button(frame, "▶  Run").invoke()
    frame.update()
    _button(frame, "Prism…").invoke()
    new, fill = root.menus[-1]
    out2 = tmp_path / "pairs.pzfx"
    _dialogs(monkeypatch, save=str(out2))
    new()
    got2 = pt.read_tables(str(out2))
    t1, t2 = f"{HR} · Post 1 vs Pre 1", f"{HR} · Post 5 vs Pre 5"
    assert got2[t1]["rows"] == ["M1", "M2", "M3", "M4"]
    assert got2[t2]["rows"] == ["M1", "M2", "M4"]
    assert list(got2[t2]["columns"]) == ["Pre 5", "Post 5"]

    # Filled into a template made from that export.
    tpl = tmp_path / "pairs template.pzfx"
    raw = _as_template(out2, tpl)
    filled = tmp_path / "pairs_filled.pzfx"
    _dialogs(monkeypatch, save=str(filled), open_=str(tpl))
    fill()
    assert not root.errors
    assert filled.read_bytes().endswith(raw[raw.index(b"<Template"):])
    assert pt.read_tables(str(filled))[t2]["rows"] == ["M1", "M2", "M4"]
    df = pd.read_excel(gt, sheet_name="Group")
    assert len(df) == 4


def _labels(frame, needle):
    return [str(w.cget("text")) for w in _walk(frame) if isinstance(w, ctk.CTkLabel)
            and needle in str(w.cget("text"))]


def test_group_prism_status_has_its_own_line_and_follows_the_table_opened(
        root, monkeypatch, tmp_path):
    gt = tmp_path / "group.xlsx"
    _group_table(gt)
    app = SimpleNamespace(_bind_hover_tip=lambda *a, **k: None, _set_btn_enabled=_enable,
                          _group_state={"path": str(gt), "a": "Pre 5", "b": "Post 5",
                                        "exclude": True, "name_a": "", "name_b": ""})
    frame = ctk.CTkFrame(root)
    frame.pack(fill="both", expand=True)
    CompareSegmentsMixin._build_group_card(app, frame)
    frame.update()
    _button(frame, "Prism…").invoke()
    new, _fill = root.menus[-1]
    out = tmp_path / "g.pzfx"
    _dialogs(monkeypatch, save=str(out))
    new()
    # The analysis warning (M3 left out) is still shown, the export on its own line.
    assert _labels(frame, "left out: M3") and _labels(frame, "Prism: ")

    # Another table opened: its rows and its name, never the last run's.
    gt2 = tmp_path / "other.xlsx"
    metrics = [("hr_mean", "Mean HR (60000/mean NN)", "bpm"), ("sdnn", "SDNN", "ms")]
    for i in range(2):
        segs = [{"label": lbl, "t_lo": 60.0 * k, "t_hi": 60.0 * k + 60, "duration_s": 60.0,
                 "hr_mean": 600 + i + k, "sdnn": 4.0 + k} for k, lbl in enumerate(["Pre 5", "Post 5"])]
        append_group_row(str(gt2), group_row(f"K{i + 1}", f"K{i + 1}.mat", segs, metrics,
                                             flags=[[], []]))
    _dialogs(monkeypatch, open_=str(gt2))
    _button(frame, "📂  Open group table…").invoke()
    frame.update()
    assert not _labels(frame, "Prism: ")                          # the old result gone
    _button(frame, "Prism…").invoke()
    new, _fill = root.menus[-1]
    out2 = tmp_path / "g2.pzfx"
    _dialogs(monkeypatch, save=str(out2))
    new()
    got = pt.read_tables(str(out2))
    assert got[HR]["rows"] == ["K1", "K2"]
    assert list(got[HR]["columns"]) == ["Pre 5", "Post 5"]
    assert "other.xlsx" in out2.read_text(encoding="utf-8")


def test_prism_buttons_explain_the_template_in_a_floating_tip(root, monkeypatch, tmp_path):
    tips: "dict" = {}
    gt = tmp_path / "group.xlsx"
    _group_table(gt)
    app = SimpleNamespace(_bind_hover_tip=lambda *a, **k: None, _set_btn_enabled=_enable,
                          _bind_float_tip=lambda w, text, **k: tips.__setitem__(w, text),
                          _group_state={"path": str(gt), "a": "Pre 5", "b": "Post 5",
                                        "exclude": True, "name_a": "", "name_b": ""})
    frame = ctk.CTkFrame(root)
    frame.pack(fill="both", expand=True)
    CompareSegmentsMixin._build_group_card(app, frame)
    assert tips[_button(frame, "Prism…")].startswith(cs.PRISM_FILL)
