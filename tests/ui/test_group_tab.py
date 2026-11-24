"""Group tab: opens a group table, lists the animals (group and use saved in
the table), runs the chosen analysis at once and enables its exports; Add
open recording writes the recording open (whole, time course, annotations,
events, segments) to the table."""
from __future__ import annotations

import os
import tkinter as tk
from types import SimpleNamespace as NS

import numpy as np
import pytest

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.io import db
from ecg.io import group_compare as gc
from ecg.io import group_data as gd
from ecg.ui import tab_group as tg
from ecg.ui.tab_group import TabGroupMixin, recording_lines
from tests.io.test_group_compare import _table


@pytest.fixture
def root(monkeypatch, tmp_path):
    errors: "list" = []
    monkeypatch.setattr(tg.messagebox, "showerror", lambda *a, **k: errors.append(a))
    try:
        r = ctk.CTk()
    except tk.TclError:
        pytest.skip("no display")
    r.withdraw()
    r.errors = errors
    yield r
    for after_id in r.tk.splitlist(r.tk.call("after", "info")):
        r.tk.call("after", "cancel", after_id)
    r.destroy()


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def _results_text(frame) -> str:
    return next(w for w in _walk(frame) if isinstance(w, tk.Text)).get("1.0", "end")


def _button(frame, text):
    return next(w for w in _walk(frame) if isinstance(w, ctk.CTkButton)
                and w.cget("text") == text)


def _page(root, state):
    app = NS(_bind_hover_tip=lambda *a, **k: None, _grouptab_state=state)
    frame = ctk.CTkFrame(root)
    frame.pack(fill="both", expand=True)
    TabGroupMixin._build_group_page(app, frame)
    frame.update()
    return app, frame


@pytest.mark.parametrize("analysis, expect", [
    (gc.AN_TIME, "RM ANOVA Group × time"),
    (gc.AN_SEGMENTS, "Δ Mean HR (bpm) (Post − Pre)"),
    (gc.AN_WHOLE, "Mean HR (60000/mean NN) (bpm)"),
    (gc.AN_EVENTS, "Δ rCSI (post − pre)"),
])
def test_each_analysis_runs_on_open(root, tmp_path, analysis, expect):
    p = _table(tmp_path)
    _app, frame = _page(root, {"path": p, "analysis": analysis, "by": gc.BY_GROUP,
                               "stim": "Stim", "from": "-4", "to": "8", "w0": "0", "w1": "2"})
    text = _results_text(frame)
    assert expect in text and "Control" in text and "CNO" in text, text
    assert _button(frame, "Figure…").cget("state") == "normal"
    assert not root.errors


def test_group_and_use_are_saved_in_the_table(root, tmp_path):
    p = _table(tmp_path, n=4)
    app, frame = _page(root, {"path": p, "analysis": gc.AN_WHOLE, "by": gc.BY_GROUP})
    combos = [w for w in _walk(frame) if isinstance(w, ctk.CTkComboBox)
              and w.cget("values") and "Control" in w.cget("values")]
    combos[0].set("Vehicle")
    combos[0]._command("Vehicle")
    checks = [w for w in _walk(frame) if isinstance(w, ctk.CTkCheckBox) and w.cget("text") == ""]
    checks[1].deselect()
    checks[1]._command()
    an = gd.animals_frame(gd.read_sheets(p))
    assert an.iloc[0]["group"] == "Vehicle" and not an.iloc[1]["use"]
    assert "Vehicle" in _results_text(frame)


def test_no_table_no_result(root, tmp_path):
    _app, frame = _page(root, {"path": str(tmp_path / "missing.xlsx")})
    assert _button(frame, "Figure…").cget("state") == "disabled"


def test_a_time_course_without_annotation_says_why(root, tmp_path):
    p = _table(tmp_path, n=4)
    sh = gd.read_sheets(p)
    del sh[gd.SHEET_ANNOT]
    gd.write_sheets(p, sh)
    _app, frame = _page(root, {"path": p, "analysis": gc.AN_TIME})
    assert "annotation" in _results_text(frame)


def _open_recording(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    rr = 120 + rng.normal(0, 2, 3000)
    t = np.cumsum(rr) / 1000
    results = {"rr_ms": rr, "rr_t_s": t, "rr_valid": np.ones(len(rr), bool), "hr": {}}
    rows = [NS(label="Mean HR (60000/mean NN)", unit="bpm", value=500.0)]
    monkeypatch.setattr(db, "list_event_rows", lambda **k: [
        {"label": "Stim", "onset_s": 100.0, "valid": True,
         "metrics": {"rCSI": {"pre": 1.0, "post": 2.0, "delta": 1.0}}}])
    app = NS(signal=NS(filepath=str(tmp_path / "Stim BNST - 1146 F.smr")),
             ent_subject=NS(get=lambda: "1146"), ent_sex=NS(get=lambda: "f"),
             analysis=NS(results=results, annotations=[{"t_start": 100.0, "label": "Stim"}]),
             export_ctrl=NS(metric_table=lambda: NS(rows=rows)),
             _current_comparison=lambda: None)
    app._group_lines_of_open_recording = (
        lambda comparison=None: TabGroupMixin._group_lines_of_open_recording(app, comparison))
    return app


def test_add_open_recording(tmp_path, monkeypatch):
    app = _open_recording(tmp_path, monkeypatch)
    p = str(tmp_path / "g.xlsx")
    n, replaced, what = TabGroupMixin._add_open_recording_to_group_table(app, p)
    assert (n, replaced) == (1, False)
    assert what == "1146: whole recording, time course, events, annotations"
    sh = gd.read_sheets(p)
    an = gd.animals_frame(sh)
    assert an.iloc[0]["sex"] == "F" and an.iloc[0]["file"] == "Stim BNST - 1146 F.smr"
    assert sh[gd.SHEET_WHOLE].iloc[0]["Mean HR (60000/mean NN) (bpm)"] == 500.0
    assert sh[gd.SHEET_TIME]["n_rr"].sum() == 3000
    # Nothing computed: refused, nothing written.
    app.analysis.results = None
    app.analysis.annotations = []
    monkeypatch.setattr(db, "list_event_rows", lambda **k: [])
    with pytest.raises(ValueError):
        TabGroupMixin._add_open_recording_to_group_table(app, str(tmp_path / "h.xlsx"))
    assert not (tmp_path / "h.xlsx").exists()


def test_recording_lines_with_a_comparison():
    segs = [{"label": "Pre", "t_lo": 0.0, "t_hi": 60.0, "hr_mean": 500.0},
            {"label": "Post", "t_lo": 60.0, "t_hi": 120.0, "hr_mean": 520.0}]
    cmp_ = NS(subject="7", filepath="/x/7.smr", segs=segs, metrics=[("hr_mean", "Mean HR", "bpm")],
              template="", flags=[[], []])
    lines = recording_lines(None, [], [], [], cmp_, "M")
    seg = lines[gd.SHEET_SEGMENTS]
    assert "subject" not in seg.columns and seg.iloc[0]["Mean HR (bpm) Post"] == 520.0
    # Not computed this time: left out, so an earlier add's lines stay.
    assert gd.SHEET_TIME not in lines and gd.SHEET_WHOLE not in lines


def test_nothing_is_opened_by_itself(root, tmp_path):
    _table(tmp_path, n=4)
    app = NS(_bind_hover_tip=lambda *a, **k: None)          # a new session
    frame = ctk.CTkFrame(root)
    TabGroupMixin._build_group_page(app, frame)
    assert app._grouptab_state["path"] == ""
    assert _button(frame, "✕  Close").cget("state") == "disabled"


def test_an_animal_is_removed_and_the_table_closed(root, tmp_path, monkeypatch):
    p = _table(tmp_path, n=4)
    monkeypatch.setattr(tg.messagebox, "askyesno", lambda *a, **k: True)
    app, frame = _page(root, {"path": p, "analysis": gc.AN_WHOLE, "by": gc.BY_GROUP})
    before = gd.animals_frame(gd.read_sheets(p))
    first = (before.iloc[0]["subject"], before.iloc[0]["file"])
    rm = [w for w in _walk(frame) if isinstance(w, ctk.CTkButton) and w.cget("text") == "✕"]
    assert len(rm) == len(before)
    rm[0].invoke()
    sh = gd.read_sheets(p)
    after = gd.animals_frame(sh)
    assert len(after) == len(before) - 1
    assert first not in set(zip(after["subject"], after["file"]))
    for name in (gd.SHEET_WHOLE, gd.SHEET_TIME):
        assert first[0] not in set(sh[name]["subject"].astype(str))
    _button(frame, "✕  Close").invoke()
    assert app._grouptab_state["path"] == "" and os.path.exists(p)
    assert _button(frame, "Figure…").cget("state") == "disabled"


def test_the_comparison_subject_keys_the_row(tmp_path, monkeypatch):
    """Audit: Add to group table keyed the row on the Subject entry, not on the
    subject the comparison ran under: the same file came in twice."""
    app = _open_recording(tmp_path, monkeypatch)
    app.ent_subject = NS(get=lambda: "")
    segs = [{"label": "Pre", "t_lo": 0.0, "t_hi": 60.0, "hr_mean": 500.0}]
    cmp_ = NS(subject="243", filepath=app.signal.filepath, segs=segs,
              metrics=[("hr_mean", "Mean HR", "bpm")], template="", flags=[[]])
    p = str(tmp_path / "g.xlsx")
    TabGroupMixin._add_open_recording_to_group_table(app, p, comparison=cmp_)
    assert list(gd.animals_frame(gd.read_sheets(p))["subject"]) == ["243"]


def test_add_open_asks_where_when_the_remembered_table_did_not_open(root, tmp_path, monkeypatch):
    """Audit: a remembered table that no longer opens is not written to silently."""
    gone = tmp_path / "moved.xlsx"
    asked = []
    monkeypatch.setattr(tg.filedialog, "asksaveasfilename",
                        lambda *a, **k: asked.append(k) or "")
    app, frame = _page(root, {"path": str(gone)})
    app._add_open_recording_to_group_table = lambda *a, **k: pytest.fail("written")
    _button(frame, "➕  Add open recording…").invoke()
    assert asked and not gone.exists()


def test_group_prism_leaves_out_the_rows_left_out_and_pairs(root, tmp_path, monkeypatch):
    """Moved from the Compare segments group card: Prism… writes the tables
    without the rows the analysis left out, one per pair and metric in All
    pairs (the titles of the former card, so its templates still fill)."""
    from ecg.io import prism_template as pt
    from ecg.ui import compare_segments as cs
    from tests.io.test_group_compare import SEG_HR, _segments_table
    monkeypatch.setattr(pt, "PREFS_PATH", tmp_path / "prism.json")
    menus = []
    monkeypatch.setattr(cs, "prism_menu", lambda parent, button, new, fill: menus.append(new))
    path = tmp_path / "g.xlsx"
    _segments_table(path)
    _app, frame = _page(root, {"path": str(path), "analysis": gc.AN_SEGMENTS,
                                "by": gc.BY_NONE, "pairs": gc.PAIRS_ALL})
    out = tmp_path / "pairs.pzfx"
    monkeypatch.setattr(tg.filedialog, "asksaveasfilename", lambda *a, **k: str(out))
    _button(frame, "Prism… (new file / template)").invoke()
    menus[-1]()
    got = pt.read_tables(str(out))
    t1, t2 = f"{SEG_HR} · Post 1 vs Pre 1", f"{SEG_HR} · Post 5 vs Pre 5"
    assert got[t1]["rows"] == ["M1", "M2", "M3", "M4"]
    assert got[t2]["rows"] == ["M1", "M2", "M4"]               # M3: Post 5 few beats
    assert list(got[t2]["columns"]) == ["Pre 5", "Post 5"]
    assert "few beats" in out.read_text(encoding="utf-8")       # in the project notes


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    """A throw-away registry and session folder (never the user's profile)."""
    from ecg.io import session as sess
    d = tmp_path / "sessions"
    monkeypatch.setattr(db, "SESSION_DIR", d)
    monkeypatch.setattr(db, "_DB_PATH", d / "ecg_registry.db")
    monkeypatch.setattr(db, "_schema_ready", False)
    monkeypatch.setattr(db, "_schema_retry_at", 0.0)
    monkeypatch.setattr(sess, "SESSION_DIR", d)
    return d


def _saved(tmp_path, name, subject, sex, project, analysed=True):
    from ecg.io.loaders import _serialise_results
    from ecg.io.session import _file_fingerprint, save_session
    fp = tmp_path / name
    fp.write_bytes(b"ecg")
    rng = np.random.default_rng(len(name))
    rr = 120 + rng.normal(0, 2, 2000)
    results = {"rr_ms": rr, "rr_t_s": np.cumsum(rr) / 1000,
               "rr_valid": np.ones(len(rr), bool), "hr": {}}
    state = {"recording_fields": {"subject": subject, "sex": sex, "project_name": project},
             "annotations": [{"t_start": 100.0, "t_end": 101.0, "label": "Stim"}]}
    if analysed:
        state["results"] = _serialise_results(results)
    sp = save_session(str(fp), state)
    db.upsert_recording(str(fp), _file_fingerprint(str(fp)), str(sp), project_name=project)
    return str(fp)


def test_import_a_project_from_its_saved_sessions(tmp_path, sessions):
    _saved(tmp_path, "a 1146.smr", "1146", "F", "BNST")
    _saved(tmp_path, "b 206.smr", "206", "m", "BNST")
    _saved(tmp_path, "c 300.smr", "300", "F", "BNST", analysed=False)
    _saved(tmp_path, "d 400.smr", "400", "F", "Other")
    assert set(db.project_names()) == {"BNST", "Other"}
    p = str(tmp_path / "g.xlsx")
    n, added, skipped = tg.import_project(p, "BNST")
    assert (n, sorted(added)) == (2, ["1146", "206"])
    assert len(skipped) == 1 and "c 300.smr" in skipped[0] and "Analyze" in skipped[0]
    sh = gd.read_sheets(p)
    an = gd.animals_frame(sh).set_index("subject")
    assert an.loc["206", "sex"] == "M" and an.loc["1146", "file"] == "a 1146.smr"
    assert an.loc["1146", "has_time"] and an.loc["1146", "has_whole"]
    assert len(sh[gd.SHEET_ANNOT]) == 2
    # Imported again: replaced, never doubled.
    n, added, _ = tg.import_project(p, "BNST")
    assert n == 2


def test_a_group_from_saved_sessions_is_held_in_memory(root, tmp_path, sessions, monkeypatch):
    _saved(tmp_path, "a 1146.smr", "1146", "F", "BNST")
    _saved(tmp_path, "b 206.smr", "206", "m", "BNST")
    monkeypatch.setattr(tg, "ask_project", lambda parent, names: "BNST")
    app, frame = _page(root, {"analysis": gc.AN_WHOLE, "by": gc.BY_SEX})
    _button(frame, "⤓  From saved sessions…").invoke()
    st = app._grouptab_state
    assert st["path"] == "" and st["name"] == "BNST"
    assert not list(tmp_path.glob("*.xlsx"))                 # no file written
    labels = [str(w.cget("text")) for w in _walk(frame) if isinstance(w, ctk.CTkLabel)]
    assert any("BNST (not saved)" in t and "2 animals" in t for t in labels)
    # A group set in memory, then Save as Excel… writes it.
    combos = [w for w in _walk(frame) if isinstance(w, ctk.CTkComboBox) and w.cget("width") == 130]
    combos[0].set("CNO")
    combos[0]._command("CNO")
    out = tmp_path / "BNST.xlsx"
    monkeypatch.setattr(tg.filedialog, "asksaveasfilename", lambda *a, **k: str(out))
    _button(frame, "💾  Save as Excel…").invoke()
    assert st["path"] == str(out)
    an = gd.animals_frame(gd.read_sheets(str(out)))
    assert len(an) == 2 and "CNO" in set(an["group"])


def _notes(frame) -> str:
    return "\n".join(w.cget("text") for w in _walk(frame) if isinstance(w, ctk.CTkLabel)
                     and "⚠" in str(w.cget("text")))


@pytest.mark.parametrize("typed, expect", [
    ("abc", "not a number"), ("0", "at least 5 s"), ("-30", "at least 5 s"),
    ("3", "at least 5 s")])
def test_an_invalid_bin_size_is_refused_with_a_message(root, tmp_path, typed, expect):
    p = _table(tmp_path, n=4)
    _app, frame = _page(root, {"path": p, "analysis": gc.AN_TIME, "by": gc.BY_GROUP,
                               "stim": "Stim", "bin": typed, "from": "-4", "to": "8",
                               "w0": "0", "w1": "2"})
    text = _results_text(frame)
    assert "Bins (s)" in text and expect in text, text
    assert _button(frame, "Figure…").cget("state") == "disabled"


def test_a_bin_size_off_the_5_s_grid_is_rounded_and_said(root, tmp_path):
    p = _table(tmp_path, n=4)
    _app, frame = _page(root, {"path": p, "analysis": gc.AN_TIME, "by": gc.BY_GROUP,
                               "stim": "Stim", "bin": "32", "from": "-4", "to": "8",
                               "w0": "0", "w1": "2"})
    assert "RM ANOVA" in _results_text(frame)
    assert "rounded to 30 s" in _notes(frame)


def test_recording_lines_lay_the_time_course_out_on_each_annotation():
    rng = np.random.default_rng(0)
    rr = 120 + rng.normal(0, 2, 3000)
    t = np.cumsum(rr) / 1000
    results = {"rr_ms": rr, "rr_t_s": t, "rr_valid": np.ones(len(rr), bool), "hr": {}}
    lines = recording_lines(results, [], [{"t_start": 103.37, "label": "Stim"},
                                          {"t_start": 200.0, "label": "Other"}], [])
    tc = lines[gd.SHEET_TIME]
    # 200 s is on the recording-start grid: two grids, each with every beat.
    assert sorted(set(tc["anchor_s"])) == [0.0, 103.37]
    assert (tc.groupby("anchor_s")["n_rr"].sum() == 3000).all()



def test_recording_lines_grid_named_annotations_and_clear_deleted_ones(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    rr = 120 + rng.normal(0, 2, 3000)
    t = np.cumsum(rr) / 1000
    results = {"rr_ms": rr, "rr_t_s": t, "rr_valid": np.ones(len(rr), bool), "hr": {}}
    # An unnamed annotation cannot be the Time 0: no grid of its own.
    lines = recording_lines(results, [], [{"t_start": 103.37, "label": "Stim"},
                                          {"t_start": 151.11, "label": "  "}], [])
    assert sorted(set(lines[gd.SHEET_TIME]["anchor_s"])) == [0.0, 103.37]
    # A new time course with no annotation any more writes an empty
    # Annotations frame: re-adding drops the deleted Time 0.
    none = recording_lines(results, [], [], [])
    assert gd.SHEET_ANNOT in none and not len(none[gd.SHEET_ANNOT])
    p = str(tmp_path / "g.xlsx")
    gd.add_recording(p, "201", "M", "201.mat", lines)
    gd.add_recording(p, "201", "M", "201.mat", none)
    sh = gd.read_sheets(p)
    an = sh.get(gd.SHEET_ANNOT)
    assert an is None or not len(an)
    # Without results (Compare segments only) the annotations are kept.
    assert gd.SHEET_ANNOT not in recording_lines(None, [], [], [])
    app = _open_recording(tmp_path, monkeypatch)
    app.analysis.annotations = []
    _n, _rep, what = TabGroupMixin._add_open_recording_to_group_table(
        app, str(tmp_path / "h.xlsx"))
    assert "annotations" not in what


def _labels(frame, start):
    return [w.cget("text") for w in _walk(frame) if isinstance(w, ctk.CTkLabel)
            and str(w.cget("text")).startswith(start)]


def test_an_empty_time_0_asks_to_choose_it(root, tmp_path):
    p = _table(tmp_path, n=4)
    _app, frame = _page(root, {"path": p, "analysis": gc.AN_TIME, "by": gc.BY_GROUP,
                               "stim": "Stim", "from": "-4", "to": "8", "w0": "0", "w1": "2"})
    cb = next(w for w in _walk(frame) if isinstance(w, ctk.CTkComboBox)
              and "Stim" in w.cget("values"))
    cb.set("")
    cb._command("")
    text = _results_text(frame)
    assert "Choose the Time 0 annotation" in text and "no annotation" not in text.lower()


def test_a_status_message_is_cleared_by_the_next_run(root, tmp_path):
    p = _table(tmp_path, n=4)
    _app, frame = _page(root, {"path": p, "analysis": gc.AN_WHOLE, "by": gc.BY_GROUP})
    combos = [w for w in _walk(frame) if isinstance(w, ctk.CTkComboBox)
              and w.cget("values") and "Control" in w.cget("values")]
    combos[0].set("Vehicle")
    combos[0]._command("Vehicle")
    # The save runs the analysis: its message stays through that run...
    assert any("group saved" in t for t in _labels(frame, "✓"))
    seg = next(w for w in _walk(frame) if isinstance(w, ctk.CTkSegmentedButton)
               and gc.BY_NONE in w._value_list)
    seg.set(gc.BY_NONE)
    seg._command(gc.BY_NONE)
    # ...and is gone after the next one.
    assert not any("group saved" in t for t in _labels(frame, "✓"))
    assert gc.ALL_LABEL in _results_text(frame)



def test_the_added_message_is_cleared_by_the_next_user_run(root, tmp_path, monkeypatch):
    rec = _open_recording(tmp_path, monkeypatch)
    p = str(tmp_path / "g.xlsx")
    monkeypatch.setattr(tg.filedialog, "asksaveasfilename", lambda *a, **k: p)
    app, frame = _page(root, {"path": "", "analysis": gc.AN_WHOLE, "by": gc.BY_NONE})
    app._add_open_recording_to_group_table = (
        lambda path, **k: TabGroupMixin._add_open_recording_to_group_table(rec, path, **k))
    _button(frame, "➕  Add open recording…").invoke()
    assert any("added to g.xlsx" in t for t in _labels(frame, "✓"))
    seg = next(w for w in _walk(frame) if isinstance(w, ctk.CTkSegmentedButton)
               and gc.BY_NONE in w._value_list)
    seg.set(gc.BY_SEX)
    seg._command(gc.BY_SEX)
    # The add ran its own analysis already: the user's first run clears it.
    assert not any("added to" in t for t in _labels(frame, "✓"))
