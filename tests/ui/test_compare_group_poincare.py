"""Compare segments, GROUP — ALL ANIMALS card: the group Poincaré of one
pair A → B for the group table the Group tab uses (its pair by default, else
the comparison's reference vs the first segment compared with it), the pair
chooser, the unticked animals and the rows to check left out, a clear
message when nothing can be drawn, the table read only when the tab is shown,
and the choice kept through a UI rebuild (theme toggle)."""
from __future__ import annotations

import tkinter as tk
from types import SimpleNamespace as NS

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.io import group_compare as gc
from ecg.io import group_data as gd
from ecg.ui import compare_segments as cs
from ecg.ui import tab_group as tg
from ecg.ui.compare_segments import CompareSegmentsMixin
from ecg.ui.tab_keys import TAB_COMPARE, TAB_GROUP

TAGS = ("Pre", "Post", "Late")
LABELS = {"Pre": "Baseline", "Post": "Stim", "Late": "Late"}


def _table(tmp_path, n=4, qc=None):
    """n animals, three segments (Pre, Post, Late): mean RR, SD1, SD2 and HR
    of each; *qc*: {animal index: quality alert of its Post}."""
    p = str(tmp_path / "group.xlsx")
    rng = np.random.default_rng(3)
    for i in range(n):
        seg = {}
        for k, tag in enumerate(TAGS):
            rr = 120.0 - 8 * k + rng.normal(0, 2)
            seg.update({f"start (s) {tag}": 100.0 * k, f"end (s) {tag}": 100.0 * k + 60,
                        f"label {tag}": LABELS[tag], f"Mean RR (ms) {tag}": rr,
                        f"Poincaré SD1 (ms) {tag}": 3.0 + k + rng.normal(0, 0.2),
                        f"Poincaré SD2 (ms) {tag}": 8.0 + k + rng.normal(0, 0.3),
                        f"Mean HR (60000/mean NN) (bpm) {tag}": 60000.0 / rr,
                        f"QC {tag}": (qc or {}).get(i, "ok") if tag == "Post" else "ok"})
        gd.add_recording(p, f"M{i + 1}", "M", f"M{i + 1}.mat",
                         {gd.SHEET_SEGMENTS: pd.DataFrame([seg])})
    return p


class _App(CompareSegmentsMixin):
    def __init__(self, state=None, comparison=None, tabs=None):
        self._grouptab_state = state
        self._bind_hover_tip = lambda *a, **k: None
        self._slots = {}
        self.tabs = tabs
        self._comparison = comparison

    def _current_comparison(self):
        return self._comparison


class _Tabs:
    def __init__(self, shown):
        self.shown = shown

    def get(self):
        return self.shown

    def set(self, name):
        self.shown = name


@pytest.fixture
def root(monkeypatch, tmp_path):
    monkeypatch.setattr(tg, "PREFS_PATH", tmp_path / "group_prefs.json")
    try:
        r = ctk.CTk()
    except tk.TclError:
        pytest.skip("no display")
    r.withdraw()
    yield r
    for after_id in r.tk.splitlist(r.tk.call("after", "info")):
        r.tk.call("after", "cancel", after_id)
    r.destroy()


def _card(root, app):
    frame = ctk.CTkFrame(root)
    frame.pack(fill="both", expand=True)
    app._build_group_pointer(frame)
    frame.update()
    return frame


def _status(app) -> str:
    return app._cmp_group_ui["lbl"].cget("text")


def _pair(app) -> "tuple[str, str]":
    ui = app._cmp_group_ui
    return ui["tags"].get(ui["a"].get()), ui["tags"].get(ui["b"].get())


def _titles(app) -> "list[str]":
    return [ax.get_title() for ax in app._cmp_group_ui["slot"].fig.axes]


def _choose(app, a, b):
    ui = app._cmp_group_ui
    shown = {t: v for v, t in ui["tags"].items()}
    ui["a"].set(shown[a])
    ui["b"].set(shown[b])
    ui["b"]._command(shown[b])


def test_draws_the_group_tabs_pair(root, tmp_path):
    p = _table(tmp_path)
    app = _App({"path": p, "a": "Pre", "b": "Late"})
    _card(root, app)
    assert _pair(app) == ("Pre", "Late")
    # No group typed in Animals (By group, the Group tab's default): every
    # ticked animal, and the status line says why.
    assert _status(app) == "n = 4 animals · no group set: all ticked animals · table: group.xlsx"
    titles = _titles(app)
    assert titles[0].startswith("Baseline") and titles[1].startswith("Late"), titles
    assert "n = 4" in titles[0]
    assert app._cmp_group_ui["fig"].cget("state") == "normal"
    assert app._cmp_group_ui["plot"].winfo_manager()


def test_default_pair_is_the_comparisons_reference_vs_its_first(root, tmp_path):
    p = _table(tmp_path)
    res = NS(tags=list(TAGS), pairs=[(1, 2)])
    app = _App({"path": p, "a": "", "b": ""}, comparison=res)
    _card(root, app)
    assert _pair(app) == ("Post", "Late")
    # No comparison, no pair in the Group tab: the first two segments.
    app2 = _App({"path": p})
    _card(root, app2)
    assert _pair(app2) == ("Pre", "Post")


def test_changing_the_pair_redraws(root, tmp_path):
    p = _table(tmp_path)
    app = _App({"path": p, "a": "Pre", "b": "Post"})
    _card(root, app)
    assert _titles(app)[1].startswith("Stim")
    _choose(app, "Post", "Late")
    assert _pair(app) == ("Post", "Late")
    titles = _titles(app)
    assert titles[0].startswith("Stim") and titles[1].startswith("Late"), titles
    # The Group tab picks another pair: the card follows it.
    app._grouptab_state.update(a="Pre", b="Late")
    app._compare_group_refresh()
    assert _pair(app) == ("Pre", "Late")


def test_unticked_animal_and_row_to_check_are_left_out(root, tmp_path):
    p = _table(tmp_path, n=5, qc={4: "noisy"})
    an = gd.animals_frame(gd.read_sheets(p))
    an.loc[an["subject"] == "M2", "use"] = False
    gd.save_animals(p, an)
    app = _App({"path": p, "a": "Pre", "b": "Post", "leave_out": True, "by": gc.BY_NONE})
    _card(root, app)
    assert _status(app) == "n = 3 animals (left out: 1 unticked, 1 to check) · table: group.xlsx"
    names = {t.get_text() for ax in app._cmp_group_ui["slot"].fig.axes for t in ax.texts}
    assert {"M1", "M3", "M4"} <= names and not names & {"M2", "M5"}
    # The rows to check kept in the Group tab: kept here too; the table
    # re-read once changed (an animal ticked again in the Group tab).
    app._grouptab_state["leave_out"] = False
    an.loc[an["subject"] == "M2", "use"] = True
    gd.save_animals(p, an)
    app._compare_group_refresh()
    assert _status(app).startswith("n = 5 animals ·")


def test_fewer_than_two_animals_says_so(root, tmp_path):
    p = _table(tmp_path, n=3)
    an = gd.animals_frame(gd.read_sheets(p))
    an["use"] = [True, False, False]
    gd.save_animals(p, an)
    app = _App({"path": p, "a": "Pre", "b": "Post"})
    _card(root, app)
    assert _status(app).startswith("Only 1 animal with a mean RR, SD1 and SD2 for both Pre and "
                                   "Post (2 needed)")
    assert not app._cmp_group_ui["plot"].winfo_manager()
    assert app._cmp_group_ui["fig"].cget("state") == "disabled"


def test_no_table_gives_a_clear_message(root, tmp_path):
    app = _App(None)                    # Group tab not built, nothing remembered
    _card(root, app)
    assert _status(app).startswith("No group table yet")
    assert not app._cmp_group_ui["plot"].winfo_manager()
    gone = _App({"path": str(tmp_path / "moved.xlsx")})
    _card(root, gone)
    assert _status(gone) == "⚠  Group table not found: moved.xlsx"
    # A table without segments (whole recording only).
    p = str(tmp_path / "whole.xlsx")
    gd.add_recording(p, "M1", "M", "M1.mat", {gd.SHEET_WHOLE: pd.DataFrame([{"HR (bpm)": 500}])})
    whole = _App({"path": p})
    _card(root, whole)
    assert "has no segments yet" in _status(whole)


def test_remembered_table_is_used_when_the_group_tab_is_not_built(root, tmp_path):
    p = _table(tmp_path)
    tg.save_last_table(p)
    app = _App(None)
    _card(root, app)
    assert _status(app).startswith("n = 4 animals")


def test_table_read_only_when_the_tab_is_shown(root, tmp_path, monkeypatch):
    p = _table(tmp_path)
    reads = []
    real = gd.read_sheets
    monkeypatch.setattr(gd, "read_sheets", lambda *a, **k: (reads.append(k), real(*a, **k))[1])
    tabs = _Tabs(TAB_GROUP)
    app = _App({"path": p, "a": "Pre", "b": "Post"}, tabs=tabs)
    _card(root, app)
    assert reads == [] and _status(app) == ""
    tabs.set(TAB_COMPARE)
    app._compare_group_refresh()
    app._compare_group_refresh()        # unchanged: not read again
    assert reads == [{"only": cs._GROUP_SHEETS}]
    assert _status(app).startswith("n = 4 animals")


def test_after_add_to_group_table_the_card_shows_that_table(root, tmp_path):
    app = _App({"path": ""})
    _card(root, app)
    assert _status(app).startswith("No group table yet")
    app._cmp_group_added = _table(tmp_path)
    app._compare_group_refresh(force=True)
    assert _status(app).startswith("n = 4 animals")


def test_the_choice_survives_a_ui_rebuild(root, tmp_path):
    p = _table(tmp_path)
    app = _App({"path": p, "a": "Pre", "b": "Post"})
    frame = _card(root, app)
    _choose(app, "Pre", "Late")
    frame.destroy()
    app._slots = {}                     # what a theme rebuild does
    _card(root, app)
    assert _pair(app) == ("Pre", "Late")
    assert _titles(app)[1].startswith("Late")
    assert app._slots["compare_group"] is app._cmp_group_ui["slot"]


def test_figure_export(root, tmp_path, monkeypatch):
    p = _table(tmp_path)
    app = _App({"path": p, "a": "Pre", "b": "Post"})
    _card(root, app)
    out = tmp_path / "fig.png"
    monkeypatch.setattr(cs.filedialog, "asksaveasfilename", lambda **k: str(out))
    app._save_compare_group_figure()
    assert out.exists() and out.stat().st_size > 10_000
    assert _status(app) == "✓  figure → fig.png"
    plt.close("all")


def test_group_poincare_data_rules(tmp_path):
    p = _table(tmp_path, n=4, qc={0: "noisy"})
    sheets = gd.read_sheets(p, only=cs._GROUP_SHEETS)
    assert set(sheets) == {gd.SHEET_ANIMALS, gd.SHEET_SEGMENTS}
    rows, out = cs.group_poincare_data(sheets, "Pre", "Post", leave_out=True)
    assert list(rows["subject"]) == ["M2", "M3", "M4"]
    assert out == {"unticked": 0, "ungrouped": 0, "to_check": 1, "pooled": False}
    rows, out = cs.group_poincare_data(sheets, "Pre", "Post", leave_out=False)
    assert len(rows) == 4 and out["to_check"] == 0
    # Pre vs Late: M1's alert is on Post only.
    rows, out = cs.group_poincare_data(sheets, "Pre", "Late")
    assert len(rows) == 4 and out["to_check"] == 0
    assert cs.segment_choices(sheets[gd.SHEET_SEGMENTS]) == [
        ("Pre", "Pre · Baseline"), ("Post", "Post · Stim"), ("Late", "Late")]


def _groups(p, groups, sexes):
    an = gd.animals_frame(gd.read_sheets(p))
    an["group"] = groups
    an["sex"] = sexes
    gd.save_animals(p, an)


@pytest.mark.parametrize("by", [gc.BY_NONE, gc.BY_GROUP, gc.BY_SEX])
def test_same_animals_as_the_group_tab_by_group_and_sex(tmp_path, by):
    # An animal without a group (M5) or of unknown sex (M4): left out by
    # the Group tab when it splits that way, so left out here too.
    from ecg.ui.group_figure import poincare_rows
    p = _table(tmp_path, n=5)
    _groups(p, ["CTRL", "CTRL", "STIM", "STIM", ""], ["M", "F", "M", "", "F"])
    sheets = gd.read_sheets(p)
    res = gc.analyse_segments(sheets, gd.animals_frame(sheets), by, "Pre", "Post")
    (drawn, *_rest), = res.figure["poincare"].values()
    want = sorted(poincare_rows(drawn, "Pre", "Post")["subject"])
    rows, out = cs.group_poincare_data(gd.read_sheets(p, only=cs._GROUP_SHEETS), "Pre", "Post",
                                       True, by)
    assert sorted(rows["subject"]) == want
    assert len(want) == (5 if by == gc.BY_NONE else 4)
    assert out["ungrouped"] == (0 if by == gc.BY_NONE else 1) and not out["pooled"]


def test_status_says_how_many_had_no_group(root, tmp_path):
    p = _table(tmp_path, n=5)
    _groups(p, ["CTRL", "CTRL", "STIM", "STIM", ""], ["M", "F", "M", "", "F"])
    app = _App({"path": p, "a": "Pre", "b": "Post", "by": gc.BY_GROUP})
    _card(root, app)
    assert _status(app) == "n = 4 animals (left out: 1 without a group) · table: group.xlsx"
    app._grouptab_state["by"] = gc.BY_SEX
    app._compare_group_refresh()
    assert _status(app) == "n = 4 animals (left out: 1 of unknown sex) · table: group.xlsx"
    names = {t.get_text() for ax in app._cmp_group_ui["slot"].fig.axes for t in ax.texts}
    assert "M4" not in names and "M5" in names


def test_to_check_counts_only_rows_that_had_something_to_draw(tmp_path):
    # M1: a quality alert on Post and no Late. Post → Late: M1 has nothing
    # to draw, so it is not counted among the rows to check left out.
    p = _table(tmp_path, n=4, qc={0: "noisy"})
    sheets = gd.read_sheets(p, only=cs._GROUP_SHEETS)
    seg = sheets[gd.SHEET_SEGMENTS]
    for col in [c for c in seg.columns if str(c).endswith(" Late")]:
        seg.loc[seg["subject"] == "M1", col] = np.nan
    rows, out = cs.group_poincare_data(sheets, "Post", "Late", leave_out=True, by=gc.BY_NONE)
    assert list(rows["subject"]) == ["M2", "M3", "M4"] and out["to_check"] == 0
    rows, out = cs.group_poincare_data(sheets, "Pre", "Post", leave_out=True, by=gc.BY_NONE)
    assert len(rows) == 3 and out["to_check"] == 1


def test_open_the_group_tab_opens_the_table_drawn(root, tmp_path):
    # Added from Compare segments while the Group tab had no table: the
    # button opens that table in the Group tab.
    p = _table(tmp_path)
    opened = []
    tabs = _Tabs(TAB_COMPARE)
    app = _App({"path": ""}, tabs=tabs)
    app._grouptab_open = opened.append
    app._cmp_group_added = p
    _card(root, app)
    app._compare_open_group_tab()
    assert opened == [p] and tabs.shown == TAB_GROUP
    # The Group tab already on that table: not read again.
    opened.clear()
    app._grouptab_state["path"] = p
    tabs.set(TAB_COMPARE)
    app._compare_open_group_tab()
    assert opened == [] and tabs.shown == TAB_GROUP
    # The Group tab not built yet: the table it will open is this one.
    app2 = _App(None, tabs=_Tabs(TAB_COMPARE))
    app2._cmp_group_added = p
    app2._compare_open_group_tab()
    assert app2._grouptab_state["path"] == p and tg.load_last_table() == p


def test_segments_keep_the_colours_of_the_comparison_above(root, tmp_path):
    # The comparison above draws Pre blue, Post orange, Late green: the card
    # on Post → Late keeps Post orange and Late green.
    p = _table(tmp_path)
    res = NS(tags=list(TAGS), pairs=[(0, 1)])
    app = _App({"path": p, "a": "Post", "b": "Late", "by": gc.BY_NONE}, comparison=res)
    _card(root, app)
    assert app._cmp_group_ui["shown"][5] == (cs.segment_color(1), cs.segment_color(2))
    # A segment the comparison does not hold: the colour of its place.
    assert CompareSegmentsMixin._compare_group_colors(None, "Post", "Late") == (
        cs.segment_color(0), cs.segment_color(1))
