"""Metric families on screen: the Group tab's figure metric is chosen family
first (FamilyCombo), its results table and the Compare segments table are
cut into family sections (Heart rate, HRV time domain...), and the choice
survives a refresh and a rebuild of the tab (a theme change)."""
from __future__ import annotations

import tkinter as tk

import pytest

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.core import metric_families as mf
from ecg.io import group_compare as gc
from ecg.io import group_data as gd
from ecg.ui.widgets import FamilyCombo
from tests.io.test_group_compare import SEG_HR, _segments_table
from tests.ui.test_compare_reference import _run, _table_text, tab  # noqa: F401 (fixture)
from tests.ui.test_group_tab import _page, _results_text, _walk, root  # noqa: F401 (fixture)

NAMES = ["Mean HR (bpm)", "SDNN (ms)", "Poincaré SD1 (ms)", "RMSSD (ms)", "LF/HF",
         "Poincaré SD2 (ms)"]


def test_family_combo_two_steps(root):
    got = []
    fc = FamilyCombo(root, command=got.append, values=NAMES)
    assert fc.cget("values") == NAMES and fc.get() == "Mean HR (bpm)"
    assert list(fc.cb_family.cget("values")) == [mf.HEART_RATE, mf.TIME_DOMAIN, mf.FREQUENCY,
                                                 mf.NONLINEAR]
    # A family lists its metrics only, in their order.
    fc._pick_family(mf.NONLINEAR)
    assert got == ["Poincaré SD1 (ms)"]
    assert list(fc.cb_metric.cget("values")) == ["Poincaré SD1 (ms)", "Poincaré SD2 (ms)"]
    fc._pick_metric("Poincaré SD2 (ms)")
    # Back to a family: its metric last chosen, then the other family's.
    fc._pick_family(mf.TIME_DOMAIN)
    fc._pick_metric("RMSSD (ms)")
    fc._pick_family(mf.NONLINEAR)
    assert fc.get() == "Poincaré SD2 (ms)" and fc.family() == mf.NONLINEAR
    fc._pick_family(mf.TIME_DOMAIN)
    assert got[-1] == "RMSSD (ms)"
    # .set does not call the command; a name not offered clears.
    n = len(got)
    fc.set("LF/HF")
    assert fc.family() == mf.FREQUENCY and fc.get() == "LF/HF" and len(got) == n
    fc.set("nope")
    assert fc.get() == "" and fc.family() == ""
    # New values keep the metric shown when it is still offered.
    fc.set("SDNN (ms)")
    fc.configure(values=["SDNN (ms)", "Mean HR (bpm)"])
    assert fc.get() == "SDNN (ms)"
    fc.configure(values=["Mean HR (bpm)"])
    assert fc.get() == "Mean HR (bpm)"


def test_family_combo_keyboard_steps(root):
    got = []
    fc = FamilyCombo(root, command=got.append, values=NAMES)
    fc._step_family(1)
    assert fc.family() == mf.TIME_DOMAIN and got == ["SDNN (ms)"]
    fc._step_metric(1)
    assert fc.get() == "RMSSD (ms)"
    fc._step_metric(1)                              # the last one: no change, no call
    assert fc.get() == "RMSSD (ms)" and len(got) == 2
    fc._step_family(-1)
    assert fc.family() == mf.HEART_RATE
    # The keys are bound on the combos' entries.
    assert fc.cb_metric._entry.bind("<Down>") and fc.cb_family._entry.bind("<Up>")


def _family_combo(frame) -> FamilyCombo:
    return next(w for w in _walk(frame) if isinstance(w, FamilyCombo))


def _seg_state(tmp_path):
    sh, an = _segments_table(tmp_path / "g.xlsx")
    an["group"] = ["A", "B", "A", "B"]
    gd.save_animals(str(tmp_path / "g.xlsx"), an)
    return {"path": str(tmp_path / "g.xlsx"), "analysis": gc.AN_SEGMENTS, "by": gc.BY_GROUP,
            "a": "Pre 1", "b": "Post 1", "all": True}


def test_family_combo_flat_when_the_list_is_short(root):
    # Up to flat_max metrics: one list in family order, no family box.
    fc = FamilyCombo(root, values=NAMES, flat_max=8)
    fc.pack()
    root.update_idletasks()
    assert not fc.cb_family.winfo_manager()
    assert list(fc.cb_metric.cget("values")) == [
        "Mean HR (bpm)", "SDNN (ms)", "RMSSD (ms)", "LF/HF", "Poincaré SD1 (ms)",
        "Poincaré SD2 (ms)"]
    fc._pick_metric("LF/HF")
    assert fc.get() == "LF/HF" and len(fc.cb_metric.cget("values")) == 6
    # A longer list (All metrics): two steps again.
    long_ = NAMES + [f"HRV_SDANN{i} (ms)" for i in range(1, 5)]
    fc.configure(values=long_)
    assert fc.cb_family.winfo_manager() and fc.get() == "LF/HF"
    assert list(fc.cb_metric.cget("values")) == ["LF/HF"]
    # With nothing chosen, the first metric of the first family: a heart
    # rate, not a count of beats (Data analysed is listed last).
    fc2 = FamilyCombo(root, values=["Beats", "Duration (s)", "Mean HR (bpm)"])
    assert fc2.get() == "Mean HR (bpm)"


def test_group_tab_figure_metric_is_chosen_by_family(root, tmp_path):
    st = _seg_state(tmp_path)
    _app, frame = _page(root, st)
    fc = _family_combo(frame)
    assert fc.winfo_manager()                          # shown for the paired figure
    assert list(fc.cb_family.cget("values")) == [mf.HEART_RATE, mf.TIME_DOMAIN, mf.NONLINEAR]
    assert fc.get() == SEG_HR and st["fig_metric"] == SEG_HR
    # A handful of metrics: one flat list in family order (no family box).
    assert not fc.cb_family.winfo_manager()
    offered = list(fc.cb_metric.cget("values"))
    assert offered[0] == SEG_HR and offered[-3:] == ["Poincaré SD1 (ms)", "Poincaré SD2 (ms)",
                                                     gc.POINCARE]
    # The group Poincaré is in the non-linear family with SD1 / SD2.
    assert mf.family_of(gc.POINCARE) == mf.NONLINEAR
    fc._pick_metric("Poincaré SD2 (ms)")
    assert st["fig_metric"] == "Poincaré SD2 (ms)"
    # A re-run (another By) keeps it.
    seg = next(w for w in _walk(frame) if isinstance(w, ctk.CTkSegmentedButton)
               and gc.BY_NONE in w._value_list)
    seg.set(gc.BY_NONE)
    seg._command(gc.BY_NONE)
    assert fc.get() == "Poincaré SD2 (ms)"
    # A rebuild of the page (theme change) shows the same choice.
    frame.destroy()
    _app2, frame2 = _page(root, st)
    fc2 = _family_combo(frame2)
    assert fc2.get() == "Poincaré SD2 (ms)" and fc2.family() == mf.NONLINEAR
    assert not root.errors


def test_group_tab_results_are_cut_by_family(root, tmp_path):
    st = _seg_state(tmp_path)
    _app, frame = _page(root, st)
    text = _results_text(frame)
    heads = [ln for ln in text.split("\n") if ln.strip() in
             ("HEART RATE", "HRV TIME DOMAIN", "HRV NON-LINEAR")]
    assert heads == ["HEART RATE", "HRV TIME DOMAIN", "HRV NON-LINEAR"]
    # Mean RR (listed last in the table) is shown under Heart rate.
    assert text.index("HEART RATE") < text.index("Mean RR (ms)") < text.index("HRV TIME DOMAIN")
    # One metric: no header.
    st2 = dict(st, all=False)
    st2.pop("fig_metric", None)
    _app2, frame2 = _page(root, {**st2, "analysis": gc.AN_WHOLE, "by": gc.BY_NONE})
    assert "HEART RATE" not in _results_text(frame2)


def test_compare_segments_table_has_family_sections(tab):
    app, frame = tab
    _run(app, frame, [("S1", 0, 300), ("S2", 300, 600)])
    lines = _table_text(frame).split("\n")
    heads = [ln.strip() for ln in lines if ln.strip().isupper() and "\t" not in ln]
    # Beats, Duration and Blocks used last, under DATA ANALYSED: they are
    # not heart rate.
    assert heads == ["HEART RATE", "HRV TIME DOMAIN", "HRV FREQUENCY DOMAIN", "HRV NON-LINEAR",
                     "DATA ANALYSED"]
    at = {h: lines.index(h) for h in heads}
    row = {ln.split("\t")[0].split("  ")[0]: i for i, ln in enumerate(lines)}
    assert at["HEART RATE"] < row["Mean RR"] < at["HRV TIME DOMAIN"] < row["RMSSD"]
    assert at["HRV FREQUENCY DOMAIN"] < row["LF/HF"] < at["HRV NON-LINEAR"] < row["SD1/SD2"]
    assert at["DATA ANALYSED"] < min(row["Beats"], row["Duration"], row["Blocks used (n)"])
    assert at["HEART RATE"] + 1 == row["HR, mean of block means"]
    # Every metric of the table is listed once.
    names = [n for _k, n, _u in app._SEGMENT_TABLE_METRICS]
    assert sorted(n for n in row if n in names) == sorted(names)
