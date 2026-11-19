"""Compare segments configurations relative to the stimulus: saved from the
rows of one animal as offsets from its stim time, applied to another at its
own stim time, clamped to the recording; Spike2 marker trains listed by
their first pulse."""
from __future__ import annotations

import json

import pytest

from ecg.core import segment_plan as sp

USER_ROWS = [("Pre Stim 1min", 1743, 1803), ("Post Stim 1min", 1803, 1863),
             ("Pre Stim 5min", 1503, 1803), ("Post Stim 5min", 1803, 2103),
             ("Pre Stim 10min", 1203, 1803), ("Post Stim 10min", 1803, 2403)]


def test_save_apply_round_trip_with_offsets(tmp_path):
    path = tmp_path / "t.json"
    cfg = sp.config_from_rows("Stim BNST 1-5-10 min", USER_ROWS, 1803.0)
    assert [(w.label, w.ref, w.lo, w.hi) for w in cfg.windows][:2] == [
        ("Pre Stim 1min", "start", -60.0, 0.0), ("Post Stim 1min", "start", 0.0, 60.0)]
    sp.save_template(cfg, path)
    # The file keeps the template format: the old reader reads it.
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["templates"][0]["windows"][5] == {
        "label": "Post Stim 10min", "ref": "start", "lo": 0.0, "hi": 600.0}
    (back,) = sp.user_templates(path)                     # built-ins hidden
    assert back.name == "Stim BNST 1-5-10 min" and not back.builtin
    same = sp.apply_config(back, 1803.0, 4200.0)
    assert [(w.label, w.lo, w.hi) for w in same] == [
        (lb, float(lo), float(hi)) for lb, lo, hi in USER_ROWS]
    assert not any(w.clamped for w in same)


def test_apply_with_another_stim_time(tmp_path):
    cfg = sp.config_from_rows("c", USER_ROWS, 1803.0)
    got = [(w.lo, w.hi) for w in sp.apply_config(cfg, 2000.0, 4200.0)]
    assert got == [(1940, 2000), (2000, 2060), (1700, 2000), (2000, 2300),
                   (1400, 2000), (2000, 2600)]


def test_windows_outside_the_recording_are_clamped_and_flagged():
    cfg = sp.config_from_rows("c", USER_ROWS, 1803.0)
    res = sp.apply_config(cfg, 300.0, 700.0)
    by = {w.label: w for w in res}
    assert (by["Pre Stim 1min"].lo, by["Pre Stim 1min"].clamped) == (240.0, False)
    # Pre 10 min would start at -300: cut at 0.
    assert (by["Pre Stim 10min"].lo, by["Pre Stim 10min"].hi) == (0.0, 300.0)
    assert by["Pre Stim 10min"].clamped
    # Post 10 min would end at 900: cut at the end of the recording.
    assert (by["Post Stim 10min"].hi, by["Post Stim 10min"].clamped) == (700.0, True)
    # A stim after the end: the post windows are empty (outside).
    out = sp.apply_config(cfg, 800.0, 700.0)
    post = [w for w in out if w.label.startswith("Post")]
    assert all(w.clamped and not w.hi > w.lo for w in post)


def test_delete_keeps_the_others(tmp_path):
    path = tmp_path / "t.json"
    sp.save_template(sp.config_from_rows("A", USER_ROWS[:2], 1803), path)
    sp.save_template(sp.config_from_rows("B", USER_ROWS[2:4], 1803), path)
    sp.delete_template("A", path)
    assert [t.name for t in sp.user_templates(path)] == ["B"]
    # A built-in name cannot be taken by a configuration.
    with pytest.raises(ValueError):
        sp.save_template(sp.config_from_rows(sp.BUILTIN_TEMPLATES[0].name,
                                             USER_ROWS[:1], 1803), path)


def test_default_stim_is_the_start_of_the_first_post_window():
    assert sp.default_stim(USER_ROWS) == 1803.0
    assert sp.default_stim([("A", 0, 10), ("post-x", 12, 20)]) == 12.0
    assert sp.default_stim([("A", 0, 10)]) is None


def test_marker_trains_and_choices():
    train = [1808.18 + 0.02 * k for k in range(500)]
    assert sp.marker_onsets(train) == [(1808.18, 500)]
    kb = [198.0 * (k + 1) for k in range(19)]
    ch = sp.marker_choices({"STIM SUB": train, "STIM COL": [], "Keyboard": kb})
    assert ch[0] == ("STIM SUB  1808.2 s  ×500", 1808.18)
    assert ch[1] == ("Keyboard  198.0 s", 198.0)
    assert len(ch) == 20 and not any(t.startswith("STIM COL") for t, _ in ch)
    many = sp.marker_choices({"A": [10.0 * k for k in range(200)],
                              "B": [5.0 + 10.0 * k for k in range(200)]})
    assert len(many) == 50 and sum(t.startswith("A") for t, _ in many) == 25
    assert sp.marker_choices({}) == []
