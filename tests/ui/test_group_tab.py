"""Group tab: opens a group table, lists the animals (group and use saved in
the table), runs the chosen analysis at once and enables its exports; Add
open recording writes the recording open (whole, time course, annotations,
events, segments) to the table."""
from __future__ import annotations

import os
from types import SimpleNamespace as NS

import numpy as np
import pytest


from ecg.io import db
from ecg.io import group_compare as gc
from ecg.io import group_data as gd
from ecg.app import group_helpers as tg
from tests.io.test_group_compare import _table
from ecg.app.group_helpers import recording_lines




def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)




















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












