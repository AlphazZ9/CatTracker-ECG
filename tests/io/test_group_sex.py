# -*- coding: utf-8 -*-
"""The sex of the animal in the group table: written by group_row, a meta
column (never a metric), shown in the Excel report and the Prism tables
(row titles "1146 (F)", the animals by sex in the notes); a table made
before it still opens and is tested as before."""
from __future__ import annotations

import pandas as pd

from ecg.io import group_stats as gs
from ecg.io import prism_template as pt
from ecg.io.group_table import append_group_row, group_row
from ecg.io.subject_name import normalise_sex

HR = "Mean HR (60000/mean NN) (bpm)"
METRICS = [("mean_hr", "Mean HR (60000/mean NN)", "bpm"), ("sdnn", "SDNN", "ms")]


def _segs(i: int):
    return [{"label": "Pre", "t_lo": 0.0, "t_hi": 60.0, "mean_hr": 500.0 + i, "sdnn": 4.0},
            {"label": "Post", "t_lo": 60.0, "t_hi": 120.0, "mean_hr": 520.0 + 2 * i,
             "sdnn": 5.0 + i}]


ANIMALS = [("1146", "F"), ("206", ""), ("1150", "f"), ("300", "M")]


def _table(tmp_path):
    path = str(tmp_path / "g.xlsx")
    for i, (subject, sex) in enumerate(ANIMALS):
        append_group_row(path, group_row(subject, f"{subject}.mat", _segs(i), METRICS,
                                         template="T", sex=sex))
    return path, gs.read_group_table(path)


def test_group_row_writes_the_sex_beside_the_subject():
    row = group_row("1146", "Stim 1146 F.mat", _segs(0), METRICS, sex="female")
    assert list(row)[:4] == ["subject", "sex", "file", "template"]
    assert row["sex"] == "F"
    assert group_row("206", "x.mat", _segs(0), METRICS)["sex"] == ""


def test_normalise_sex():
    assert [normalise_sex(v) for v in ("m", " F ", "Male", "femelle", "", None,
                                       float("nan"), "?")] == ["M", "F", "M", "F", "", "",
                                                               "", "?"]


def test_the_sex_is_not_a_metric(tmp_path):
    _path, df = _table(tmp_path)
    assert list(df.columns)[:3] == ["subject", "sex", "file"]
    assert gs.segment_tags(df) == ["Pre", "Post"]
    assert gs.metric_names(df, "Pre", "Post") == [HR, "SDNN (ms)"]
    assert pt.group_metric_names(df) == [HR, "SDNN (ms)"]
    assert "sex" in gs.ID_COLUMNS
    # The statistics use every animal, whatever the sex, as without the column.
    st = gs.paired_stats(df, "Pre", "Post", "all").set_index("Metric")
    st0 = gs.paired_stats(df.drop(columns="sex"), "Pre", "Post", "all").set_index("Metric")
    assert int(st.loc[HR, "n"]) == 4
    pd.testing.assert_frame_equal(st, st0)
    res = gs.all_pairs_stats(df)
    assert len(res.pairs) == 1 and len(res.pairs[0].used) == 4
    assert all(c.ok for c in res.pairs[0].checks)
    assert {c.subject: c.sex for c in res.pairs[0].checks} == {
        "1146": "F", "1150": "F", "206": "", "300": "M"}


def test_an_old_table_without_sex_opens_and_gets_it_beside_the_subject(tmp_path):
    path = str(tmp_path / "old.xlsx")
    old = pd.DataFrame([group_row("100", "100.mat", _segs(0), METRICS, template="T")])
    old = old.drop(columns="sex")
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        old.to_excel(xw, sheet_name="Group", index=False)
    df = gs.read_group_table(path)
    assert "sex" not in df.columns and gs.metric_names(df, "Pre", "Post") == [HR, "SDNN (ms)"]
    # Its reports and Prism tables are as before.
    frames = gs._report_frames(df, "Pre", "Post", gs.paired_stats(df, "Pre", "Post"),
                               gs.check_rows(df, "Pre", "Post"))
    assert "sex" not in frames["Rows checked"].columns
    assert "sex" not in frames["Paired values"].columns
    assert pt.sex_notes(df) == []
    assert list(pt.group_tables(df)[0][1].index) == ["100"]

    append_group_row(path, group_row("1146", "1146.mat", _segs(1), METRICS, template="T",
                                     sex="F"))
    df = gs.read_group_table(path)
    assert list(df.columns)[:3] == ["subject", "sex", "file"]
    assert df.set_index("subject")["sex"].map(gs.sex_text).to_dict() == {100: "", 1146: "F"}


def test_the_excel_report_shows_the_sex(tmp_path):
    _path, df = _table(tmp_path)
    st = gs.paired_stats(df, "Pre", "Post")
    out = tmp_path / "report.xlsx"
    gs.write_group_report(str(out), df, "Pre", "Post", st, gs.check_rows(df, "Pre", "Post"))
    book = pd.read_excel(out, sheet_name=None, dtype=str)
    rows = book["Rows checked"]
    assert list(rows.columns) == ["subject", "sex", "file", "status", "problems"]
    assert rows.set_index("subject")["sex"].fillna("").to_dict() == {
        "1146": "F", "1150": "F", "206": "", "300": "M"}
    paired = book["Paired values"]
    assert list(paired.columns)[:4] == ["Metric", "subject", "sex", "file"]
    assert set(paired.loc[paired["subject"] == "1146", "sex"]) == {"F"}
    # All pairs: the same tables on each pair's sheet.
    out2 = tmp_path / "all.xlsx"
    sheets = gs.write_all_pairs_report(str(out2), gs.all_pairs_stats(df))
    raw = pd.read_excel(out2, sheet_name=sheets[0], header=None, dtype=str)
    assert (raw == "sex").sum().sum() == 2          # Rows checked and Paired values


def test_prism_row_titles_carry_the_sex(tmp_path):
    _path, df = _table(tmp_path)
    tables = dict(pt.group_tables(df))
    titles = ["1146 (F)", "1150 (F)", "206", "300 (M)"]
    assert sorted(tables[HR].index) == titles
    res = gs.all_pairs_stats(df)
    ptab = dict(pt.pair_tables(res))
    assert sorted(next(iter(ptab.values())).index) == titles
    notes = dict(pt.sex_notes(df))
    assert notes["Sex"] == "F: 1146, 1150; M: 300; not given: 206"   # in table order
    assert "1146 (F)" in notes["Row titles"]
    # Written to a .pzfx: rows titled so, notes in the info sheet.
    out = tmp_path / "g.pzfx"
    pt.write_metric_pzfx(pt.group_tables(df), str(out), notes=pt.sex_notes(df))
    got = pt.read_tables(str(out))
    assert got[HR]["rows"] == list(tables[HR].index) and sorted(got[HR]["rows"]) == titles
    text = out.read_text(encoding="utf-8")
    assert "F: 1146, 1150; M: 300; not given: 206" in text


def test_a_subject_on_two_rows_keeps_its_sex_and_gets_its_file():
    df = pd.DataFrame([{"subject": "1146", "sex": "F", "file": "a.mat"},
                       {"subject": "1146", "sex": "F", "file": "b.mat"},
                       {"subject": "", "sex": "M", "file": "M_7.mat"}])
    assert pt._row_titles(df) == ["1146 (F) (a)", "1146 (F) (b)", "M_7 (M)"]
    assert pt.sex_notes(df)[0] == ("Sex", "F: 1146; M: M_7")
