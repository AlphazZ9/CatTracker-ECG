"""The Group tab's analyses (ecg.io.group_compare): groups by free group, sex
or none; tests within and between groups; the mixed ANOVA, checked against
pingouin's mixed_anova / rm_anova on the same data (values pinned here)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ecg.io import group_compare as gc
from ecg.io import group_data as gd
from ecg.io import group_stats as gs
from ecg.io.prism_template import read_tables, write_metric_pzfx


def _anova_data():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(9, 5)) * np.array([1, 2, 3, 1.5, 0.5]) + np.arange(5)
    g = np.array(["A"] * 4 + ["B"] * 5)
    y[g == "B"] += 0.8
    y[g == "B", 2] += 1.5
    return y, g


def test_mixed_anova_matches_pingouin():
    y, g = _anova_data()
    an = gc.mixed_anova(y, g).set_index("Source")
    # pingouin 0.5: mixed_anova(dv, within, between, subject, correction=True)
    assert an.loc["Group", ["df1", "df2"]].tolist() == [1, 7]
    assert an.loc["Group", "F"] == pytest.approx(3.70192, abs=1e-4)
    assert an.loc["Group", "p"] == pytest.approx(0.09576, abs=1e-4)
    assert an.loc["Within", "F"] == pytest.approx(9.73795, abs=1e-4)
    assert an.loc["Within", "epsilon"] == pytest.approx(0.52324, abs=1e-4)
    assert an.loc["Within", "p GG"] == pytest.approx(0.00186, abs=1e-4)
    assert an.loc["Group × within", "F"] == pytest.approx(1.23318, abs=1e-4)
    assert an.loc["Group × within", "p GG"] == pytest.approx(0.32165, abs=1e-4)


def test_one_group_is_the_repeated_measures_anova():
    y, g = _anova_data()
    an = gc.mixed_anova(y[g == "A"], g[g == "A"])
    assert list(an["Source"]) == ["Within"]
    r = an.iloc[0]
    assert (r["df1"], r["df2"]) == (4, 12)
    assert r["F"] == pytest.approx(5.53957, abs=1e-4)
    assert r["p"] == pytest.approx(0.0092, abs=1e-4)
    assert r["epsilon"] == pytest.approx(0.49876, abs=1e-4)
    assert r["p GG"] == pytest.approx(0.04353, abs=1e-4)


def test_two_levels_need_no_correction():
    y, g = _anova_data()
    an = gc.mixed_anova(y[:, :2], g)
    assert (an["epsilon"].dropna() == 1.0).all()


def test_between_test_two_and_more_groups():
    two = gc.between_test([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert two["rank test"] == "Mann-Whitney" and two["p"] == pytest.approx(0.0286, abs=1e-3)
    three = gc.between_test([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert three["rank test"] == "Kruskal-Wallis" and three["p"] < 0.05
    assert math.isnan(gc.between_test([[1], []])["p"])


def _animals(rows):
    return pd.DataFrame(rows, columns=["subject", "sex", "group", "use", "file"])


def test_groups_by_group_sex_none_and_the_control_first():
    an = _animals([("1", "M", "CNO", True, "a"), ("2", "F", "Control", True, "b"),
                   ("3", "", "", True, "c"), ("4", "F", "CNO", False, "d")])
    gmap, order, notes = gc.animal_groups(an, gc.BY_GROUP)
    assert order == ["Control", "CNO"] and ("4", "d") not in gmap
    assert any("3" in n for n in notes)
    gmap, order, _ = gc.animal_groups(an, gc.BY_SEX)
    assert order == ["Males", "Females"] and ("3", "c") not in gmap
    gmap, order, _ = gc.animal_groups(an, gc.BY_NONE)
    assert order == [gc.ALL_LABEL] and len(gmap) == 3


def _table(tmp_path, n=8):
    """n animals, half Control (HR + 20 bpm after the stimulation), half CNO
    (+ 5), a stimulation at 300 s + 10 s per animal."""
    p = str(tmp_path / "g.xlsx")
    rng = np.random.default_rng(2)
    rows = []
    for i in range(n):
        grp = "Control" if i % 2 == 0 else "CNO"
        sex = "M" if i < n // 2 else "F"
        stim = 300.0 + 10 * i
        gain = 20.0 if grp == "Control" else 5.0
        t, rr = 0.0, []
        ts = []
        while t < 900:
            hr = 500.0 + (gain if t > stim else 0.0) + rng.normal(0, 2)
            r = 60000.0 / hr
            t += r / 1000.0
            ts.append(t)
            rr.append(r)
        ts, rr = np.array(ts), np.array(rr)
        seg = {"Mean HR (bpm) Pre": 500 + rng.normal(0, 2), "Mean HR (bpm) Post": 500 + gain,
               "start (s) Pre": stim - 120, "start (s) Post": stim, "label Pre": "Pre",
               "label Post": "Post"}
        lines = {gd.SHEET_TIME: gd.time_frame(ts, rr, np.ones(len(rr), bool)),
                 gd.SHEET_ANNOT: gd.annotations_frame([{"t_start": stim, "label": "Stim"}]),
                 gd.SHEET_WHOLE: pd.DataFrame([{"Mean HR (60000/mean NN) (bpm)":
                                                float(np.mean(60000 / rr))}]),
                 gd.SHEET_SEGMENTS: pd.DataFrame([seg]),
                 gd.SHEET_EVENTS: gd.events_frame([
                     {"label": "Stim", "onset_s": stim, "valid": True,
                      "metrics": {"rCSI": {"pre": 1.0 + 0.01 * i, "post": 1.0 + gain / 10}}},
                     {"label": "Other", "onset_s": 10.0, "valid": True,
                      "metrics": {"rCSI": {"pre": 9.0, "post": 9.0}}}])}
        gd.add_recording(p, str(100 + i), sex, f"{100 + i}.smr", lines)
        rows.append((str(100 + i), grp))
    sh = gd.read_sheets(p)
    an = gd.animals_frame(sh)
    an["group"] = an["subject"].map(dict(rows))
    gd.save_animals(p, an)
    return p


def test_time_course_by_group(tmp_path):
    p = _table(tmp_path)
    sh = gd.read_sheets(p)
    an = gd.animals_frame(sh)
    res = gc.analyse_time(sh, an, gc.BY_GROUP, "stim", "HR", bin_s=30, start_min=-4,
                          end_min=8, window_min=(0, 2))
    assert res.groups == ["Control", "CNO"]
    s = res.stats.iloc[0]
    assert s["Control Δ mean"] == pytest.approx(20, abs=1.5)
    assert s["CNO Δ mean"] == pytest.approx(5, abs=1.5)
    assert s["between p"] == pytest.approx(0.0286, abs=1e-3)      # 4 vs 4, all apart
    assert set(res.anova["Source"]) == {"Group", "Time", "Group × time"}
    curve = res.figure["curves"]["Control"]
    t = res.figure["t_min"]
    assert len(t) == 24 and abs(np.nanmean(curve["mean"][t < 0])) < 1.0
    assert np.nanmean(curve["mean"][(t > 1) & (t < 2)]) == pytest.approx(20, abs=2)
    xy = [tb for tb in res.prism if len(tb) == 3]
    assert len(xy) == 2 and xy[0][1].columns[0] == "Time (min)"
    out = tmp_path / "t.pzfx"
    write_metric_pzfx(res.prism, str(out))
    assert len(read_tables(str(out))) == len(res.prism)


def test_time_course_leaves_out_an_animal_without_the_annotation(tmp_path):
    p = _table(tmp_path, n=6)
    sh = gd.read_sheets(p)
    sh[gd.SHEET_ANNOT] = sh[gd.SHEET_ANNOT].iloc[1:]
    res = gc.analyse_time(sh, gd.animals_frame(sh), gc.BY_NONE, "Stim", "RR", bin_s=60,
                          start_min=-4, end_min=8, window_min=(0, 2), delta=False)
    assert any("100" in n for n in res.notes) and res.n_animals == 5
    with pytest.raises(ValueError):
        gc.analyse_time(sh, gd.animals_frame(sh), gc.BY_NONE, "Stim", start_min=1)


def test_segments_whole_and_events(tmp_path):
    p = _table(tmp_path)
    sh = gd.read_sheets(p)
    an = gd.animals_frame(sh)
    seg = gc.analyse_segments(sh, an, gc.BY_GROUP, "Pre", "Post")
    r = seg.stats.iloc[0]
    assert r["Control Δ mean"] == pytest.approx(20, abs=3) and r["between p"] < 0.05
    assert set(seg.anova["Source"]) == {"Group", "Segment", "Group × segment"}
    assert {"Test", "Control", "CNO", "p", "p Holm"} <= set(seg.display.columns)
    whole = gc.analyse_whole(sh, an, gc.BY_SEX)
    assert whole.groups == ["Males", "Females"] and "between p (Holm)" in whole.stats
    one = gc.analyse_whole(sh, an, gc.BY_NONE)
    assert any("descriptive" in n for n in one.notes)
    ev = gc.analyse_events(sh, an, gc.BY_GROUP, "Stim")
    r = ev.stats.set_index("Metric").loc["rCSI"]
    assert r["Control post mean"] == pytest.approx(3.0)
    assert r["CNO post mean"] == pytest.approx(1.5)
    assert gc.event_labels(sh) == ["Other", "Stim"]
    out = tmp_path / "s.xlsx"
    gc.write_results(str(out), seg, "g.xlsx")
    assert {"Results", "Statistics", "ANOVA", "Per animal", "Settings"} <= set(
        pd.read_excel(out, sheet_name=None))


def test_an_unticked_animal_is_left_out(tmp_path):
    p = _table(tmp_path)
    sh = gd.read_sheets(p)
    an = gd.animals_frame(sh)
    an.loc[0, "use"] = False
    res = gc.analyse_whole(sh, an, gc.BY_NONE)
    assert res.n_animals == len(an) - 1


def _event_sheets(rows):
    ev = pd.concat([gd.events_frame([r]).assign(subject=s, file=f"{s}.smr")
                    for s, r in rows], ignore_index=True)
    an = pd.DataFrame([{"subject": s, "sex": "", "group": "", "use": True, "file": f"{s}.smr"}
                       for s in sorted({s for s, _ in rows})])
    return {gd.SHEET_EVENTS: ev}, an


def _event(onset, method, window, hr_post, valid=True, hr_valid=True):
    return {"label": "Stim", "onset_s": onset, "method": method, "window_s": window,
            "valid": valid, "metrics": {"HR": {"pre": 500.0, "post": hr_post, "valid": hr_valid},
                                        "rCSI": {"pre": 1.0, "post": 2.0, "valid": valid}}}


def test_events_keep_one_configuration():
    """Codex review: the same event saved with two methods / windows must not
    be pooled (counted twice, weighted by configuration)."""
    sheets, an = _event_sheets([
        ("1", _event(100.0, "robust", 15.0, 520.0)),
        ("1", _event(100.0, "classic", 30.0, 600.0)),
        ("2", _event(90.0, "robust", 15.0, 530.0)),
    ])
    assert gc.event_configs(sheets) == ["robust · 15 s", "classic · 30 s"]
    res = gc.analyse_events(sheets, an, gc.BY_NONE, metrics=["HR"])
    vals = res.figure["values"]["HR"]
    assert vals[("1", "1.smr")] == (500.0, 520.0) and vals[("2", "2.smr")] == (500.0, 530.0)
    assert any("classic · 30 s" in n for n in res.notes)
    other = gc.analyse_events(sheets, an, gc.BY_NONE, metrics=["HR"], config="classic · 30 s")
    assert other.figure["values"]["HR"] == {("1", "1.smr"): (500.0, 600.0)}


def test_events_use_each_metrics_own_validity():
    """Codex review: an event unusable for the CSI family can be fine for HR."""
    sheets, an = _event_sheets([
        ("1", _event(100.0, "robust", 15.0, 520.0, valid=False, hr_valid=True)),
        ("1", _event(200.0, "robust", 15.0, 540.0, valid=True, hr_valid=False)),
    ])
    res = gc.analyse_events(sheets, an, gc.BY_NONE, metrics=["HR", "rCSI"])
    assert res.figure["values"]["HR"] == {("1", "1.smr"): (500.0, 520.0)}
    assert res.figure["values"]["rCSI"] == {("1", "1.smr"): (1.0, 2.0)}



def test_no_group_set_says_so(tmp_path):
    p = _table(tmp_path, n=4)
    sh = gd.read_sheets(p)
    an = gd.animals_frame(sh)
    an["group"] = ""
    with pytest.raises(ValueError, match="No animal has a group"):
        gc.analyse_time(sh, an, gc.BY_GROUP, "Stim", start_min=-4, end_min=8, window_min=(0, 2))


def test_two_recordings_of_one_subject_draw_two_curves(tmp_path):
    p = _table(tmp_path, n=4)
    sh = gd.read_sheets(p)
    for name in (gd.SHEET_TIME, gd.SHEET_ANNOT):
        df = sh[name]
        sh[name] = df.assign(subject=df["subject"].replace({"101": "100"}))
    an = gd.animals_frame(sh)
    res = gc.analyse_time(sh, an, gc.BY_NONE, "Stim", bin_s=60, start_min=-4, end_min=8,
                          window_min=(0, 2))
    curves = res.figure["curves"][gc.ALL_LABEL]
    assert curves["n"] == 4 and len(curves["animals"]) == 4
    assert {"100 (100.smr)", "100 (101.smr)"} <= set(curves["animals"])


# ── moved from the Compare segments group card ──────────────────────────

SEG_HR = "Mean HR (60000/mean NN) (bpm)"


def _segments_table(path):
    """4 animals, segments Pre 1 / Post 1 / Pre 5 / Post 5; M3's Post 5 has a
    quality alert (a row to check for the Pre 5 / Post 5 pair)."""
    from ecg.io.group_table import append_group_row, group_row
    metrics = [("hr_mean", "Mean HR (60000/mean NN)", "bpm"), ("sdnn", "SDNN", "ms"),
               ("sd1", "Poincaré SD1", "ms"), ("sd2", "Poincaré SD2", "ms"),
               ("rr_mean", "Mean RR", "ms")]
    labels = ["Pre 1", "Post 1", "Pre 5", "Post 5"]
    for i in range(4):
        segs = [{"label": lbl, "t_lo": 60.0 * k, "t_hi": 60.0 * k + 60, "duration_s": 60.0,
                 "hr_mean": 500 + i + 10 * k + (i % 2), "sdnn": 3 + 0.1 * i + k,
                 "sd1": 2 + 0.1 * k, "sd2": 5 + 0.1 * i, "rr_mean": 120 - k}
                for k, lbl in enumerate(labels)]
        flags = [[] for _ in labels]
        if i == 2:
            flags[3] = ["few beats"]
        append_group_row(str(path), group_row(f"M{i + 1}", f"M{i + 1}.mat", segs, metrics,
                                              flags=flags))
    sh = gd.read_sheets(str(path))
    return sh, gd.animals_frame(sh)


def test_rows_to_check_are_left_out_and_listed(tmp_path):
    sh, an = _segments_table(tmp_path / "g.xlsx")
    out = gc.analyse_segments(sh, an, gc.BY_NONE, "Pre 5", "Post 5")
    assert len(out.figure["values"][SEG_HR]) == 3
    assert any(n.startswith("Left out") and "M3" in n and "few beats" in n for n in out.notes)
    kept = gc.analyse_segments(sh, an, gc.BY_NONE, "Pre 5", "Post 5", leave_out=False)
    assert len(kept.figure["values"][SEG_HR]) == 4
    assert any(n.startswith("To check (kept)") for n in kept.notes)
    # The other pair has nothing to check: all four animals.
    other = gc.analyse_segments(sh, an, gc.BY_NONE, "Pre 1", "Post 1")
    assert len(other.figure["values"][SEG_HR]) == 4


def test_all_pairs(tmp_path):
    sh, an = _segments_table(tmp_path / "g.xlsx")
    res = gc.analyse_segments(sh, an, gc.BY_NONE, "", "", pairs=gc.PAIRS_ALL)
    assert set(res.stats["Pair"]) == {"Post 1 vs Pre 1", "Post 5 vs Pre 5"}
    assert any(t.startswith("[Post 5 vs Pre 5]") for t in res.display["Test"])
    assert f"{SEG_HR} · Post 1 vs Pre 1" in res.figure["values"]
    assert res.figure["names_by"][f"{SEG_HR} · Post 5 vs Pre 5"] == ("Pre 5", "Post 5")
    assert len(res.figure["values"][f"{SEG_HR} · Post 5 vs Pre 5"]) == 3   # M3 left out
    assert any(t[0].endswith("· Post 5 vs Pre 5") for t in res.prism)
    assert set(res.figure["poincare"]) == {f"{gc.POINCARE} · Post 1 vs Pre 1",
                                           f"{gc.POINCARE} · Post 5 vs Pre 5"}


def test_poincare_group_figure_is_drawn(tmp_path):
    from matplotlib.figure import Figure
    from ecg.ui.group_tab_figure import draw_group_tab_figure
    sh, an = _segments_table(tmp_path / "g.xlsx")
    res = gc.analyse_segments(sh, an, gc.BY_NONE, "Pre 1", "Post 1")
    fig = Figure()
    assert draw_group_tab_figure(fig, res, metric=gc.POINCARE) is True
    assert len(fig.axes) >= 2


def test_post_hoc_and_effect_sizes():
    assert gc.hedges_g([1, 2, 3, 4], [3, 4, 5, 6]) == pytest.approx(1.347, abs=1e-3)   # d 1.549 x J 0.870
    assert gc.cohen_dz([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    ph = gc.posthoc({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8], "C": [9, 10, 11, 12]})
    assert [r["pair"] for r in ph] == ["B vs A", "C vs A", "C vs B"]
    assert all(r["p Holm"] >= r["p"] for r in ph)


def test_three_groups_get_post_hoc_rows(tmp_path):
    p = _table(tmp_path, n=6)
    sh = gd.read_sheets(p)
    an = gd.animals_frame(sh)
    an["group"] = ["Control", "CNO", "Sal", "Control", "CNO", "Sal"]
    res = gc.analyse_whole(sh, an, gc.BY_GROUP)
    assert res.stats.iloc[0]["between test"] == "Kruskal-Wallis"
    assert sum(t.strip().startswith("post-hoc") for t in res.display["Test"]) == 3
    assert "post-hoc (Mann-Whitney, Holm)" in res.stats.columns
    two = gc.analyse_whole(sh, an.assign(group=["Control", "CNO"] * 3), gc.BY_GROUP)
    assert "g = " in two.display.iloc[0]["p"] and "Hedges g" in two.stats.columns


# ── time course: exact alignment, per-bin baseline, bins anchored on 0 ──

def _raw_recording(seed, stim, dur=2700.0, drift=0.0, bad=None):
    """(t, rr, valid) of a synthetic recording: RR around 120 ms with beat
    noise, a 300 s oscillation of *drift* ms (the same over any 5 min, but a
    10 min SDNN holds much more of it than a 1 min one) and ~2 % invalid beats (plus the [a, b) s
    span from *stim* of *bad* when given)."""
    rng = np.random.default_rng(seed)
    n = int(dur / 0.12) + 50
    rr = 120.0 + rng.normal(0, 3, n) + drift * np.sin(2 * np.pi * np.arange(n) * 0.12 / 300)
    t = np.cumsum(rr) / 1000.0
    keep = t < dur
    t, rr = t[keep], rr[keep]
    valid = rng.random(len(rr)) > 0.02
    if bad is not None:
        valid &= ~((t - stim >= bad[0]) & (t - stim < bad[1]))
    return t, rr, valid


def _raw_table(tmp_path, recs, old=False):
    """A group table of the raw recordings *recs* ({subject: (t, rr, valid,
    stim)}), through the workbook; *old*: TimeCourse lines as written before
    anchor_s (the recording-start grid only, no anchor column)."""
    p = str(tmp_path / "raw.xlsx")
    for s, (t, rr, valid, stim) in recs.items():
        an = gd.annotations_frame([{"t_start": stim, "label": "STIM"}])
        tc = gd.time_frame(t, rr, valid, anchors=an["t_start"].tolist())
        if old:
            tc = tc.loc[tc["anchor_s"] == 0.0].drop(columns="anchor_s")
        gd.add_recording(p, s, "M", f"{s}.mat", {gd.SHEET_TIME: tc, gd.SHEET_ANNOT: an})
    sh = gd.read_sheets(p)
    return sh, gd.animals_frame(sh)


def _direct(t, rr, valid, stim, metric, a, b):
    """*metric* of the raw beats whose time lies in [a, b) s from *stim*."""
    rel = t - stim
    m = valid & (rel >= a) & (rel < b)
    if metric == "HR":
        return float(60000.0 / np.mean(rr[m]))
    if metric == "RR":
        return float(np.mean(rr[m]))
    if metric == "SDNN":
        return float(np.std(rr[m], ddof=1))
    pair = valid[1:] & valid[:-1] & (rel[1:] >= a) & (rel[1:] < b)
    return float(np.sqrt(np.mean(np.diff(rr)[pair] ** 2)))


def _curves(res):
    return {k: v for c in res.figure["curves"].values() for k, v in c["animals"].items()}


@pytest.mark.parametrize("metric", ["HR", "RR", "SDNN", "RMSSD"])
@pytest.mark.parametrize("bin_s", [60, 30])
def test_time_course_bins_are_exact_from_a_stim_off_the_5_s_grid(tmp_path, metric, bin_s):
    recs = {"206": (*_raw_recording(0, 1803.37), 1803.37),
            "207": (*_raw_recording(1, 1211.82), 1211.82),
            "208": (*_raw_recording(2, 1499.99), 1499.99)}
    sh, an = _raw_table(tmp_path, recs)
    res = gc.analyse_time(sh, an, gc.BY_NONE, "STIM", metric, bin_s=bin_s, start_min=-10,
                          end_min=15, window_min=(0, 5), delta=False)
    assert not any("Re-add" in n for n in res.notes)
    edges = np.arange(-600, 900 + 1, bin_s)
    curves = _curves(res)
    for s, (t, rr, valid, stim) in recs.items():
        want = [_direct(t, rr, valid, stim, metric, a, b) for a, b in zip(edges[:-1], edges[1:])]
        assert np.allclose(curves[s], want, rtol=0, atol=1e-9), s
        row = res.per_animal.set_index("subject").loc[s]
        assert row["baseline"] == pytest.approx(np.mean(want[:600 // bin_s]), abs=1e-9)
        assert row["window"] == pytest.approx(
            np.mean(want[600 // bin_s:900 // bin_s]), abs=1e-9)


def test_delta_sdnn_averages_zero_before_the_stim(tmp_path):
    # A slow drift makes a 10 min SDNN much larger than a 1 min one: the
    # baseline must be the mean of the bins, not one value over its span.
    recs = {str(300 + i): (*_raw_recording(10 + i, 1803.37 + i, drift=15.0), 1803.37 + i)
            for i in range(4)}
    sh, an = _raw_table(tmp_path, recs)
    for metric in ("SDNN", "RMSSD", "HR", "RR"):
        res = gc.analyse_time(sh, an, gc.BY_NONE, "STIM", metric, bin_s=60, start_min=-10,
                              end_min=10, window_min=(0, 5))
        t = res.figure["t_min"]
        for v in _curves(res).values():
            assert abs(np.nanmean(v[t < 0])) < 1e-9, metric
        # Stationary around the stim: the window does not differ from the baseline.
        d = res.stats.iloc[0][f"{gc.ALL_LABEL} Δ mean"]
        assert abs(d) < (1.0 if metric != "HR" else 3.0), (metric, d)
        if metric == "SDNN":
            # One SDNN over the 10 min would sit well above the bins' own.
            t_, rr, valid, stim = recs["300"]
            whole = _direct(t_, rr, valid, stim, "SDNN", -600, 0)
            base = res.per_animal.set_index("subject").loc["300", "baseline"]
            assert whole - base > 3.0


def test_an_old_table_without_anchors_still_runs_with_a_note(tmp_path):
    recs = {"206": (*_raw_recording(0, 1803.37), 1803.37),
            "207": (*_raw_recording(1, 1500.0), 1500.0)}
    sh, an = _raw_table(tmp_path, recs, old=True)
    assert "anchor_s" not in sh[gd.SHEET_TIME].columns
    res = gc.analyse_time(sh, an, gc.BY_NONE, "STIM", "HR", bin_s=60, start_min=-10,
                          end_min=10, window_min=(0, 5))
    note = next(n for n in res.notes if "Re-add" in n)
    # 207's stim is on the recording-start grid: exact, not named.
    assert "206" in note and "207" not in note and "Add open recording" in note
    assert np.isfinite(_curves(res)["206"]).all()


def test_bin_size_is_a_multiple_of_5_s_and_bins_start_at_0(tmp_path):
    recs = {"206": (*_raw_recording(0, 1803.37), 1803.37),
            "207": (*_raw_recording(1, 1211.82), 1211.82)}
    sh, an = _raw_table(tmp_path, recs)
    assert gc.time_bin_size(60) == (60.0, "")
    b, note = gc.time_bin_size(62)
    assert b == 60.0 and "rounded" in note
    for bad in (0, -5, 3, float("nan"), "abc"):
        with pytest.raises(ValueError):
            gc.time_bin_size(bad)
    res = gc.analyse_time(sh, an, gc.BY_NONE, "STIM", "HR", bin_s=70, start_min=-4,
                          end_min=8, window_min=(0, 2))
    t = res.figure["t_min"] * 60
    # Edges -240, -210, -140, -70, 0, 70...: no bin straddles the stim.
    assert t[0] == pytest.approx(-225) and t[1] == pytest.approx(-175)
    assert t[3] == pytest.approx(-35) and t[4] == pytest.approx(35)
    assert any("First bin cut" in n for n in res.notes)
    assert any("Last bin cut" in n for n in res.notes)
    res = gc.analyse_time(sh, an, gc.BY_NONE, "STIM", "HR", bin_s=61, start_min=-4,
                          end_min=8, window_min=(0, 2))
    assert res.figure["bin_s"] == 60 and any("rounded to 60 s" in n for n in res.notes)
    assert not any("cut" in n for n in res.notes)


def test_too_few_valid_bins_leave_an_animal_out_of_the_tests(tmp_path):
    recs = {"206": (*_raw_recording(0, 1803.37, bad=(-600, -200)), 1803.37),
            "207": (*_raw_recording(1, 1211.82), 1211.82),
            "208": (*_raw_recording(2, 1499.99), 1499.99)}
    sh, an = _raw_table(tmp_path, recs)
    res = gc.analyse_time(sh, an, gc.BY_NONE, "STIM", "HR", bin_s=60, start_min=-10,
                          end_min=10, window_min=(0, 5), delta=False)
    row = res.per_animal.set_index("subject")
    assert math.isnan(row.loc["206", "baseline"]) and math.isfinite(row.loc["207", "baseline"])
    note = next(n for n in res.notes if n.startswith("206:"))
    # 400 s of the 600 s before the stim invalid (and half of the next bin): 3 of 10 bins.
    assert "only 3 of its 10 baseline bins have data (at least 5 needed)" in note
    assert "left out of the baseline, window and Δ means and the tests" in note
    assert not any(n.startswith(("207:", "208:")) for n in res.notes)
    st = res.stats.iloc[0]
    # The baseline, window and Δ means are over the same animals.
    for lab in ("baseline", "window", "Δ"):
        assert st[f"{gc.ALL_LABEL} {lab} n"] == 2, lab
    assert st[f"{gc.ALL_LABEL} window mean"] - st[f"{gc.ALL_LABEL} baseline mean"] == \
        pytest.approx(st[f"{gc.ALL_LABEL} Δ mean"], abs=1e-9)


@pytest.mark.parametrize("start, end, moved", [(-10.02, 15, "'from' -10.02 min moved to -10 min"),
                                               (-9.99, 15, "'from' -9.99 min moved to -10 min"),
                                               (-10, 15.02, "'to' 15.02 min moved to 15 min")])
def test_from_and_to_off_the_5_s_grid_are_moved_onto_it(tmp_path, start, end, moved):
    recs = {"206": (*_raw_recording(0, 1803.37), 1803.37),
            "207": (*_raw_recording(1, 1211.82), 1211.82),
            "208": (*_raw_recording(2, 1500.0), 1500.0)}
    sh, an = _raw_table(tmp_path, recs)
    res = gc.analyse_time(sh, an, gc.BY_NONE, "STIM", "HR", bin_s=60, start_min=start,
                          end_min=end, window_min=(0, 5), delta=False)
    assert any(n.startswith(moved) for n in res.notes), res.notes
    # No sliver bin empty for every animal: all of them stay in the ANOVA.
    assert not any("Left out of the ANOVA" in n or "cut" in n for n in res.notes)
    assert len(res.anova) and len(res.figure["t_min"]) == 25
    ref = gc.analyse_time(sh, an, gc.BY_NONE, "STIM", "HR", bin_s=60, start_min=-10,
                          end_min=15, window_min=(0, 5), delta=False)
    for k, v in _curves(ref).items():
        assert np.allclose(_curves(res)[k], v, rtol=0, atol=1e-12)
    # A 'from' that rounds to 0 still starts one fine bin before it.
    assert gc.time_span(-0.02, 30)[0] == -gd.TC_BIN_S


def test_the_window_labels_name_the_bins_averaged(tmp_path):
    recs = {"206": (*_raw_recording(0, 1803.37), 1803.37),
            "207": (*_raw_recording(1, 1211.82), 1211.82)}
    sh, an = _raw_table(tmp_path, recs)
    res = gc.analyse_time(sh, an, gc.BY_NONE, "STIM", "HR", bin_s=120, start_min=-10,
                          end_min=15, window_min=(0, 5))
    # 2 min bins centred in 0–5 min: 0–2 and 2–4, so 0–4 min is tested.
    assert res.stats.iloc[0]["window (min)"] == "0–4"
    titles = [tb[0] for tb in res.prism]
    assert any("baseline vs 0–4 min" in t for t in titles), titles
    assert not any("0–5" in t for t in titles)
    paired = next(tb[1] for tb in res.prism if "baseline vs" in tb[0])
    assert "0–4 min" in paired.columns
    assert res.figure["window"] == (0, 4)
    assert dict(res.settings)["Test window (min)"] == "0–5 (bins 0–4)"
    assert any("Within group, 0–4 min" in str(t) for t in res.display["Test"])


def test_a_time_course_past_the_annotation_span_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(gd, "TC_ANCHOR_SPAN_S", (-300.0, 900.0))
    recs = {"206": (*_raw_recording(0, 1803.37), 1803.37),
            "207": (*_raw_recording(1, 1500.0), 1500.0)}
    sh, an = _raw_table(tmp_path, recs)
    ok = gc.analyse_time(sh, an, gc.BY_NONE, "STIM", "HR", bin_s=60, start_min=-5,
                         end_min=15, window_min=(0, 5))
    assert not any("±2.5 s" in n for n in ok.notes)
    res = gc.analyse_time(sh, an, gc.BY_NONE, "STIM", "HR", bin_s=60, start_min=-10,
                          end_min=15, window_min=(0, 5))
    note = next(n for n in res.notes if "±2.5 s" in n)
    # 207's stim is on the recording-start grid: exact over everything.
    assert "206" in note and "207" not in note and "-0.0833333 to +0.25 h" not in note
    assert "go past" in note and "Re-add" not in note
    assert np.isfinite(_curves(res)["206"]).all()


# ── time course: notes, titles and checks ───────────────────────────────

def _one_subject_twice(tmp_path):
    """The table of _table(n=4) with recording 101.smr renamed subject 100
    (one animal recorded twice), in every sheet."""
    sh = gd.read_sheets(_table(tmp_path, n=4))
    for name in (gd.SHEET_TIME, gd.SHEET_ANNOT, gd.SHEET_SEGMENTS, gd.SHEET_EVENTS,
                 gd.SHEET_WHOLE):
        df = sh[name]
        sh[name] = df.assign(subject=df["subject"].replace({"101": "100"}))
    sh.pop(gd.SHEET_ANIMALS, None)
    return sh, gd.animals_frame(sh)


def test_prism_keeps_both_recordings_of_one_subject(tmp_path):
    sh, an = _one_subject_twice(tmp_path)
    res = gc.analyse_time(sh, an, gc.BY_NONE, "Stim", bin_s=60, start_min=-4, end_min=8,
                          window_min=(0, 2))
    xy = next(tb[1] for tb in res.prism if len(tb) == 3)
    assert xy.shape[1] == 1 + 4 and xy.columns.is_unique
    assert {"100 (100.smr)", "100 (101.smr)"} <= set(xy.columns)
    paired = next(tb[1] for tb in res.prism if len(tb) == 2)
    assert len(paired) == 4 and paired.index.is_unique
    for other in (gc.analyse_segments(sh, an, gc.BY_NONE, "Pre", "Post", leave_out=False),
                  gc.analyse_events(sh, an, gc.BY_NONE, "Stim")):
        for _title, frame in other.prism:
            assert frame.index.is_unique, _title
        assert any("100 (101.smr)" in tb[1].index for tb in other.prism)


def test_two_recordings_of_one_subject_are_named_in_a_note(tmp_path):
    sh, an = _one_subject_twice(tmp_path)
    for res in (gc.analyse_time(sh, an, gc.BY_NONE, "Stim", bin_s=60, start_min=-4,
                                end_min=8, window_min=(0, 2)),
                gc.analyse_whole(sh, an, gc.BY_NONE)):
        note = next(n for n in res.notes if "separate animal" in n)
        assert "100 (100.smr, 101.smr)" in note
        # The statistics are unchanged: both recordings count.
        assert res.n_animals == 4


def test_unticked_animals_are_named_and_all_unticked_says_so(tmp_path):
    p = _table(tmp_path, n=4)
    sh = gd.read_sheets(p)
    an = gd.animals_frame(sh)
    an.loc[an["subject"] == "102", "use"] = False
    res = gc.analyse_time(sh, an, gc.BY_GROUP, "Stim", start_min=-4, end_min=8,
                          window_min=(0, 2))
    assert "Unticked in Animals, left out: 102" in res.notes
    # One subject with two recordings: the note says which one.
    sh2, an2 = _one_subject_twice(tmp_path)
    an2["sex"] = "M"
    an2.loc[an2["file"] == "101.smr", "use"] = False
    an2.loc[an2["file"] == "100.smr", "sex"] = ""
    res = gc.analyse_time(sh2, an2, gc.BY_SEX, "Stim", start_min=-4, end_min=8,
                          window_min=(0, 2))
    assert "Unticked in Animals, left out: 100 (101.smr)" in res.notes
    assert "100 (100.smr): sex unknown, left out" in res.notes
    an["use"] = False
    an["group"] = ""
    for f in (lambda: gc.analyse_time(sh, an, gc.BY_GROUP, "Stim", start_min=-4, end_min=8,
                                      window_min=(0, 2)),
              lambda: gc.analyse_whole(sh, an, gc.BY_GROUP)):
        with pytest.raises(ValueError, match="No animal is ticked"):
            f()


def test_a_second_matching_annotation_is_noted(tmp_path):
    p = _table(tmp_path, n=4)
    sh = gd.read_sheets(p)
    extra = sh[gd.SHEET_ANNOT].iloc[[0]].assign(t_start=700.0, label="STIM")
    sh[gd.SHEET_ANNOT] = pd.concat([sh[gd.SHEET_ANNOT], extra], ignore_index=True)
    res = gc.analyse_time(sh, gd.animals_frame(sh), gc.BY_NONE, "Stim", start_min=-4,
                          end_min=3, window_min=(0, 2))
    assert "2 “Stim” annotations, the 1st is used: 100" in res.notes


def test_smallest_mann_whitney_p_is_noted_for_small_groups(tmp_path):
    assert gs.smallest_mann_whitney_p(3, 3) == pytest.approx(0.1)
    assert gs.smallest_mann_whitney_p(4, 4) == pytest.approx(2 / 70)
    note = gc._small_n_note([3, 3], [[3, 3]])
    assert "n = 3: the smallest Wilcoxon p is 0.25" in note
    assert "3 vs 3: the smallest Mann-Whitney p is 0.1" in note
    assert gc._small_n_note([8, 8], [[8, 8]]) is None
    p = _table(tmp_path, n=6)
    sh = gd.read_sheets(p)
    res = gc.analyse_time(sh, gd.animals_frame(sh), gc.BY_GROUP, "Stim", start_min=-4,
                          end_min=8, window_min=(0, 2))
    assert any("3 vs 3: the smallest Mann-Whitney p is 0.1" in n for n in res.notes)
    whole = gc.analyse_whole(sh, gd.animals_frame(sh), gc.BY_GROUP)
    assert any("3 vs 3: the smallest Mann-Whitney p is 0.1" in n for n in whole.notes)


def test_a_test_window_past_the_range_or_on_the_baseline_is_noted(tmp_path):
    p = _table(tmp_path, n=4)
    sh = gd.read_sheets(p)
    an = gd.animals_frame(sh)
    ok = gc.analyse_time(sh, an, gc.BY_NONE, "Stim", start_min=-4, end_min=8,
                         window_min=(0, 2))
    assert not any(n.startswith("Test window") for n in ok.notes)
    past = gc.analyse_time(sh, an, gc.BY_NONE, "Stim", start_min=-4, end_min=8,
                           window_min=(6, 12))
    assert any("goes past the time course" in n for n in past.notes)
    over = gc.analyse_time(sh, an, gc.BY_NONE, "Stim", start_min=-4, end_min=8,
                           window_min=(-2, 2))
    assert any("overlaps the baseline" in n for n in over.notes)


def test_more_time_levels_than_animals_is_noted(tmp_path):
    p = _table(tmp_path, n=4)
    sh = gd.read_sheets(p)
    an = gd.animals_frame(sh)
    many = gc.analyse_time(sh, an, gc.BY_GROUP, "Stim", bin_s=30, start_min=-4, end_min=8,
                           window_min=(0, 2))
    assert len(many.anova) and "The ANOVA has 24 time levels for 4 animals" in " ".join(
        many.notes)
    few = gc.analyse_time(sh, an, gc.BY_GROUP, "Stim", bin_s=240, start_min=-4, end_min=8,
                          window_min=(0, 4))
    assert not any("time levels" in n for n in few.notes)


# ── metric families: the results table in sections ──────────────────────────

def _section_metrics(sections):
    return [(head, list(dict.fromkeys(rows["_metric"]))) for head, rows in sections]


def test_results_are_cut_by_metric_family_in_family_order(tmp_path):
    sh, an = _segments_table(tmp_path / "g.xlsx")
    an["group"] = ["A", "B", "A", "B"]
    res = gc.analyse_segments(sh, an, gc.BY_GROUP, "Pre 1", "Post 1", "all")
    # The table lists HR, SDNN, SD1, SD2 then Mean RR: on screen Mean RR joins HR.
    assert _section_metrics(gc.display_by_family(res.display)) == [
        ("Heart rate", [SEG_HR, "Mean RR (ms)"]), ("HRV time domain", ["SDNN (ms)"]),
        ("HRV non-linear", ["Poincaré SD1 (ms)", "Poincaré SD2 (ms)"])]
    # Every row of the table is in one section, none twice.
    rows = [i for _h, r in gc.display_by_family(res.display) for i in r.index]
    assert sorted(rows) == list(range(len(res.display)))
    # The workbook: no screen-only column in Results, a Family column in Statistics.
    out = tmp_path / "s.xlsx"
    gc.write_results(str(out), res, "g.xlsx")
    book = pd.read_excel(out, sheet_name=None)
    assert not [c for c in book["Results"].columns if str(c).startswith("_")]
    st = book["Statistics"]
    assert list(st.columns[:2]) == ["Metric", "Family"]
    assert dict(zip(st["Metric"], st["Family"]))["Mean RR (ms)"] == "Heart rate"


def test_all_pairs_sections_keep_each_pair_together(tmp_path):
    sh, an = _segments_table(tmp_path / "g.xlsx")
    res = gc.analyse_segments(sh, an, gc.BY_NONE, "", "", "all", pairs=gc.PAIRS_ALL)
    heads = [h for h, _r in gc.display_by_family(res.display)]
    assert heads == [f"{f} · {p}" for p in ("Post 1 vs Pre 1", "Post 5 vs Pre 5")
                     for f in ("Heart rate", "HRV time domain", "HRV non-linear")]


def test_one_metric_has_no_family_header(tmp_path):
    p = _table(tmp_path, n=4)
    sh = gd.read_sheets(p)
    an = gd.animals_frame(sh)
    res = gc.analyse_time(sh, an, gc.BY_GROUP, "Stim", "HR", bin_s=30, start_min=-4,
                          end_min=8, window_min=(0, 2))
    (head, rows), = gc.display_by_family(res.display)
    assert head == "" and len(rows) == len(res.display)
    assert set(res.display["_metric"]) == {"HR"}
    # Post-hoc rows (3 groups) carry the metric of the test above them.
    (tmp_path / "six").mkdir()
    p6 = _table(tmp_path / "six", n=6)
    sh6 = gd.read_sheets(p6)
    an6 = gd.animals_frame(sh6)
    an6["group"] = ["Control", "CNO", "Sal"] * 2
    whole = gc.analyse_whole(sh6, an6, gc.BY_GROUP)
    post = whole.display.loc[whole.display["Test"].str.strip().str.startswith("post-hoc")]
    assert len(post) == 3 and set(post["_metric"]) == {"Mean HR (60000/mean NN) (bpm)"}


# ── Data analysed rows (beats, duration, blocks used): described, not tested ─

def _segments_with_counts(path):
    """Four animals (groups A / B), two segments, the outcomes and the amount
    of data (Beats, Duration, Blocks used) of each segment."""
    from ecg.io.group_table import append_group_row, group_row
    metrics = [("hr_mean", "Mean HR (60000/mean NN)", "bpm"), ("sdnn", "SDNN", "ms"),
               ("sd1", "Poincaré SD1", "ms"), ("n_beats", "Beats", ""),
               ("dur", "Duration", "s"), ("blocks", "Blocks used", "n")]
    for i in range(6):
        segs = [{"label": lbl, "t_lo": 60.0 * k, "t_hi": 60.0 * k + 60, "duration_s": 60.0,
                 "hr_mean": 500 + 3 * i + (12 + i) * k * (1 + i % 2), "sdnn": 3 + 0.4 * i - k,
                 "sd1": 2 + 0.1 * i + 0.3 * k * (i % 2), "n_beats": 500 + 7 * i + 40 * k,
                 "dur": 60.0, "blocks": 6 - k * (i % 2)}
                for k, lbl in enumerate(("Pre", "Post"))]
        append_group_row(str(path), group_row(f"M{i + 1}", f"M{i + 1}.mat", segs, metrics))
    sh = gd.read_sheets(str(path))
    an = gd.animals_frame(sh)
    an["group"] = ["A", "B"] * 3
    return sh, an


OUTCOMES = [SEG_HR, "SDNN (ms)", "Poincaré SD1 (ms)"]


def test_data_analysed_rows_are_described_not_tested(tmp_path):
    sh, an = _segments_with_counts(tmp_path / "g.xlsx")
    names = gs.metric_names(sh[gd.SHEET_SEGMENTS], "Pre", "Post")
    qc = [m for m in names if m not in OUTCOMES]
    assert qc and all(gc.is_descriptive(m) for m in qc)
    only = gc.analyse_segments(sh, an, gc.BY_GROUP, "Pre", "Post", OUTCOMES)
    both = gc.analyse_segments(sh, an, gc.BY_GROUP, "Pre", "Post", "all")
    a = only.stats.set_index("Metric")
    b = both.stats.set_index("Metric")
    holm_cols = ["between p (Holm)", "A Wilcoxon p (Holm)", "B Wilcoxon p (Holm)"]
    # The Holm p of the outcomes is the same with the data rows in the table.
    for m in OUTCOMES:
        for c in holm_cols:
            assert b.loc[m, c] == pytest.approx(a.loc[m, c], nan_ok=True)
    assert any(a.loc[m, "between p (Holm)"] > a.loc[m, "between p"] for m in OUTCOMES)
    # The data rows: values per group, no p.
    for m in qc:
        assert not b.loc[m, "Tested"]
        assert all(math.isnan(b.loc[m, c]) for c in holm_cols + ["between p", "A Wilcoxon p"])
        assert math.isfinite(b.loc[m, "A Pre mean"])
    rows = both.display.loc[both.display["_metric"].isin(qc)]
    assert len(rows) and set(rows["p"]) == {gc.DESCRIPTIVE}
    assert not any("p =" in str(v) for v in rows["A"]) and set(rows["p Holm"]) == {""}
    assert rows["_p"].isna().all()
    assert not len(both.anova) or not set(both.anova["Metric"]) & set(qc)
    # The workbook says it too.
    out = tmp_path / "s.xlsx"
    gc.write_results(str(out), both, "g.xlsx")
    st = pd.read_excel(out, sheet_name="Statistics").set_index("Metric")
    assert st.loc[qc, "between p (Holm)"].isna().all()
    assert st.loc[OUTCOMES, "between p (Holm)"].notna().all()


def test_data_analysed_rows_are_not_tested_in_whole_recording_or_compare_workbook(tmp_path):
    rng = np.random.default_rng(4)
    whole = pd.DataFrame({
        "subject": [f"M{i}" for i in range(6)], "file": [f"M{i}.mat" for i in range(6)],
        SEG_HR: rng.normal(500, 10, 6), "SDNN (ms)": rng.normal(5, 1, 6),
        "Beats": [1000 + 50 * i for i in range(6)], "Analysed duration (s)": [600.0] * 6})
    an = _animals([(f"M{i}", "", "AB"[i % 2], True, f"M{i}.mat") for i in range(6)])
    sh = {gd.SHEET_WHOLE: whole}
    res = gc.analyse_whole(sh, an, gc.BY_GROUP, "all")
    ref = gc.analyse_whole(sh, an, gc.BY_GROUP, [SEG_HR, "SDNN (ms)"])
    st, st0 = res.stats.set_index("Metric"), ref.stats.set_index("Metric")
    for m in (SEG_HR, "SDNN (ms)"):
        assert st.loc[m, "between p (Holm)"] == pytest.approx(st0.loc[m, "between p (Holm)"])
    for m in ("Beats", "Analysed duration (s)"):
        assert math.isnan(st.loc[m, "between p"]) and math.isnan(st.loc[m, "between p (Holm)"])
    assert set(res.display.loc[res.display["_metric"] == "Beats", "p"]) == {gc.DESCRIPTIVE}
    # Compare segments' statistics workbook (paired_stats): no p, not in Holm.
    sh2, _an = _segments_with_counts(tmp_path / "g.xlsx")
    seg = sh2[gd.SHEET_SEGMENTS]
    p_all = gs.paired_stats(seg, "Pre", "Post", "all").set_index("Metric")
    p_out = gs.paired_stats(seg, "Pre", "Post", OUTCOMES).set_index("Metric")
    for m in OUTCOMES:
        assert p_all.loc[m, "Wilcoxon p (Holm)"] == pytest.approx(
            p_out.loc[m, "Wilcoxon p (Holm)"], nan_ok=True)
    qc = [m for m in p_all.index if m not in OUTCOMES]
    assert qc and p_all.loc[qc, ["Wilcoxon p", "Wilcoxon p (Holm)"]].isna().all().all()
    frames = gs._report_frames(seg, "Pre", "Post", p_all.reset_index(), [])
    about = dict(zip(frames["About"]["Item"], frames["About"]["Value"]))
    assert "Beats" not in about["Metrics tested"]
    assert "Beats" in about["Described, not tested"]


def test_hedges_g_is_nan_when_both_groups_have_no_spread_up_to_rounding():
    """0.1 + 0.2 style rounding gave a pooled SD of ~1e-16 and g = 1e15."""
    import math
    from ecg.io.group_compare import hedges_g
    a = [0.1 + 0.2, 0.3, 0.30000000000000004]
    b = [0.6, 0.6, 0.6000000000000001]
    assert math.isnan(hedges_g(a, b))
    assert math.isfinite(hedges_g([1.0, 2.0, 3.0], [2.0, 3.0, 5.0]))
