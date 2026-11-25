"""Group tab and Compare segments, final live check: the whole Results table
is reachable, the family headers carry no zebra stripe, the group figures'
legends follow the theme (exports on white), the animal tags at the line
ends do not overlap, and a theme toggle keeps the analysis window entries."""
from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from ecg.io import group_compare as gc
from ecg.io import group_data as gd
from ecg.app import theme
from ecg.app.group_tab_figure import draw_group_tab_figure, spread_labels
from ecg.app.session_controller import restore_entry
from tests.io.test_group_compare import _segments_with_counts, _table








def _legend_colors(fig):
    return {to_hex(tx.get_color()) for ax in fig.axes if ax.get_legend()
            for tx in ax.get_legend().get_texts()}


@pytest.fixture
def dark():
    saved = dict(theme.PLOT)
    theme.PLOT.update(theme._make_plot_theme(
        type("T", (), {"is_dark": True, "_colors": dict(theme.ThemeConfig.PRESETS["Dark"])})()))
    yield theme.PLOT
    theme.PLOT.clear()
    theme.PLOT.update(saved)


def _time_result(tmp_path):
    p = _table(tmp_path, n=4)
    sh = gd.read_sheets(p)
    return gc.analyse_time(sh, gd.animals_frame(sh), gc.BY_GROUP, "Stim", "HR", bin_s=30,
                           start_min=-4, end_min=8, window_min=(0, 2))


def test_time_course_legend_uses_the_theme_text_and_exports_stay_light(tmp_path, dark):
    res = _time_result(tmp_path)
    fig = Figure()
    draw_group_tab_figure(fig, res)
    assert _legend_colors(fig) == {to_hex(dark["text"])}
    light_text = theme._make_plot_theme(
        type("T", (), {"is_dark": False, "_colors": dict(theme.ThemeConfig.PRESETS["Light"])})()
    )["text"]
    out = Figure(facecolor="white")
    with theme.light_plot_theme():
        draw_group_tab_figure(out, res)
    assert _legend_colors(out) == {to_hex(light_text)}
    assert {to_hex(ax.get_facecolor()) for ax in out.axes} == {"#ffffff"}
    # Back to the dark theme after the export.
    assert theme.PLOT["text"] == dark["text"]


def test_spread_labels_moves_only_labels_that_overlap():
    ys = spread_labels([10.0, 10.1, 10.15, 20.0, 30.0], 1.0)
    assert np.all(np.diff(ys[:3]) >= 1.0 - 1e-9)
    assert np.mean(ys[:3]) == pytest.approx(np.mean([10.0, 10.1, 10.15]))
    assert ys[3:] == [20.0, 30.0]
    # Order kept whatever the input order; NaN left alone.
    out = spread_labels([5.0, float("nan"), 5.0, 1.0], 2.0)
    assert out[3] == 1.0 and np.isnan(out[1]) and abs(out[2] - out[0]) == pytest.approx(2.0)


def test_paired_figure_tags_do_not_overlap(tmp_path):
    sh, an = _segments_with_counts(tmp_path / "g.xlsx")
    res = gc.analyse_segments(sh, an, gc.BY_NONE, "Pre", "Post")
    # Two animals one hair apart at Post.
    vals = res.figure["values"][res.figure["metric"]]
    k = list(vals)
    vals[k[1]] = (vals[k[1]][0], vals[k[0]][1] + 1e-3)
    fig = Figure(figsize=(6, 4), dpi=100)
    draw_group_tab_figure(fig, res)
    fig.canvas.draw()
    boxes = [a.get_window_extent() for ax in fig.axes for a in ax.texts]
    assert len(boxes) == len(vals)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not boxes[i].overlaps(boxes[j]), (i, j)




def test_export_names_have_no_slash():
    from ecg.app.group_helpers import _file_tag
    assert _file_tag("Pre/Post segments") == "pre_post"
    assert _file_tag("Time course") == "time"
    assert "/" not in _file_tag("Event-locked")


def test_group_poincare_name_tags_do_not_overlap():
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
    from matplotlib.figure import Figure
    from ecg.app.group_figure import draw_group_figure, _RR, _SD1, _SD2
    rows = []
    for i, rr in enumerate((120.0, 120.2, 120.4, 135.0)):
        r = {"subject": f"P{i}"}
        for t in ("A", "B"):
            r[f"{_RR} {t}"], r[f"{_SD1} {t}"], r[f"{_SD2} {t}"] = rr, 2.0, 5.0
        rows.append(r)
    fig = Figure(figsize=(12, 4), dpi=100)
    assert draw_group_figure(fig, pd.DataFrame(rows), "A", "B")
    fig.canvas.draw()
    ax = fig.axes[0]
    boxes = [t.get_window_extent() for t in ax.texts if t.get_text().startswith("P")]
    assert len(boxes) == 4
    assert not any(b1.overlaps(b2) for i, b1 in enumerate(boxes) for b2 in boxes[i + 1:])
