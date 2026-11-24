"""Group tab and Compare segments, final live check: the whole Results table
is reachable, the family headers carry no zebra stripe, the group figures'
legends follow the theme (exports on white), the animal tags at the line
ends do not overlap, and a theme toggle keeps the analysis window entries."""
from __future__ import annotations

import tkinter as tk

import numpy as np
import pandas as pd
import pytest

import customtkinter as ctk  # type: ignore[import-untyped]
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from ecg.io import group_compare as gc
from ecg.io import group_data as gd
from ecg.ui import theme
from ecg.ui.compare_segments import own_line_stripes
from ecg.ui.group_tab_figure import draw_group_tab_figure, spread_labels
from ecg.ui.session_controller import restore_entry
from tests.io.test_group_compare import _segments_with_counts, _table
from tests.ui.test_group_tab import _page, _walk, root  # noqa: F401  (fixture)


def _text(frame) -> tk.Text:
    return next(w for w in _walk(frame) if isinstance(w, tk.Text))


def test_the_results_box_shows_every_row_of_a_long_table(root, tmp_path):
    # 40 whole-recording metrics, 4 animals in two groups: 40+ lines.
    p = str(tmp_path / "g.xlsx")
    rng = np.random.default_rng(1)
    for i in range(4):
        whole = pd.DataFrame([{f"SDNN{k} (ms)": float(rng.normal(5, 1)) for k in range(40)}
                              | {"Beats": 1000.0 + i}])
        gd.add_recording(p, f"M{i}", "", f"M{i}.mat", {gd.SHEET_WHOLE: whole})
    an = gd.animals_frame(gd.read_sheets(p))
    an["group"] = ["A", "B", "A", "B"]
    gd.save_animals(p, an)
    _app, frame = _page(root, {"path": p, "analysis": gc.AN_WHOLE, "by": gc.BY_GROUP,
                               "all": True})
    txt = _text(frame)
    content = txt.get("1.0", "end-1c")
    n_lines = int(txt.index("end-1c").split(".")[0])
    assert "DATA ANALYSED" in content and "SDNN39" in content and n_lines > 40, content
    # No cap: the box is as tall as the table (the page scrolls).
    assert int(txt.cget("height")) >= n_lines
    assert not root.errors


def test_compare_table_header_rows_have_no_stripe(root):
    t = tk.Text(root)
    t.tag_configure("stripe", background="#eee")
    # As the Compare segments table: each row starts with its newline.
    t.insert("end", "Metric", "head")
    t.insert("end", "\nHEART RATE", "family")
    t.insert("end", "\nMean HR\t500", "stripe")
    t.insert("end", "\nSDNN\t3")
    t.insert("end", "\nHRV NON-LINEAR", "family")
    t.insert("end", "\nSD1\t2", "stripe")
    own_line_stripes(t)
    for ln, striped in ((1, False), (2, False), (3, True), (4, False), (5, False)):
        assert ("stripe" in t.tag_names(f"{ln}.end")) is striped, ln
    t.destroy()


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


def test_a_theme_rebuild_puts_the_window_placeholders_back(root):
    e0 = ctk.CTkEntry(root, placeholder_text="0")
    e1 = ctk.CTkEntry(root, placeholder_text="4206.9")
    e2 = ctk.CTkEntry(root, placeholder_text="end")
    restore_entry(e0, "")
    restore_entry(e1, "")
    restore_entry(e2, "120")
    assert e0._entry.get() == "0" and e0.get() == ""
    assert e1._entry.get() == "4206.9" and e1.get() == ""
    assert e2.get() == "120"


def test_export_names_have_no_slash():
    from ecg.ui.tab_group import _file_tag
    assert _file_tag("Pre/Post segments") == "pre_post"
    assert _file_tag("Time course") == "time"
    assert "/" not in _file_tag("Event-locked")


def test_group_poincare_name_tags_do_not_overlap():
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
    from matplotlib.figure import Figure
    from ecg.ui.group_figure import draw_group_figure, _RR, _SD1, _SD2
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
