"""RR tachograms of Compare segments (draw_rr_tachograms): one superimposed
plot, except in Pairs mode (2 vs 1, 4 vs 3, 6 vs 5), where each pair gets
its own panel -- segments of different pairs (and durations) are never
drawn together."""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from ecg.core.segment_plan import PAIRS, PAIRWISE, reference_comparators
from ecg.app.compare_helpers import draw_rr_tachograms, tachogram_groups

COLORS = ["C0", "C1", "C2", "C3", "C4", "C5"]


def _segs(durations):
    out = []
    for k, d in enumerate(durations):
        t = 1000.0 * k + np.arange(0.0, d, 0.12)
        out.append({"label": f"S{k + 1}", "t_rr": t, "rr_ms": 120.0 + k + 0 * t})
    return out


def _labels(ax):
    return [ln.get_label() for ln in ax.get_lines() if not ln.get_label().startswith("_")]


def test_groups():
    assert tachogram_groups(4, reference_comparators(4, PAIRS)) == [(0, 1), (2, 3)]
    assert tachogram_groups(5, reference_comparators(5, PAIRS)) == [(0, 1), (2, 3), (4,)]
    assert tachogram_groups(6, reference_comparators(6, PAIRS)) == [(0, 1), (2, 3), (4, 5)]
    for ref in (0, 2, PAIRWISE):
        assert tachogram_groups(4, reference_comparators(4, ref)) == [(0, 1, 2, 3)]
    assert tachogram_groups(3, None) == [(0, 1, 2)]
    assert tachogram_groups(0, []) == []


def test_pairs_one_panel_per_pair():
    segs = _segs([60, 60, 600, 600, 300, 300])
    names = ["Pre Stim 1min", "Post Stim 1min", "Pre Stim 10min", "Post Stim 10min",
             "Pre 5", "Post 5"]
    fig = Figure()
    axes = draw_rr_tachograms(fig, segs, COLORS, reference_comparators(6, PAIRS), names)
    assert len(axes) == len(fig.axes) == 3
    assert [a.get_title(loc="left") for a in axes] == [
        "Post Stim 1min vs Pre Stim 1min", "Post Stim 10min vs Pre Stim 10min",
        "Post 5 vs Pre 5"]
    assert [_labels(a) for a in axes] == [names[0:2], names[2:4], names[4:6]]
    # Each panel's own x range: time within its segments.
    assert axes[0].get_xlim()[1] < 100 and axes[1].get_xlim()[1] > 500
    # The same colours as the superimposed view, one RR scale for all.
    assert [ln.get_color() for ln in axes[1].get_lines()[::2]] == ["C2", "C3"]
    assert axes[0].get_shared_y_axes().joined(axes[0], axes[2])


def test_pairs_odd_segment_alone():
    segs = _segs([60, 60, 120, 120, 300])
    names = ["A", "B", "C", "D", "E"]
    fig = Figure()
    axes = draw_rr_tachograms(fig, segs, COLORS, reference_comparators(5, PAIRS), names)
    assert [a.get_title(loc="left") for a in axes] == ["B vs A", "D vs C", "E"]
    assert [_labels(a) for a in axes] == [["A", "B"], ["C", "D"], ["E"]]


def test_other_modes_single_superimposed_plot():
    segs = _segs([60, 60, 600, 600])
    for comps in (None, reference_comparators(4), reference_comparators(4, 1),
                  reference_comparators(4, PAIRWISE)):
        fig = Figure()
        axes = draw_rr_tachograms(fig, segs, COLORS, comps)
        assert len(axes) == len(fig.axes) == 1
        assert _labels(axes[0]) == ["S1", "S2", "S3", "S4"]
        assert "superimposed" in axes[0].get_title(loc="left")
    # Two segments: Pairs and "against the first" are the same comparators.
    fig = Figure()
    assert len(draw_rr_tachograms(fig, _segs([60, 60]), COLORS,
                                  reference_comparators(2, PAIRS))) == 1
