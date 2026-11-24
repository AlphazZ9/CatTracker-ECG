# -*- coding: utf-8 -*-
"""
ecg.ui.group_tab_figure
-----------------------
The figure of the Group tab, from GroupResult.figure (ecg.io.group_compare):

* time: mean ± SEM per group and bin (a band), the time 0 dashed, the test
  window shaded, optionally one thin line per animal;
* paired (Pre/Post segments, Event-locked): one panel per group, one line
  per animal from the reference to the condition, the group mean thick;
* dots (Whole recording): one column per group, a dot per animal, the mean
  ± SEM.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from ecg.ui.plots import style_axes
from ecg.ui.theme import PLOT

# Blue / orange first: told apart by lightness too, not hue alone.
GROUP_COLORS = ("#2563eb", "#ea580c", "#059669", "#7c3aed", "#db2777", "#0891b2")


def group_color(i: int) -> str:
    return GROUP_COLORS[i % len(GROUP_COLORS)]


def spread_labels(values: "Sequence[float]", gap: float) -> "list[float]":
    """Heights for labels at *values* so that no two are closer than *gap*:
    labels that would overlap are pushed apart, each pushed block centred on
    the values it labels; the order is kept and labels far apart stay put."""
    v = np.asarray(values, dtype=float)
    out = v.copy()
    ok = np.isfinite(v)
    if ok.sum() < 2 or not gap > 0:
        return out.tolist()
    idx = np.flatnonzero(ok)[np.argsort(v[ok], kind="stable")]
    # Blocks of labels laid out gap apart around the mean of what they
    # label; a block that reaches into the one below merges with it.
    blocks: "list[list[float]]" = []          # [first height, sum of values, count]
    for y in v[idx]:
        blocks.append([float(y), float(y), 1])
        while len(blocks) > 1:
            lo, hi = blocks[-2], blocks[-1]
            if hi[0] >= lo[0] + gap * lo[2] - 1e-9 * gap:
                break
            n = lo[2] + hi[2]
            tot = lo[1] + hi[1]
            blocks[-2:] = [[tot / n - gap * (n - 1) / 2, tot, n]]
    k = 0
    for first, _tot, n in blocks:
        out[idx[k:k + n]] = first + gap * np.arange(n)
        k += n
    return out.tolist()


def label_line_ends(ax, items: "Sequence[tuple[str, float]]", x: float, fontsize: float = 7,
                    color: str = "", dx: float = 0.04) -> None:
    """Name tags (name, y) at the end of lines at *x*, nudged apart
    vertically where they would overlap (call after every line is drawn, so
    that the y range is final)."""
    if not items:
        return
    y0, y1 = ax.get_ylim()
    height_px = max(1.0, ax.bbox.height)
    # One line of text, in data units (the axes shrink a little once laid out).
    gap = abs(y1 - y0) * fontsize * 1.25 * ax.figure.dpi / 72.0 / height_px * 1.15
    ys = spread_labels([y for _n, y in items], gap)
    for (name, y), yt in zip(items, ys):
        ax.annotate(name, (x, y), xytext=(x + dx, yt), textcoords="data", va="center",
                    fontsize=fontsize, color=color or PLOT.get("muted", "#888"),
                    annotation_clip=False)


def _empty(fig, text: str) -> bool:
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
            color=PLOT.get("muted", "#888"))
    ax.set_axis_off()
    return False


def draw_group_tab_figure(fig, res: Any, metric: Optional[str] = None,
                          show_animals: bool = False, title: str = "") -> bool:
    """Draw *res* (a GroupResult) on *fig*; *metric* chooses the metric of a
    paired or dots figure (default: the first one tested)."""
    data = getattr(res, "figure", None) or {}
    kind = data.get("kind")
    if kind == "time":
        return _draw_time(fig, data, res.groups, show_animals, title)
    poincare = data.get("poincare") or {}
    if kind == "paired" and metric in poincare:
        from ecg.ui.group_figure import draw_group_figure
        rows, a, b, names = poincare[metric]
        return draw_group_figure(fig, rows, a, b, names, title=title)
    if kind in ("paired", "dots"):
        m = metric if metric in data.get("values", {}) else data.get("metric", "")
        if not m:
            return _empty(fig, "No metric to draw")
        if kind == "paired":
            return _draw_paired(fig, data, res.groups, m, title)
        return _draw_dots(fig, data, res.groups, m, title)
    return _empty(fig, "Open a group table and choose an analysis")


def _draw_time(fig, data: dict, groups: "Sequence[str]", show_animals: bool, title: str) -> bool:
    t = np.asarray(data["t_min"], dtype=float)
    muted = PLOT.get("muted", "#888")
    ax = fig.add_subplot(111)
    style_axes(ax)
    w0, w1 = data["window"]
    ax.axvspan(w0, w1, color="#2563eb", alpha=0.07, lw=0, zorder=0)
    ax.axvline(0, color=PLOT.get("text", "#222"), ls="--", lw=1, zorder=1)
    ax.text(0, 1.0, f" {data.get('stim', 'stim')}", transform=ax.get_xaxis_transform(),
            va="top", ha="left", fontsize=9, color=muted)
    if str(data.get("label", "")).startswith("Δ"):
        ax.axhline(0, color=muted, lw=0.6, zorder=1)
    any_line = False
    for i, g in enumerate(groups):
        c = data["curves"].get(g)
        if not c or c["n"] == 0:
            continue
        col = group_color(i)
        if show_animals:
            for v in c["animals"].values():
                ax.plot(t, v, color=col, lw=0.7, alpha=0.35, zorder=2)
        m, s = np.asarray(c["mean"], float), np.asarray(c["sem"], float)
        ax.fill_between(t, m - s, m + s, color=col, alpha=0.18, lw=0, zorder=2)
        ls = "-" if i % 2 == 0 else (0, (6, 3))
        ax.plot(t, m, color=col, lw=2.2, ls=ls, zorder=3, label=f"{g} (n = {c['n']})")
        any_line = True
    if not any_line:
        fig.delaxes(ax)
        return _empty(fig, "No animal to draw")
    ax.set_xlabel("Time from the time 0 (min)", fontsize=9)
    ax.set_ylabel(str(data.get("label", "")), fontsize=9)
    # The theme's text colour, as the app's other legends: matplotlib's
    # default is near-black, unreadable on the dark theme.
    ax.legend(frameon=False, fontsize=9, loc="best", labelcolor=PLOT.get("text", "#222"))
    ax.set_title((title + "  ·  " if title else "")
                 + f"mean ± SEM per {data.get('bin_s', 0):g} s bin · shaded: test window "
                   f"{w0:g}–{w1:g} min", fontsize=9, color=muted)
    return True


def _by_group(data: dict, groups: "Sequence[str]", metric: str) -> "dict[str, list]":
    vals = data["values"].get(metric, {})
    gmap = data["gmap"]
    return {g: [(k[0], vals[k]) for k in gmap if gmap[k] == g and k in vals] for g in groups}


def _draw_paired(fig, data: dict, groups: "Sequence[str]", metric: str, title: str) -> bool:
    by = _by_group(data, groups, metric)
    if not any(by.values()):
        return _empty(fig, f"No animal with both values of {metric}")
    na, nb = (data.get("names_by") or {}).get(metric) or data.get("names", ("A", "B"))
    muted = PLOT.get("muted", "#888")
    axes = fig.subplots(1, len(groups), sharey=True, squeeze=False)[0]
    for i, (ax, g) in enumerate(zip(axes, groups)):
        style_axes(ax)
        col = group_color(i)
        pts = by[g]
        for name, (a, b) in pts:
            ax.plot([0, 1], [a, b], "-o", color=col, ms=4, lw=1, alpha=0.45)
        if pts:
            ma = float(np.mean([p[1][0] for p in pts]))
            mb = float(np.mean([p[1][1] for p in pts]))
            ax.plot([0, 1], [ma, mb], "-", color=PLOT.get("text", "#222"), lw=3, zorder=5)
        ax.set_xticks([0, 1], [na, nb], fontsize=9)
        ax.set_xlim(-0.3, 1.45)
        ax.set_title(f"{g} (n = {len(pts)})", fontsize=10, color=col)
        if i == 0:
            ax.set_ylabel(metric, fontsize=9)
    # The name tags once the shared y range is set, apart where they would overlap.
    for ax, g in zip(axes, groups):
        label_line_ends(ax, [(name, b) for name, (_a, b) in by[g]], 1.0, color=muted)
    fig.suptitle((title + "  ·  " if title else "")
                 + "one line per animal · thick: group mean", fontsize=8, color=muted)
    return True


def _draw_dots(fig, data: dict, groups: "Sequence[str]", metric: str, title: str) -> bool:
    by = _by_group(data, groups, metric)
    if not any(by.values()):
        return _empty(fig, f"No animal with {metric}")
    muted = PLOT.get("muted", "#888")
    ax = fig.add_subplot(111)
    style_axes(ax)
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        col = group_color(i)
        v = np.array([p[1] for p in by[g]], dtype=float)
        if not len(v):
            continue
        x = i + rng.uniform(-0.12, 0.12, len(v))
        ax.plot(x, v, "o", color=col, ms=6, alpha=0.75)
        for xi, (name, vi) in zip(x, by[g]):
            ax.annotate(name, (xi, vi), xytext=(6, -3), textcoords="offset points",
                        fontsize=7, color=muted)
        m = float(v.mean())
        s = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        ax.errorbar(i + 0.32, m, yerr=s, fmt="_", color=PLOT.get("text", "#222"),
                    ms=18, mew=2.5, capsize=5, lw=1.5)
    ax.set_xticks(range(len(groups)),
                  [f"{g}\n(n = {len(by[g])})" for g in groups], fontsize=9)
    ax.set_xlim(-0.6, len(groups) - 0.2)
    ax.set_ylabel(metric, fontsize=9)
    ax.set_title((title + "  ·  " if title else "") + "one dot per animal · bar: mean ± SEM",
                 fontsize=9, color=muted)
    return True
