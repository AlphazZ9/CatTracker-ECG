# -*- coding: utf-8 -*-
"""
ecg.figures.cohort_trends
-------------------------
Cohort Trends: heart rate, SDNN and RMSSD of every recording saved in the
registry, ordered by save time and coloured by project ("Ungrouped" in
grey). Moved unchanged from the Tk CohortTrendsDialog._draw, so both
interfaces draw the same figure.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from matplotlib.figure import Figure

from ecg.ui import theme as _theme


def _palette() -> "list[str]":
    t = _theme
    return [t.BLUE, t.GREEN, t.ORANGE, t.PURPLE, t.TEAL, t.PINK, t.CYAN, t.AMBER, t.RED]


def usable_rows(rows: "list[dict]") -> "list[dict]":
    """The registry rows with a save time (the others cannot be placed)."""
    return [r for r in rows if r.get("saved_at")]


def draw_cohort_trends(fig: Figure, rows: "list[dict]") -> None:
    PLOT, MUTED = _theme.PLOT, _theme.MUTED
    fig.patch.set_facecolor(PLOT["bg"])

    def _parse(ts: str):
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            return None

    parsed = [(r, _parse(r.get("saved_at", ""))) for r in rows]
    parsed = [(r, dt) for r, dt in parsed if dt is not None]
    parsed.sort(key=lambda item: item[1])

    groups: "dict[str, list[tuple[dict, Any]]]" = {}
    for r, dt in parsed:
        key = (r.get("project_name") or "").strip() or "Ungrouped"
        groups.setdefault(key, []).append((r, dt))
    names = sorted(k for k in groups if k != "Ungrouped")
    colors = {name: _palette()[i % len(_palette())]
              for i, name in enumerate(names)}
    colors["Ungrouped"] = MUTED

    specs = [
        ("hr_mean", "Heart rate (bpm)", "HR"),
        ("sdnn",    "SDNN (ms)",        "SDNN"),
        ("rmssd",   "RMSSD (ms)",       "RMSSD"),
    ]
    axes = fig.subplots(3, 1, sharex=True)
    for ax, (key, ylabel, short) in zip(axes, specs):
        n_present = 0
        for name, items in groups.items():
            pts = [(dt, r[key]) for r, dt in items if r.get(key) is not None]
            n_present += len(pts)
            if not pts:
                continue
            xs, ys = zip(*pts)
            col = colors[name]
            if len(pts) > 1:
                ax.plot(xs, ys, "-", color=col, alpha=0.5, lw=1.2, zorder=2)
            ax.scatter(xs, ys, color=col, s=28, zorder=3, label=name,
                      edgecolors=PLOT["bg"], linewidths=0.6)
        # Most existing registry rows predate SDNN/RMSSD being saved at
        # all -- say so, instead of leaving a sparse plot that reads as
        # a bug rather than old data waiting to be backfilled.
        n_missing = len(rows) - n_present
        if n_missing:
            ax.text(0.99, 0.05,
                    f"{short} missing for {n_missing} of {len(rows)} "
                    "recordings — save session again to backfill",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color=PLOT["muted"], style="italic")
        ax.set_facecolor(PLOT["axes"])
        ax.set_ylabel(ylabel, color=PLOT["muted"], fontsize=9)
        ax.tick_params(colors=PLOT["muted"], labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(PLOT["border"])
        ax.grid(True, color=PLOT["grid"], lw=0.5, alpha=0.6)

    # One shared legend on the top subplot -- the colour mapping is
    # identical across all three, repeating it three times would just
    # be noise. Anchored above the axes (not "upper left" inside it) so
    # it never overprints the earliest data points.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        axes[0].legend(by_label.values(), by_label.keys(),
                      framealpha=0, loc="lower left",
                      bbox_to_anchor=(0.0, 1.02, 1.0, 0.08), mode="expand",
                      fontsize=7, ncol=min(4, len(by_label)))

    axes[-1].set_xlabel("Saved at", color=PLOT["muted"], fontsize=9)
    fig.autofmt_xdate(rotation=25)
