# -*- coding: utf-8 -*-
"""
ecg.app.compare_helpers
-----------------------
Compare segments without its widgets: the setup kept in sessions
(normalize_compare_setup), the segment colours, the RR tachograms, the
group table's segment choices and the group Poincaré data. Moved unchanged
from the Tk compare_segments.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from ecg.app.theme import BLUE, GREEN, ORANGE_DARK, PLOT, PURPLE, RED, TEAL
from ecg.app.plots import style_axes
from ecg.core.models import MouseECG
from ecg.core.segment_plan import (MAX_SEGMENTS, PAIRS, PAIRWISE, comparators_reference,
                                   comparison_pairs)

log = logging.getLogger("ecg")


_VIEWS = ("rr", "poincare")


def _fmt_s(v: "Any") -> str:
    """A time in seconds for an entry or a message: millisecond precision,
    trailing zeros dropped ("1808.183", "2000", "0.5"). Unlike ``:g`` it
    keeps every digit before the point, so 12345.678 stays 12345.678."""
    s = f"{float(v):.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _finite(v: "Any") -> "Optional[float]":
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _cell(v: "Any") -> "Any":
    """A From / To cell: its number, else the text typed ("" for none)."""
    f = _finite(v)
    return f if f is not None else ("" if v is None else str(v))


def normalize_compare_setup(raw: "Any") -> "Optional[dict]":
    """A Compare segments setup as saved in a session (or snapshot before a
    UI rebuild), checked: {"rows": [[label, from, to], ...] (1 to
    MAX_SEGMENTS, from / to a number or, typed but not valid, the text),
    "block_s" (text), "ref" (a segment index, PAIRWISE or PAIRS), "view"
    ("rr" / "poincare"), "jitter" (bool), "run": the windows and block
    length of the comparison shown ({"rows": [[label, lo, hi]...],
    "block_s"}) or None, "ran": whether there is one, "stim": the Stim at
    entry, text, "" when empty}. None when *raw* is not one (an older session: the
    tab keeps its defaults)."""
    if not isinstance(raw, dict):
        return None
    rows = []
    for r in raw.get("rows") or []:
        if isinstance(r, (list, tuple)) and len(r) == 3:
            rows.append(["" if r[0] is None else str(r[0]), _cell(r[1]), _cell(r[2])])
    rows = rows[:MAX_SEGMENTS]
    ref = raw.get("ref", 0)
    if isinstance(ref, bool) or not (
            (isinstance(ref, int) and 0 <= ref < MAX_SEGMENTS) or ref in (PAIRWISE, PAIRS)):
        ref = 0
    view = raw.get("view") if raw.get("view") in _VIEWS else "rr"
    blk = raw.get("block_s")
    if isinstance(blk, str):
        block_s = blk
    elif _finite(blk) is not None:
        block_s = _fmt_s(_finite(blk))
    else:
        block_s = f"{MouseECG.SEGMENT_BLOCK_S:g}"
    run = None
    rr = raw.get("run")
    if isinstance(rr, dict):
        r_rows: "list[list]" = []
        for r in rr.get("rows") or []:
            lo = _finite(r[1]) if isinstance(r, (list, tuple)) and len(r) == 3 else None
            hi = _finite(r[2]) if lo is not None else None
            if lo is None or hi is None or lo >= hi:
                r_rows = []
                break
            r_rows.append([str(r[0]), lo, hi])
        r_blk = _finite(rr.get("block_s"))
        if r_rows and len(r_rows) <= MAX_SEGMENTS and r_blk is not None and r_blk > 0:
            run = {"rows": r_rows, "block_s": r_blk}
    if not rows and run is None:
        return None
    if not rows:
        rows = [list(r) for r in run["rows"]]
    st_raw = raw.get("stim")
    if isinstance(st_raw, str):
        stim = st_raw.strip()
    else:
        stim = _fmt_s(_finite(st_raw)) if _finite(st_raw) is not None else ""
    return {"rows": rows, "block_s": block_s, "ref": ref, "view": view,
            "jitter": bool(raw.get("jitter", True)), "ran": run is not None, "run": run,
            "stim": stim}


def tachogram_groups(n: int, comparators: "Optional[list[Optional[int]]]" = None
                     ) -> "list[tuple[int, ...]]":
    """The segments each RR tachogram panel shows: in Pairs mode
    (comparators_reference == PAIRS) one panel per pair, (comparator,
    segment), and an odd segment left alone in its own; otherwise one panel
    with every segment superimposed."""
    if n <= 0:
        return []
    if comparators is not None and len(comparators) == n \
            and comparators_reference(comparators) == PAIRS:
        groups: "list[tuple[int, ...]]" = [(j, k) for j, k in comparison_pairs(comparators)]
        paired = {i for g in groups for i in g}
        groups += [(i,) for i in range(n) if i not in paired]
        return sorted(groups, key=lambda g: g[0])
    return [tuple(range(n))]


def draw_rr_tachograms(fig, segs: "list[dict]", colors: "list[str]",
                       comparators: "Optional[list[Optional[int]]]" = None,
                       names: "Optional[list[str]]" = None) -> list:
    """The RR tachograms of *segs* on *fig*: all superimposed on one axes,
    or, in Pairs mode, one axes per pair (the two superimposed, titled
    "B vs A") stacked vertically, sharing the RR scale so the pairs compare
    at a glance (the Y-scale bar and the wheel zoom then act on every
    panel). Time is within each segment. Returns the axes."""
    names = list(names) if names is not None else [str(s.get("label", "")) for s in segs]
    groups = tachogram_groups(len(segs), comparators)
    if not groups:
        return []
    axes = list(np.atleast_1d(fig.subplots(len(groups), 1, sharey=True, squeeze=False)).ravel())
    for ax, grp in zip(axes, groups):
        style_axes(ax)
        drawn = False
        for i in grp:
            s, c = segs[i], colors[i % len(colors)]
            t, r = s["t_rr"], s["rr_ms"]
            if len(t) and len(r):
                ax.plot(t - t[0], r, color=c, lw=1.0, alpha=0.85, label=names[i])
                ax.axhline(float(np.mean(r)), color=c, lw=1.0, ls="--", alpha=0.5)
                drawn = True
        ax.set_ylabel("RR (ms)", fontsize=9)
        if len(groups) == 1:
            ax.set_title("RR intervals — segments superimposed", loc="left", fontsize=9)
        elif len(grp) == 2:
            ax.set_title(f"{names[grp[1]]} vs {names[grp[0]]}", loc="left", fontsize=9)
        else:
            ax.set_title(names[grp[0]], loc="left", fontsize=9)
        if drawn:
            ax.legend(framealpha=0, fontsize=8, labelcolor=PLOT.get("text", "#222"))
    for ax in axes:
        ax.set_xlabel("Time within segment (s)", fontsize=9)
    return axes


# The two sheets the group Poincaré needs: the TimeCourse sheet of a big
# table took most of the 1.5 s of reading it whole.
_GROUP_SHEETS = ("Animals", "Group")


def segment_choices(seg: "pd.DataFrame") -> "list[tuple[str, str]]":
    """(tag, shown) of each segment of a group table's Group sheet: 'Pre ·
    Baseline' when the group's label is not the tag."""
    from ecg.io.group_stats import segment_labels, segment_tags
    out = []
    for tag in segment_tags(seg):
        label = segment_labels(seg, tag)[0]
        out.append((tag, f"{tag} · {label}" if label and label != tag else tag))
    return out


def segment_color(i: int) -> str:
    """The colour of the *i*-th segment of a comparison (the first, the
    reference, blue): the tachograms, the table and the group Poincaré. Read
    at call time, so a theme change re-synced into this module applies."""
    seq = (BLUE, ORANGE_DARK, GREEN, PURPLE, TEAL, RED)
    return seq[i % len(seq)]


def group_poincare_data(sheets: "dict", a: str, b: str, leave_out: bool = True,
                        by: "Optional[str]" = None) -> "tuple[pd.DataFrame, dict]":
    """The rows of the Group sheet the group Poincaré of A → B draws, with
    the Group tab's rules: unticked animals out; with *by* (the Group tab's
    By: group or sex; None, no groups) the animals without a group or sex
    out, as there; with *leave_out* the rows to check (quality alert,
    subject on several rows, another template or duration:
    ecg.io.group_stats.check_rows) out too; then the animals with a mean RR,
    SD1 and SD2 for both segments. And the counts left out ({"unticked": n,
    "ungrouped": n, "to_check": n, "pooled": bool}). When no ticked animal
    has a group (or a sex) at all, every ticked animal is drawn ("pooled"
    True): the Group tab then asks for groups or No groups, but the Poincaré
    of all the animals is still worth seeing here. ValueError when no animal
    is ticked."""
    from ecg.io import group_compare as gc
    from ecg.io import group_data as gd
    from ecg.io.group_stats import check_rows
    from ecg.app.group_figure import poincare_rows
    df = sheets[gd.SHEET_SEGMENTS].reset_index(drop=True)
    animals = gd.animals_frame(sheets)
    by = by or gc.BY_NONE
    gmap, _order, _notes = gc.animal_groups(animals, by)
    pooled = False
    if not gmap and by != gc.BY_NONE:
        gmap, _order, _notes = gc.animal_groups(animals, gc.BY_NONE)
        pooled = bool(gmap)
    gc._need_animals(gmap, gc.BY_NONE, animals)
    keys = list(zip(df["subject"].astype(str), df["file"].astype(str)))
    in_table = set(keys)
    ticked = {(str(r["subject"]), str(r["file"])) for _, r in animals.iterrows()
              if bool(r["use"])}
    unticked = sum(1 for _, r in animals.iterrows() if not bool(r["use"])
                   and (str(r["subject"]), str(r["file"])) in in_table)
    ungrouped = sum(1 for k in ticked & in_table if k not in gmap)
    # A row merely without one of the two segments is not "to check": it
    # simply has nothing to draw.
    doubtful = {(c.subject, c.file) for c in check_rows(df, a, b)
                if not c.ok and not all(p.startswith("no segment") for p in c.problems)}
    used = [k in gmap for k in keys]
    keep = [u and not (leave_out and k in doubtful) for u, k in zip(used, keys)]
    # Counted only when it had something to draw: a row with a quality alert
    # on A and no B is not drawn either way.
    to_check = 0
    if leave_out:
        out_rows = df.loc[[u and k in doubtful for u, k in zip(used, keys)]]
        to_check = len(poincare_rows(out_rows, a, b))
    return poincare_rows(df.loc[keep], a, b), {"unticked": unticked, "ungrouped": ungrouped,
                                               "to_check": to_check, "pooled": pooled}

