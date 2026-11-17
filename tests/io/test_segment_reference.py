"""Compare segments with a chosen reference: every segment against segment k,
or each against the one before it (pairwise). The comparators drive the Δ %
columns, the block-mean summary, the quality check of durations and every
export; without them (None) everything stays against the first segment."""
from __future__ import annotations

import pandas as pd
import pytest

from ecg.core import segment_plan as sp
from ecg.core.analysis import compare_segments
from ecg.io import segment_export as se
from tests.io.test_segment_export import _result


def _with(res: se.ComparisonResult, reference) -> se.ComparisonResult:
    """*res* re-pointed at *reference*, as the tab does when it changes."""
    comps = sp.reference_comparators(len(res.segs), reference)
    res.comparators = comps
    res.cmps = [compare_segments(res.segs[j], res.segs[k]) for j, k in sp.comparison_pairs(comps)]
    res.flags = sp.quality_flags(res.segs, res.event, res.artefact_margin_s or 0.0, comps)
    return res


# ── The comparator logic ─────────────────────────────────────────────────

def test_reference_comparators():
    assert sp.reference_comparators(3) == [None, 0, 0]
    assert sp.reference_comparators(3, 2) == [2, 2, None]
    assert sp.reference_comparators(4, sp.PAIRWISE) == [None, 0, 1, 2]
    assert sp.reference_comparators(2, 5) == [None, 0]          # out of range: the first
    assert sp.reference_comparators(1, sp.PAIRWISE) == [None]
    assert sp.reference_comparators(0) == []


def test_pairs_and_what_they_are_against():
    assert sp.comparison_pairs([2, 2, None]) == [(2, 0), (2, 1)]
    assert sp.comparison_pairs([None, 0, 1]) == [(0, 1), (1, 2)]
    assert sp.comparators_reference([1, None, 1]) == 1
    assert sp.comparators_reference([None, 0, 1]) == sp.PAIRWISE
    assert sp.comparators_reference([None, 0]) == 0       # two segments: both readings agree
    labels = ["Base", "Post Stim 1min", "Late"]
    assert sp.comparators_caption([1, None, 1], labels) == "Δ % against Post Stim 1min"
    assert "previous segment" in sp.comparators_caption([None, 0, 1], labels)


def test_stale_comparators_fall_back_to_the_first():
    assert sp.resolve_comparators(None, 3) == [None, 0, 0]
    assert sp.resolve_comparators([None, 0], 3) == [None, 0, 0]      # wrong length
    assert sp.resolve_comparators([None, 1, 0], 3) == [None, 0, 0]   # compared with itself
    assert sp.resolve_comparators([0, 0, 0], 3) == [None, 0, 0]      # no reference
    assert sp.resolve_comparators([None, 0, 7], 3) == [None, 0, 0]   # out of range
    assert sp.resolve_comparators([1, None, 1], 3) == [1, None, 1]


def test_duration_check_follows_the_comparator():
    from tests.core.test_segment_plan import _seg
    segs = [_seg(0, 300, label="A"), _seg(300, 600, label="B"), _seg(600, 800, label="C")]
    # Default: against the first -- C (200 s) is flagged, and says against whom.
    flags = sp.quality_flags(segs)
    assert flags[0] == [] and flags[1] == []
    assert any("lasts 200 s, not 300 s like A" in f for f in flags[2])
    # C as the reference: A and B are flagged against C, C itself is not.
    flags = sp.quality_flags(segs, comparators=sp.reference_comparators(3, 2))
    assert flags[2] == []
    assert any("not 200 s like C" in f for f in flags[0])
    assert any("not 200 s like C" in f for f in flags[1])
    # Pairwise: C against B.
    flags = sp.quality_flags(segs, comparators=sp.reference_comparators(3, sp.PAIRWISE))
    assert flags[1] == [] and any("like B" in f for f in flags[2])


def test_duration_check_names_a_shared_label_by_position():
    from tests.core.test_segment_plan import _seg
    segs = [_seg(0, 300, label="A"), _seg(300, 600, label="A"), _seg(600, 800, label="B")]
    flags = sp.quality_flags(segs, comparators=sp.reference_comparators(3, 1))
    assert any("like A (segment 2) (its comparison)" in f for f in flags[2])


# ── Exports follow the comparators ───────────────────────────────────────

def test_default_is_unchanged():
    res = _result(3)
    assert res.comparators is None and res.comps == [None, 0, 0]
    assert list(se.comparison_frame(res).columns) == [
        "Metric", "Before", "After", "Late", "Δ% Before→After", "Δ% Before→Late"]
    rows = dict(se.summary_rows(res))
    assert rows["Reference"] == "Before" and "Before vs After" in rows
    assert rows["Δ %"] == "Δ % against Before"


def test_frame_against_a_chosen_reference():
    res = _with(_result(3), 1)
    df = se.comparison_frame(res)
    assert list(df.columns) == ["Metric", "Before", "After", "Late",
                                "Δ% After→Before", "Δ% After→Late"]
    hr = df.set_index("Metric").loc["Mean HR (bpm)"]
    assert hr["Δ% After→Late"] == pytest.approx(100 * (hr["Late"] - hr["After"]) / hr["After"])
    rows = dict(se.summary_rows(res))
    assert rows["Reference"] == "After"
    assert "After vs Before" in rows and "After vs Late" in rows and "Before vs After" not in rows


def test_frame_pairwise():
    res = _with(_result(3), sp.PAIRWISE)
    df = se.comparison_frame(res)
    assert list(df.columns)[-2:] == ["Δ% Before→After", "Δ% After→Late"]
    hr = df.set_index("Metric").loc["Mean HR (bpm)"]
    assert hr["Δ% After→Late"] == pytest.approx(100 * (hr["Late"] - hr["After"]) / hr["After"])
    rows = dict(se.summary_rows(res))
    assert rows["Reference"] == "previous segment (pairwise)"
    assert "Before vs After" in rows and "After vs Late" in rows
    # The TSV and CSV are the same table.
    assert se.comparison_tsv(res).split("\n")[0].split("\t") == list(df.columns)


def test_xlsx_and_prism_follow_the_reference(tmp_path):
    res = _with(_result(3), 2)
    path = tmp_path / "cmp.xlsx"
    se.write_xlsx(res, str(path))
    book = pd.read_excel(path, sheet_name=None)
    assert list(book["Comparison"].columns)[-2:] == ["Δ% Late→Before", "Δ% Late→After"]
    summary = dict(zip(book["Summary"]["Item"], book["Summary"]["Value"]))
    assert summary["Reference"] == "Late" and "Late vs Before" in summary
    # Prism: the values of each segment, no Δ (unchanged by the reference).
    name, table, _x, _r = se.prism_tables(res)[0]
    assert list(table.columns) == ["Before", "After", "Late"]


def test_main_workbook_sheets_follow_the_reference():
    import openpyxl
    res = _with(_result(3), sp.PAIRWISE)
    wb = openpyxl.Workbook()
    se.add_sheets(wb, res)
    head = [c.value for c in wb["Seg Comparison"][1]]
    assert head[-2:] == ["Δ% Before→After", "Δ% After→Late"]



# ── Pairs: 2 vs 1, 4 vs 3, 6 vs 5 (Pre / Post protocols) ────────────────────

def test_pairs_comparators_and_caption():
    assert sp.reference_comparators(6, sp.PAIRS) == [None, 0, None, 2, None, 4]
    assert sp.reference_comparators(5, sp.PAIRS) == [None, 0, None, 2, None]   # last alone
    assert sp.comparators_reference([None, 0, None, 2]) == sp.PAIRS
    # Not mistaken for pairwise (which compares 3 with 2 as well), nor the reverse.
    assert sp.comparators_reference([None, 0, 1, 2]) == sp.PAIRWISE
    assert sp.comparators_reference([None, 0]) == 0      # two segments: the index wins
    assert sp.resolve_comparators([None, 0, None, 2], 4) == [None, 0, None, 2]
    labels = ["Pre 1", "Post 1", "Pre 5", "Post 5"]
    cap = sp.comparators_caption([None, 0, None, 2], labels)
    assert cap == "Δ % within pairs: Post 1 vs Pre 1  ·  Post 5 vs Pre 5"


def test_frame_and_summary_within_pairs():
    base = _result(3)
    pre_post = [dict(base.segs[0], label="Pre 1"), dict(base.segs[1], label="Post 1"),
                dict(base.segs[2], label="Pre 5"), dict(base.segs[1], label="Post 5")]
    res = se.ComparisonResult(
        segs=pre_post, metrics=base.metrics, cmps=[], flags=[[]] * 4, template="",
        subject="M1", filepath="", event=None, artefact_margin_s=None, block_s=10.0,
        lf_band=base.lf_band, hf_band=base.hf_band)
    res = _with(res, sp.PAIRS)
    df = se.comparison_frame(res)
    assert list(df.columns)[-2:] == ["Δ% Pre 1→Post 1", "Δ% Pre 5→Post 5"]
    hr = df.set_index("Metric").loc["Mean HR (bpm)"]
    assert hr["Δ% Pre 5→Post 5"] == pytest.approx(
        100 * (hr["Post 5"] - hr["Pre 5"]) / hr["Pre 5"])
    rows = dict(se.summary_rows(res))
    assert rows["Reference"].startswith("within pairs")
    assert "Pre 1 vs Post 1" in rows and "Pre 5 vs Post 5" in rows
    assert "Post 1 vs Pre 5" not in rows and "Pre 1 vs Pre 5" not in rows
