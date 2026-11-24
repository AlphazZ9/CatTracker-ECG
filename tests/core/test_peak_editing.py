"""The rules of ecg.core.peak_editing, called directly (no UI): how a
threshold change or a manual edit rebuilds the accepted peaks, and the
signal-quality score. The app-level behaviour is held by tests/regression
and tests/ui/test_audit_fix_artifact_review.py."""
import numpy as np
import pytest

from ecg.core import peak_editing
from ecg.core.peak_editing import apply_peak_edits, signal_quality_score

FS = 2000.0
CANDS = np.arange(1, 21) * 240          # 20 candidates, 120 ms apart


@pytest.fixture(autouse=True)
def _threshold_keeps_every_candidate(monkeypatch):
    monkeypatch.setattr(peak_editing, "apply_threshold",
                        lambda sig, c, p, t, fs=0.0: (np.asarray(c).copy(), np.array([], int), 0.5))


def _edits(**kw):
    args = dict(manual_excluded=set(), manual_added=set())
    args.update(kw)
    return apply_peak_edits(np.zeros(6000), CANDS, np.ones(len(CANDS)), 0.5, FS,
                            args.pop("manual_excluded"), args.pop("manual_added"), **args)


def test_no_edit_keeps_the_threshold_result():
    e = _edits()
    assert np.array_equal(e.accepted, CANDS)
    assert len(e.manual_excl) == 0 and len(e.manual_added) == 0 and e.thresh_amp == 0.5


def test_excluded_peaks_are_taken_out_and_reported():
    excl = {int(CANDS[3]), int(CANDS[7]), 99}           # 99 is not a candidate
    e = _edits(manual_excluded=excl)
    assert np.array_equal(e.manual_excl, CANDS[[3, 7]])
    assert not np.isin(CANDS[[3, 7]], e.accepted).any() and len(e.accepted) == 18


def test_added_peaks_bypass_the_threshold_sorted_and_unique():
    e = _edits(manual_added={int(CANDS[2]) + 120, int(CANDS[2])})
    assert np.array_equal(e.manual_added, np.sort([CANDS[2], CANDS[2] + 120]))
    assert int(CANDS[2]) + 120 in e.accepted and len(e.accepted) == 21
    assert np.all(np.diff(e.accepted) > 0)


def test_inputs_are_not_modified():
    excl, added = {int(CANDS[1])}, {int(CANDS[1])}
    _edits(manual_excluded=excl, manual_added=added)
    assert excl == {int(CANDS[1])} and added == {int(CANDS[1])}


def test_removed_beat_still_absent_is_dropped_again_but_not_a_hand_added_one():
    removed = [int(CANDS[5]), int(CANDS[9])]
    previous = CANDS[~np.isin(CANDS, removed)]           # both absent before the rebuild
    e = _edits(removed_samples=removed, previous_peaks=previous, tol=2)
    assert not np.isin(removed, e.accepted).any()
    kept = _edits(removed_samples=removed, previous_peaks=previous, tol=2,
                  manual_added={removed[0]})
    assert removed[0] in kept.accepted and removed[1] not in kept.accepted


def test_removed_beat_already_back_stays_back():
    removed = [int(CANDS[5])]
    e = _edits(removed_samples=removed, previous_peaks=CANDS, tol=2)
    assert removed[0] in e.accepted


def test_quality_uses_beat_correlation_and_completeness():
    peaks = np.arange(500) * 240     # 500 bpm for 60 s
    assert signal_quality_score(peaks, FS, 60.0, np.full(10, 0.9)) == 90
    # Half the expected beats: completeness 0.5 halves the score.
    assert signal_quality_score(peaks[:250], FS, 60.0, np.full(10, 0.9)) == 45


def test_quality_falls_back_to_rr_regularity_and_needs_six_peaks():
    regular = np.arange(500) * 240
    assert signal_quality_score(regular, FS, 60.0) == 100
    assert signal_quality_score(regular[:5], FS, 60.0) is None
