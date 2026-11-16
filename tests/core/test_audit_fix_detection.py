"""
Regression tests for audit 2026-11-13 bugs #3, #4 and #5 (ecg.core.detection).

#3  resolve_r_vs_j_peaks doublet pass compared a peak with one already
    discarded in the same pass and dropped the real beat after it.
#4  _composite_qrs_score rewarded WIDER peaks (+1.5 * z_width), the opposite
    of its documented rationale (a QRS is narrower than a T/J wave or EMG).
#5  detect_rr_artifacts doublet pre-pass skipped the pair after a flagged
    peak, so a triplet left a sub-Min-RR interval and a real beat was
    flagged ectopic instead.
"""
from __future__ import annotations

import numpy as np

from ecg.core import detection as det
from ecg.core.detection import (
    apply_artifact_decisions,
    detect_rr_artifacts,
    resolve_r_vs_j_peaks,
)


# ── #3 ─────────────────────────────────────────────────────────────────────

def _put_peak(sig: np.ndarray, p: int, rise: int, amp: float = 1.0) -> None:
    """Linear upstroke over *rise* samples then a 5-sample downstroke."""
    for k in range(rise + 1):
        sig[p - rise + k] = max(sig[p - rise + k], amp * k / rise)
    for k in range(1, 6):
        sig[p + k] = max(sig[p + k], amp * (1 - k / 5))


def test_resolve_r_vs_j_does_not_drop_beat_after_discarded_spike():
    fs = 1000.0
    sig = np.zeros(2000)
    beats = list(range(100, 1900, 120))
    # Steepest upstroke at 1060, a spike at 1120 (60 ms later, < Min RR)
    # with an intermediate slope, the next real beat at 1180 slightly slower.
    for b in beats:
        _put_peak(sig, b, 1 if b == 1060 else 3)
    _put_peak(sig, 1120, 2)
    peaks = np.array(sorted(beats + [1120]))

    out = resolve_r_vs_j_peaks(peaks, sig, fs)

    assert 1120 not in out.tolist()
    # 1060 -> 1180 is 120 ms: 1180 must never be compared with the
    # already-discarded 1120.
    assert sorted(out.tolist()) == beats


def test_resolve_r_vs_j_chain_of_close_spikes_keeps_one():
    # Three spikes under Min RR of each other: only the steepest survives,
    # and every real beat around them is kept.
    fs = 1000.0
    sig = np.zeros(2000)
    beats = list(range(100, 1900, 120))
    for b in beats:
        _put_peak(sig, b, 1)
    _put_peak(sig, 1080, 3, 0.6)
    _put_peak(sig, 1100, 3, 0.6)
    peaks = np.array(sorted(beats + [1080, 1100]))

    out = resolve_r_vs_j_peaks(peaks, sig, fs)

    assert sorted(out.tolist()) == beats


# ── #4 ─────────────────────────────────────────────────────────────────────

def test_composite_score_rewards_narrow_peaks(monkeypatch):
    """With every other descriptor identical, the narrower candidates must
    score higher than the wider ones (QRS narrower than T/J/EMG)."""
    n = 40
    peaks = np.arange(n) * 100 + 50
    width50 = np.where(np.arange(n) % 2 == 0, 7.0, 38.0)  # even = narrow R

    monkeypatch.setattr(det, "_topographic_prominences",
                        lambda signal, pk, fs: np.ones(len(pk)))
    monkeypatch.setattr(det, "_extended_morphology_descriptors",
                        lambda pk, signal, fs, prom: dict(
                            width50=width50.copy(),
                            # Same value for all: isolates the width term.
                            width_height_ratio=np.ones(len(pk)),
                            curvature=-np.ones(len(pk)),
                            slope_up=np.ones(len(pk)),
                            slope_down=-np.ones(len(pk)),
                        ))
    monkeypatch.setattr(det, "_modulus_maxima_alignment",
                        lambda coeffs, scales, pk, s: np.zeros(len(pk)))
    monkeypatch.setattr(det, "_build_qrs_template", lambda *a, **k: None)

    signal = np.zeros(n * 100 + 100)
    energy = np.ones_like(signal)
    persistence = np.ones_like(signal)
    score, _ = det._composite_qrs_score(
        signal, peaks, np.zeros((1, len(signal))), np.array([1.0]),
        energy, persistence, 1, 1000.0, 1.0, 3, 10,
    )

    narrow, wide = score[0::2], score[1::2]
    assert narrow.min() > wide.max()


def test_composite_score_ranks_r_above_t_on_synthetic_ecg():
    """End-to-end on the morphology part: narrow R peaks get a higher
    composite score than the wide T waves that follow them."""
    fs = 2000.0
    n = int(fs * 5)
    t = np.arange(n)
    sig = np.zeros(n)
    R, T = [], []
    for k in range(40):
        r = 200 + k * 240
        R.append(r)
        T.append(r + 90)
        sig += np.exp(-0.5 * ((t - r) / 3.0) ** 2)
        sig += 0.5 * np.exp(-0.5 * ((t - (r + 90)) / 16.0) ** 2)
    sig += np.random.default_rng(0).normal(0, 0.01, n)
    peaks = np.array(sorted(R + T))
    is_r = np.isin(peaks, R)

    prom = det._topographic_prominences(sig, peaks, fs)
    m = det._extended_morphology_descriptors(peaks, sig, fs, prom)
    assert m["width50"][is_r].mean() < m["width50"][~is_r].mean()

    energy = np.ones(n)
    persistence = np.ones(n)
    score, _ = det._composite_qrs_score(
        sig, peaks, np.zeros((1, n)), np.array([1.0]),
        energy, persistence, 1, fs, 1.0, 3, 20,
    )
    assert score[is_r].mean() > score[~is_r].mean()


# ── #5 ─────────────────────────────────────────────────────────────────────

def _triplet_peaks():
    beats = list(range(100, 3000, 120))
    return beats, np.array(sorted(beats + [1085, 1105]))


def test_detect_rr_artifacts_triplet_flags_both_extras_not_real_beat():
    fs = 1000.0
    beats, peaks = _triplet_peaks()

    cands = detect_rr_artifacts(peaks, fs)
    flagged = {c["sample"] for c in cands}

    assert {1085, 1105} <= flagged
    assert 1180 not in flagged
    corrected, _ = apply_artifact_decisions(peaks, cands, fs)
    assert sorted(np.asarray(corrected).tolist()) == beats
    assert np.diff(corrected).min() / fs * 1000 >= det.MouseECG.RR_MIN_MS


def test_detect_rr_artifacts_triplet_with_signal_keeps_tallest():
    fs = 1000.0
    beats, peaks = _triplet_peaks()
    sig = np.zeros(3200)
    sig[beats] = 1.0
    sig[[1085, 1105]] = 0.5

    cands = detect_rr_artifacts(peaks, fs, signal=sig)
    corrected, _ = apply_artifact_decisions(peaks, cands, fs)

    assert sorted(np.asarray(corrected).tolist()) == beats


def test_detect_rr_artifacts_extra_taller_than_beat_replaces_it():
    # When the extra detection is taller, it is kept and the pointer moves
    # to it: the next comparison is against the survivor.
    fs = 1000.0
    beats, peaks = _triplet_peaks()
    sig = np.zeros(3200)
    sig[beats] = 1.0
    sig[1085] = 2.0
    sig[1105] = 0.5

    cands = detect_rr_artifacts(peaks, fs, signal=sig)
    dup = {c["sample"] for c in cands if c["type"] == "duplicate"}

    assert dup == {1060, 1105}
