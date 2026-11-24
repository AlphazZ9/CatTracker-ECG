# -*- coding: utf-8 -*-
"""
ecg.core.peak_editing
---------------------
The accepted R-peaks after a threshold change or a manual edit, and the
signal-quality score shown beside them.

Every click in edit mode, threshold nudge, undo / redo and Detect Peaks
rebuilds the peaks from the full candidate set, in this order:

1. candidates kept by the prominence threshold (``apply_threshold``);
2. peaks excluded by hand are taken out;
3. peaks added by hand are put in (they bypass every threshold);
4. beats removed by Artifact Review or Analyze's auto-correction, and still
   absent from the previous peaks, are dropped again -- a hand-added peak
   never is.

Moved unchanged from ecg.ui.detection_controller (DetectionController.
run_detection and update_signal_quality) so it runs without the UI. The
reference outputs of tests/regression check it is the same computation.

Peaks are sample indices. Nothing here may import from ecg.ui.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ecg.core.detection import active_removed_samples, apply_threshold, drop_removed_peaks
from ecg.core.models import MouseECG


@dataclass
class PeakEdits:
    """The peaks rebuilt by ``apply_peak_edits`` (sample indices)."""
    accepted: np.ndarray          # the accepted peaks (rpeaks_ok)
    rejected: np.ndarray          # candidates under the threshold (rpeaks_rej)
    thresh_amp: float             # the threshold in normalised amplitude
    manual_excl: np.ndarray       # accepted by the threshold but excluded by hand
    manual_added: np.ndarray      # added by hand, sorted


def apply_peak_edits(
    signal: np.ndarray,
    candidates: np.ndarray,
    prominences: np.ndarray,
    thresh: float,
    fs: float,
    manual_excluded: "set[int]",
    manual_added: "set[int]",
    removed_samples: "Optional[list]" = None,
    previous_peaks: "Optional[np.ndarray]" = None,
    tol: int = 1,
) -> PeakEdits:
    """Rebuild the accepted peaks from the candidates (see the module text).

    Parameters
    ----------
    signal          : prepared (filtered, normalised) signal the candidates are on.
    candidates, prominences : every R-peak candidate and its prominence.
    thresh          : threshold, a fraction of the median R-peak prominence.
    fs              : sampling rate, Hz.
    manual_excluded : samples excluded by hand (not modified here).
    manual_added    : samples added by hand (never dropped).
    removed_samples : the artifact report's ``removed_samples``, or None.
    previous_peaks  : the accepted peaks before this rebuild: a removed beat
                      that was already back in them stays back.
    tol             : matching tolerance between a removal and a peak, samples.

    The caller removes *manual_added* from *manual_excluded* afterwards when
    peaks were added (a peak cannot be both): the exclusion above is applied
    with the set as given, as the app always did.
    """
    accepted, rejected, thresh_amp = apply_threshold(
        signal, candidates, prominences, thresh, fs=fs)

    # ── Manual exclusions ────────────────────────────────────
    if manual_excluded:
        manual_excl_mask = np.array([p in manual_excluded for p in accepted], dtype=bool)
        excl = accepted[manual_excl_mask] if manual_excl_mask.any() else np.array([], int)
        accepted = accepted[~manual_excl_mask]
    else:
        excl = np.array([], dtype=int)

    # ── Manually-added peaks ─────────────────────────────────
    # Never in the candidate set; they bypass all thresholds.
    if manual_added:
        added_arr = np.array(sorted(manual_added), dtype=int)
        accepted = np.unique(np.concatenate([accepted, added_arr]))
    else:
        added_arr = np.array([], dtype=int)

    # ── Artifact removals still in effect ────────────────────
    # Artifact Review and auto-correction write the corrected peaks into
    # rpeaks_ok, not into the manual sets, so the rebuild above (full
    # candidate set) brings every removed beat back. Drop again the removals
    # whose beat was still absent from the peaks before this rebuild: a beat
    # that is already back (a fresh Detect Peaks, or re-added by hand) stays
    # back, and hand-added peaks are never dropped.
    if removed_samples and previous_peaks is not None and len(previous_peaks) and len(accepted):
        still = active_removed_samples(removed_samples, previous_peaks, tol)
        if len(still):
            kept = drop_removed_peaks(accepted, still, tol, protected=manual_added)
            if len(kept) != len(accepted):
                accepted = kept.astype(np.asarray(accepted).dtype, copy=False)

    return PeakEdits(accepted=accepted, rejected=rejected, thresh_amp=thresh_amp,
                     manual_excl=excl, manual_added=added_arr)


def signal_quality_score(
    accepted: np.ndarray,
    fs: float,
    duration_s: float,
    beat_corr: "Optional[np.ndarray]" = None,
) -> "Optional[int]":
    """0-100 signal-quality score of the accepted peaks (the quality gauge).

    1. Beat morphology (primary): mean beat-to-template correlation of the
       last analysis (*beat_corr*). High correlation = clean, consistent QRS.
       Without it, RR regularity is used instead (1 - CV of the RR).
    2. Detection completeness (secondary): ratio of detected to expected
       beats at the resting HR (MouseECG.HR_REST_BPM) over *duration_s*
       (the time of the last sample), clipped to [0.5, 1.5] so it modulates
       but never dominates.

    The earlier formula (1 - rr_cv) alone was unreliable: a healthy mouse at
    high HR during stress has a low rr_cv not from noise but from genuine
    sympathetic activation. None with 5 peaks or fewer (no score).
    """
    n = len(accepted)
    if n <= 5:
        return None
    expected_n = duration_s / 60 * MouseECG.HR_REST_BPM
    ratio = float(np.clip(n / max(expected_n, 1), 0.5, 1.5))

    if beat_corr is not None and len(beat_corr) > 0:
        morpho = float(np.nanmean(beat_corr))            # 0–1 (Pearson r)
        morpho = float(np.clip(morpho, 0.0, 1.0))
        return int(np.clip(100 * morpho * ratio, 0, 100))
    # Fallback before full analysis: RR regularity heuristic
    rr_tmp = np.diff(accepted) / fs * 1000
    rr_cv = rr_tmp.std() / (rr_tmp.mean() + 1e-6)
    return int(np.clip(100 * (1 - rr_cv) * ratio, 0, 100))
