# -*- coding: utf-8 -*-
"""
ecg.services.analysis_service
-----------------------------
The core analysis of Analyze, without the UI: optional automatic artifact
correction, the cumulative artifact report, the hand-marked noisy regions,
the analysis window, then ecg.core.analysis.analyse_core.

Moved unchanged from ecg.ui.analysis_controller (the body of
AnalysisController.analysis_worker). The reference outputs of
tests/regression check it is the same computation.

Order that matters (kept as it was): the analysis window is applied AFTER
the artifact correction, and the full corrected peak set is returned beside
the windowed one, so changing the window always starts again from the whole
detection.

Peaks are sample indices; times in seconds; RR in ms.
Nothing here may import from ecg.ui.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from ecg.core.analysis import analyse_core, noise_breaks
from ecg.core.detection import correct_rr_artifacts, merge_artifact_reports
from ecg.core.models import MouseECG

log = logging.getLogger("ecg")


def run_core_analysis(
    rpeaks: np.ndarray,
    signal: "Optional[np.ndarray]",
    fs: float,
    params: dict,
    prev_report: "dict | None" = None,
    progress_cb: "Optional[Callable[[int, str], None]]" = None,
) -> dict:
    """Analyze's core computation on the accepted peaks *rpeaks*.

    Parameters
    ----------
    rpeaks      : accepted R-peaks (sample indices); not modified.
    signal      : prepared signal (filtered, normalised) the peaks are on.
    fs          : sampling rate, Hz.
    params      : the Analyze snapshot -- ``artifact_correction``,
                  ``min_rr_ms``, ``hrv_rr_bounds`` (ms), ``break_tol_samples``,
                  ``noise_regions``, ``analysis_t_start`` / ``analysis_t_end``
                  (s, 0 = no bound), ``auto_epochs``.
    prev_report : the artifact report in effect before this run (an earlier
                  auto-correction or a reviewed one). Its removals whose peak
                  is still absent stay in the report -- counts and HRV breaks
                  alike (merge_artifact_reports): the peaks given here are
                  already corrected, so a fresh correction pass finds nothing
                  to remove and would otherwise report "no artifacts" while
                  the breaks persist.
    progress_cb : (percent, message), or None.

    Returns {"results", "rpeaks_ok" (the full corrected set), "rpeaks_analysed"
    (its windowed subset), "artifact_report", "auto_epochs",
    "analysis_window"}. Raises ValueError for a window with fewer than 5
    peaks, RuntimeError without a signal.
    """
    rp = np.asarray(rpeaks).copy()

    def _prog(pct: int, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(pct, msg)

    rr_lo, rr_hi = params.get("hrv_rr_bounds",
                              (MouseECG.RR_MIN_MS, MouseECG.RR_MAX_MS))
    pass_report = None
    if params["artifact_correction"]:
        _prog(2, "Artifact correction (auto)…")
        try:
            rp_corrected, pass_report = correct_rr_artifacts(
                rp, fs,
                rr_min_ms=params.get("min_rr_ms", MouseECG.RR_MIN_MS),
                rr_max_ms=rr_hi,
                window_beats=11, dev_threshold=0.20,
                signal=signal,
            )
            rp = rp_corrected
            removed = pass_report["n_in"] - pass_report["n_out"]
            log.info("Artifact correction: −%d peaks (non-physio=%d ectopic=%d dup=%d, "
                     "%d HRV breaks)",
                     removed, pass_report["n_nonphysio"], pass_report["n_ectopic"],
                     pass_report["n_duplicate"], len(pass_report["break_samples"]))
        except Exception as exc:
            log.warning("Artifact correction failed: %s", exc)
            pass_report = None
    # Cumulative report: this pass's removals plus earlier ones whose peak
    # is still absent, counts recomputed from them. Without a new pass
    # (correction off or failed) the earlier report is kept while any of
    # its removals is still in effect, and dropped once all are back.
    artifact_report = merge_artifact_reports(
        prev_report, pass_report, rp, params.get("break_tol_samples", 1))
    breaks = (np.asarray(artifact_report["break_samples"], dtype=np.int64)
              if artifact_report and artifact_report["break_samples"] else None)
    # Noisy regions marked by hand: their intervals are left out of HR /
    # HRV (on the corrected peaks; the windowing below keeps them).
    noisy = noise_breaks(rp, fs or MouseECG.FS_DEFAULT,
                         params.get("noise_regions"))
    if len(noisy):
        breaks = noisy if breaks is None else np.union1d(breaks, noisy)

    # ── Save full (artifact-corrected) peak set BEFORE windowing ─────────
    # This is what _on_analysis_done will write back to _rpeaks_ok so that
    # changing the analysis window on the next run always starts from the
    # complete detection result — not a previously-windowed subset.
    rp_full = rp.copy()

    # ── Apply analysis window AFTER artifact correction ───────────────────
    # Window is applied to the analysis only — rp_full is always preserved.
    ana_t0 = params.get("analysis_t_start", 0.0)
    ana_t1 = params.get("analysis_t_end",   0.0)
    if (ana_t0 > 0 or ana_t1 > 0) and fs is not None:
        fs_snap = fs
        mask = rp / fs_snap >= ana_t0
        if ana_t1 > 0:
            mask &= rp / fs_snap <= ana_t1
        rp_windowed = rp[mask]
        if len(rp_windowed) < 5:
            raise ValueError(
                f"Analysis window too short: only {len(rp_windowed)} peaks "
                f"between {ana_t0:.1f} s and "
                f"{'the end' if ana_t1 == 0 else f'{ana_t1:.1f} s'}.\n"
                "Widen the window or click Full to reset it.")
        rp = rp_windowed
        log.info("Analysis window applied: %.1f s → %s  (%d / %d peaks)",
                 ana_t0, f"{ana_t1:.1f} s" if ana_t1 > 0 else "end",
                 len(rp), len(rp_full))

    _prog(10, "Core analysis (RR, HR, time-domain HRV, beat template)…")
    if signal is None:
        raise RuntimeError("Signal not loaded — click Detect Peaks first.")
    results = analyse_core(signal, rp, fs,
                           progress_cb=lambda p, m: _prog(10 + int(p * 0.9), m),
                           rr_min_ms=rr_lo, rr_max_ms=rr_hi, breaks=breaks)

    return {
        "results":         results,
        "rpeaks_ok":       rp_full,   # ← always full set: never overwrite with windowed
        "rpeaks_analysed": rp,         # ← windowed subset used for this analysis
        "artifact_report": artifact_report,
        "auto_epochs":     params.get("auto_epochs", False),
        "analysis_window": (float(ana_t0), float(ana_t1)),
    }
