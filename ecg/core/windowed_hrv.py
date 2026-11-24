# -*- coding: utf-8 -*-
"""
ecg.core.windowed_hrv
---------------------
HRV over successive time windows: Epochs (fixed windows, with overlap) and
Sliding (a window of win_s seconds every step_s seconds), the two modes of
the Time course tab.

Moved unchanged from ecg.ui.analysis_controller (the workers of
AnalysisController.compute_epochs and compute_rolling_hrv, and
_window_lf_hf) so they run without the UI. The reference outputs of
tests/regression check it is the same computation.

The two loops are deliberately NOT merged: Epochs takes SDNN / RMSSD from
hrv_time_domain (NeuroKit2, gap-aware), Sliding computes them directly, and
tests/ui/test_epoch_sliding_characterisation.py pins a case where their RMSSD
differs. Merging them would be a scientific change, to be decided and
validated on its own.

Times are in seconds, peaks are sample indices, RR in ms.
Nothing here may import from ecg.ui.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np

from ecg.core.analysis import (
    analyse_hrv_freq, compute_poincare_metrics, hrv_time_domain, mean_hr_bpm,
    poincare_sd1_sd2, rr_interval_series, successive_valid_pairs,
)
from ecg.core.models import MouseECG

log = logging.getLogger("ecg")

# Hard floors of a window: fewer peaks, or fewer valid RR intervals, and the
# window is skipped (counted in n_skipped) rather than given a row of NaN.
MIN_WINDOW_BEATS = 5
MIN_VALID_INTERVALS = 3
# Epochs columns computed only when ticked.
EPOCH_OPTIONAL = ("SD1", "SD2", "LF_nu", "HF_nu", "LF_HF")

ProgressCb = Optional[Callable[[int, str], None]]


def window_lf_hf(ep: np.ndarray, fs: float, rr_lo: float, rr_hi: float, breaks: Any,
                  bands: Any, t0: float) -> "tuple[float, float, float]":
    """(LF n.u. %, HF n.u. %, LF/HF) of one window -- the exact same
    analyse_hrv_freq() path as the Frequency panel (mouse-specific bands,
    Welch PSD, its own MIN_BEATS_SPECTRAL guard and gap-withholding rule).
    It returns an empty DataFrame -- NaN here -- rather than a number when a
    window is too short, or its excluded intervals too long, for a meaningful
    spectral estimate, instead of silently fabricating one."""
    try:
        fdf = analyse_hrv_freq(ep, fs, rr_min_ms=rr_lo, rr_max_ms=rr_hi,
                               breaks=breaks, bands=bands)
        if not fdf.empty:
            return (float(fdf["HRV_LFn"].values[0]) * 100.0,
                    float(fdf["HRV_HFn"].values[0]) * 100.0,
                    float(fdf["HRV_LFHF"].values[0]))
    except Exception as exc:
        log.debug("window_lf_hf: analyse_hrv_freq failed at t=%.1f: %s", t0, exc)
    return np.nan, np.nan, np.nan


def epoch_grid(rpeaks: np.ndarray, fs: float, epoch_s: float,
               overlap_s: float) -> "tuple[float, np.ndarray, float]":
    """(step s, start times s, span s) of the epochs over *rpeaks*: from the
    first to the last peak, one epoch of *epoch_s* every epoch_s - overlap_s
    seconds (at least 1 s), the last one ending at or before the last peak
    (to within half a step)."""
    t_peaks = rpeaks / fs
    dur = float(t_peaks[-1] - t_peaks[0])
    step = max(1.0, epoch_s - overlap_s)
    t_win_start = float(t_peaks[0])    # absolute start of the windowed range
    starts = np.arange(t_win_start, t_win_start + dur - epoch_s + step * 0.5, step)
    return step, starts, dur


def epoch_hrv(
    rp: np.ndarray,
    fs: float,
    starts: np.ndarray,
    epoch_s: float,
    rr_lo: float,
    rr_hi: float,
    breaks: Any,
    bands: Any,
    need_sd: bool,
    need_freq: bool,
    extra: "set[str]",
    optional: "tuple[str, ...]" = EPOCH_OPTIONAL,
    min_beats: int = MIN_WINDOW_BEATS,
    min_valid: int = MIN_VALID_INTERVALS,
    progress_cb: ProgressCb = None,
) -> dict:
    """HRV of each epoch [t0, t0 + epoch_s) for t0 in *starts* (s).

    Gap-aware like analyse_core: RR outside [rr_lo, rr_hi] ms or spanning a
    removed beat (*breaks*) are excluded, and successive differences never
    straddle them. SDNN / RMSSD come from hrv_time_domain; pNN6 from the
    successive valid pairs. The *optional* columns are kept only when in
    *extra* (SD1 / SD2 if *need_sd*, LF / HF n.u. and LF/HF if *need_freq*,
    with the frequency bands *bands*).

    Returns {"rows": [one dict per epoch kept], "n_skipped": epochs with
    fewer than *min_beats* peaks or *min_valid* valid intervals}.
    """
    rows = []
    n_skipped = 0
    n_ep = len(starts)
    for idx_ep, t0 in enumerate(starts):
        t1    = t0 + epoch_s
        ep_rp = rp[(rp / fs >= t0) & (rp / fs < t1)]
        if len(ep_rp) < min_beats:
            n_skipped += 1
            continue
        # Gap-aware, like analyse_core: out-of-range RR intervals and
        # intervals spanning a removed beat are excluded, and
        # successive differences never straddle them.
        ser = rr_interval_series(ep_rp, fs, rr_lo, rr_hi, breaks=breaks)
        rr  = ser["rri"]
        if len(rr) < min_valid:
            n_skipped += 1
            continue
        rr_a, rr_b = successive_valid_pairs(ser["rr_ms"], ser["valid"])
        sdiff = rr_b - rr_a
        try:
            # Same gap-aware time-domain helper as analyse_core --
            # nk.hrv_time alone can difference across an excluded
            # interval in short epochs (see hrv_time_domain).
            hrv_ep = hrv_time_domain(ser, fs)
            sdnn   = float(hrv_ep["HRV_SDNN"].values[0])
            rmssd  = float(hrv_ep["HRV_RMSSD"].values[0])
        except Exception as exc:
            log.warning("Epoch hrv_time failed, manual calc: %s", exc)
            # ddof=1 (sample std) to match nk.hrv_time's own convention
            # and compute_rolling_hrv's manual computation below --
            # this previously defaulted to ddof=0 (population std),
            # silently disagreeing with both whenever this fallback
            # actually triggered.
            sdnn  = float(rr.std(ddof=1)) if len(rr) > 1 else float("nan")
            rmssd = float(np.sqrt(np.mean(sdiff ** 2))) if len(sdiff) else float("nan")
        diffs = np.abs(sdiff)
        pnn6  = float(100.0 * np.sum(diffs > MouseECG.PNN_THRESHOLD) / len(diffs)) \
                if len(diffs) else float("nan")
        row = {
            "Epoch_start_s": round(t0, 1),
            "Epoch_end_s":   round(t1, 1),
            "N_beats":       len(ep_rp),
            "N_valid":       len(rr),
            "N_excluded":    ser["n_excluded"],
            "HR_mean":       round(mean_hr_bpm(rr), 1),   # 60000 / mean NN
            "MeanNN":        round(float(rr.mean()), 1),
            "SDNN":          round(sdnn, 2),
            "RMSSD":         round(rmssd, 2),
            "pNN6":          round(pnn6, 1),
        }
        # The ticked extras, with the same helpers and rounding as Sliding.
        if need_sd:
            pc = poincare_sd1_sd2(ser["rr_ms"], ser["valid"])
            row.update(SD1=round(float(pc["sd1"]), 3) if np.isfinite(pc["sd1"]) else np.nan,
                       SD2=round(float(pc["sd2"]), 3) if np.isfinite(pc["sd2"]) else np.nan)
        if need_freq:
            lf_nu, hf_nu, lf_hf = window_lf_hf(
                ep_rp, fs, rr_lo, rr_hi, breaks, bands, t0)
            row.update(LF_nu=round(lf_nu, 1) if np.isfinite(lf_nu) else np.nan,
                       HF_nu=round(hf_nu, 1) if np.isfinite(hf_nu) else np.nan,
                       LF_HF=round(lf_hf, 3) if np.isfinite(lf_hf) else np.nan)
        rows.append({k: v for k, v in row.items()
                     if k not in optional or k in extra})
        # Progress throttled to every 5%
        pct = int((idx_ep + 1) / max(n_ep, 1) * 100)
        if progress_cb is not None and idx_ep % max(1, n_ep // 20) == 0:
            progress_cb(pct, f"Epoch {idx_ep + 1}/{n_ep}…")
    return {"rows": rows, "n_skipped": n_skipped}


def sliding_hrv(
    rpeaks: np.ndarray,
    fs: float,
    win_s: float,
    step_s: float,
    rr_lo: float,
    rr_hi: float,
    breaks: Any,
    bands: Any,
    need_freq: bool,
    need_article_poincare: bool,
    min_beats: int = MIN_WINDOW_BEATS,
    min_valid: int = MIN_VALID_INTERVALS,
    progress_cb: ProgressCb = None,
) -> dict:
    """HRV in a window of *win_s* s every *step_s* s, from the first peak.

    Gap-aware like epoch_hrv. HR, SDNN, RMSSD and pNN6 are computed here
    directly (not by hrv_time_domain: see the module text); SD1 / SD2 always;
    CCD and SD1_LW / SD2_LW if *need_article_poincare*; LF / HF n.u. and
    LF/HF if *need_freq*. Rows carry the window's mid time "t_mid" (s).

    Returns {"rows": [one dict per window kept], "n_skipped": windows with
    fewer than *min_beats* peaks or *min_valid* valid intervals}.
    """
    t_peaks  = rpeaks / fs
    t_start  = t_peaks[0]
    t_end    = t_peaks[-1]
    starts   = np.arange(t_start, t_end - win_s + step_s * 0.5, step_s)
    rows    = []
    n_skipped = 0
    n_wins  = len(starts)
    for i, t0 in enumerate(starts):
        t1   = t0 + win_s
        mask = (t_peaks >= t0) & (t_peaks < t1)
        ep   = rpeaks[mask]
        # Same inclusion floor as compute_epochs: a window with too
        # few peaks (detection dropout, long pause) or too few valid
        # intervals is skipped, not emitted as a row of NaN.
        if len(ep) < min_beats:
            n_skipped += 1
            continue
        # Gap-aware, like analyse_core: out-of-range RR intervals and
        # intervals spanning a removed beat are excluded, and a
        # successive difference is only taken between two adjacent
        # valid intervals (never across a gap).
        ser    = rr_interval_series(ep, fs, rr_lo, rr_hi, breaks=breaks)
        rr_all = ser["rr_ms"]
        rr     = ser["rri"]
        if len(rr) < min_valid:
            n_skipped += 1
            continue
        rr_a, rr_b = successive_valid_pairs(rr_all, ser["valid"])
        sdiff  = rr_b - rr_a
        diffs  = np.abs(sdiff)
        hr   = mean_hr_bpm(rr)                 # 60000 / mean NN
        sdnn = float(rr.std(ddof=1))        if len(rr) > 1 else np.nan
        rmssd = float(np.sqrt(np.mean(sdiff**2))) if len(sdiff) > 1 else np.nan
        pnn6  = float(100.0 * np.sum(diffs > MouseECG.PNN_THRESHOLD) / len(diffs)) \
                if len(diffs) else np.nan
        # Poincare SD1/SD2 -- the app-wide definition
        # (poincare_sd1_sd2, also used by the Non-linear panel,
        # Compare Segments and the Report), over successive valid
        # pairs only. Cheap enough to always compute. SD1_LW/SD2_LW
        # below are the explicitly named Ledoit-Wolf variant.
        _pc = poincare_sd1_sd2(rr_all, ser["valid"])
        sd1, sd2 = float(_pc["sd1"]), float(_pc["sd2"])
        article_poincare = (
            # every interval + the window's validity mask (bounds
            # and removed-beat breaks): pairs never straddle a gap
            compute_poincare_metrics(rr_all, ser["valid"]) if need_article_poincare else {
                "CCD": np.nan,
                "SD1_LW": np.nan,
                "SD2_LW": np.nan,
                "n_poincare_pairs": np.nan,
            }
        )
        lf_nu = hf_nu = lf_hf = np.nan
        if need_freq:
            lf_nu, hf_nu, lf_hf = window_lf_hf(
                ep, fs, rr_lo, rr_hi, breaks, bands, t0)
        rows.append({
            "t_mid": round(t0 + win_s / 2, 2),
            "HR":    round(hr,    1),
            "SDNN":  round(sdnn,  2),
            "RMSSD": round(rmssd, 2),
            "pNN6":  round(pnn6,  1),
            "SD1":   round(sd1, 3) if np.isfinite(sd1) else np.nan,
            "SD2":   round(sd2, 3) if np.isfinite(sd2) else np.nan,
            "CCD":   round(float(article_poincare["CCD"]), 3)
                     if np.isfinite(article_poincare["CCD"]) else np.nan,
            "SD1_LW": round(float(article_poincare["SD1_LW"]), 3)
                      if np.isfinite(article_poincare["SD1_LW"]) else np.nan,
            "SD2_LW": round(float(article_poincare["SD2_LW"]), 3)
                      if np.isfinite(article_poincare["SD2_LW"]) else np.nan,
            "LF_nu": round(lf_nu, 1) if np.isfinite(lf_nu) else np.nan,
            "HF_nu": round(hf_nu, 1) if np.isfinite(hf_nu) else np.nan,
            "LF_HF": round(lf_hf, 3) if np.isfinite(lf_hf) else np.nan,
            "n_beats": len(ep),
            "n_valid": len(rr),
            "n_excluded": ser["n_excluded"],
            "n_poincare_pairs": article_poincare["n_poincare_pairs"],
        })
        pct = int((i + 1) / max(n_wins, 1) * 100)
        if progress_cb is not None and i % max(1, n_wins // 20) == 0:
            progress_cb(pct, f"Window {i + 1}/{n_wins}…")
    return {"rows": rows, "n_skipped": n_skipped}
