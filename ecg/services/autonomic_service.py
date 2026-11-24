# -*- coding: utf-8 -*-
"""
ecg.services.autonomic_service
------------------------------
The Autonomic view's computation without the UI: time-resolved CSI / CPI
(Candia-Rivera et al. 2025) and, when asked, the SPWVD LF / HF of the same
peaks.

Moved unchanged from ecg.ui.analysis_controller (the worker of
AnalysisController.compute_autonomic). The reference outputs of
tests/regression check it is the same computation.

Peaks are sample indices; times in seconds; RR in ms.
Nothing here may import from ecg.ui.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from ecg.core.analysis import rr_interval_series
from ecg.core.autonomic import compute_autonomic_indices
from ecg.core.spwvd import spwvd_lf_hf

# SPWVD run time per second of recording, for the progress message (s / s).
SPWVD_S_PER_S = 0.012


def run_autonomic(
    rpeaks: np.ndarray,
    fs: float,
    p: dict,
    rr_lo: float,
    rr_hi: float,
    breaks: Any,
    lf_band: Any,
    hf_band: Any,
    window: "tuple[float, float]",
    peak_span: "tuple[float, float]",
    span_s: float,
    progress_cb: "Optional[Callable[[int, str], None]]" = None,
    spwvd_s_per_s: float = SPWVD_S_PER_S,
) -> dict:
    """CSI / CPI of *rpeaks* with the Autonomic parameters *p* (normalised:
    window_s, method, ks, kp, min_pairs, lf_hf), on the RR within
    [rr_lo, rr_hi] ms and not spanning *breaks*. *window* (the analysis
    window, s) and *peak_span* (first and last peak, s) are recorded in the
    result's params. With p["lf_hf"], also the SPWVD LF / HF in *lf_band* /
    *hf_band* (Hz) -- {"error": message} when the series is unusable (too
    short, a band outside the resampled range, the memory guard).

    Returns {"result": AutonomicResult, "spwvd": dict or None}.
    """
    def _progress(pct: int, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(pct, msg)

    _progress(5, "Poincaré windows…")
    res = compute_autonomic_indices(
        rpeaks, fs, window_s=p["window_s"], method=p["method"],
        ks=p["ks"], kp=p["kp"], rr_min_ms=rr_lo, rr_max_ms=rr_hi,
        breaks=breaks, min_pairs=p["min_pairs"])
    res.params["analysis_window"] = window
    res.params["peak_span_s"] = peak_span
    spwvd = None
    if p["lf_hf"]:
        est = span_s * spwvd_s_per_s
        _progress(30, "LF / HF (SPWVD)" + (f" — about {est:.0f} s…" if est >= 5 else "…"))
        ser = rr_interval_series(rpeaks, fs, rr_lo, rr_hi, breaks=breaks)
        try:
            spwvd = spwvd_lf_hf(ser["rr_ms"], ser["t_s"], ser["valid"],
                                lf_band=tuple(lf_band), hf_band=tuple(hf_band))
        except ValueError as exc:
            # Too short, a band outside the resampled range, the
            # memory guard: disclosed, the CSI / CPI still shown.
            spwvd = {"error": str(exc)}
    _progress(95, "Drawing…")
    return {"result": res, "spwvd": spwvd}
