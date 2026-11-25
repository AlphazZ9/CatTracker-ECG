# -*- coding: utf-8 -*-
"""
ecg.services.agreement_service
------------------------------
Check agreement: the accepted beats cross-checked against the other,
independently implemented detectors run on the same filtered signal.

Moved unchanged from the Tk interface's DetectionController.
check_method_agreement (its background worker), so both interfaces count
the same beats. No UI import: safe in a background thread.

A beat that several DIFFERENT algorithms all miss is a far stronger "this
one is questionable" signal than any single method's own confidence.
"Auto" and the active method are left out: re-running the active method
against itself would trivially agree on every beat. A method that is
unavailable (no PyWavelets, no trained model, a genuine failure) is
skipped with its reason.
"""
from __future__ import annotations

import numpy as np

from ecg.core.detection import (detect_peaks_envelope_max, detect_peaks_sg_derivative,
                                detect_peaks_wavelet, peak_agreement_counts)
from ecg.core.ml_detector import MLPeakModel, NoModelError, detect_peaks_ml

TOLERANCE_S = 0.010     # 10 ms snap tolerance


def method_agreement(sig: np.ndarray, fs: float, primary: np.ndarray,
                     min_rr_ms: float, active: str) -> dict:
    """*active*: the Method box's text, lower case. Returns {"ok": False,
    "methods_skipped"} when no other method ran, else the counts:
    methods_run, methods_skipped, n_methods, n_beats, n_disagree, n_full,
    histogram (beats confirmed by k = 0..n_methods methods) and
    disagree_samples (the accepted peaks no other method found)."""
    tolerance_samples = max(1, int(round(TOLERANCE_S * fs)))
    other_peaks: "list[np.ndarray]" = []
    methods_run: "list[str]" = []
    methods_skipped: "list[str]" = []

    def _try(name: str, active_key: str, fn) -> None:
        if active_key in active:
            methods_skipped.append(f"{name} (already the active method)")
            return
        try:
            peaks, _proms, _t = fn()
            other_peaks.append(np.asarray(peaks))
            methods_run.append(f"{name} ({len(peaks)} peaks)")
        except Exception as exc:
            methods_skipped.append(f"{name} ({exc})")

    _try("Wavelet", "wavelet",
         lambda: detect_peaks_wavelet(sig, fs, min_rr_ms=min_rr_ms))
    _try("SG+Derivative", "sg",
         lambda: detect_peaks_sg_derivative(sig, fs, min_rr_ms=min_rr_ms))
    _try("Envelope Max", "envelope",
         lambda: detect_peaks_envelope_max(sig, fs, min_rr_ms=min_rr_ms))
    if "ml" in active or "machine" in active:
        methods_skipped.append("ML Detector (already the active method)")
    else:
        try:
            model = MLPeakModel.load()
            peaks, _proms, _t = detect_peaks_ml(sig, fs, model)
            other_peaks.append(np.asarray(peaks))
            methods_run.append(f"ML Detector ({len(peaks)} peaks)")
        except NoModelError:
            methods_skipped.append("ML Detector (no model trained yet)")
        except Exception as exc:
            methods_skipped.append(f"ML Detector ({exc})")

    if not other_peaks:
        return {"ok": False, "methods_skipped": methods_skipped}

    counts     = peak_agreement_counts(primary, other_peaks, tolerance_samples)
    n_methods  = len(other_peaks)
    n_disagree = int(np.sum(counts == 0))
    n_full     = int(np.sum(counts == n_methods))
    histogram  = [int(np.sum(counts == k)) for k in range(n_methods + 1)]
    return {
        "ok": True,
        "methods_run": methods_run,
        "methods_skipped": methods_skipped,
        "n_methods": n_methods,
        "n_beats": len(primary),
        "n_disagree": n_disagree,
        "n_full": n_full,
        "histogram": histogram,
        "disagree_samples": primary[counts == 0],
    }


def agreement_summary(result: dict) -> "tuple[str, float]":
    """The one-line summary and the % of beats confirmed by >= 1 other
    method (DetectionController's _done)."""
    n_beats, n_disagree = result["n_beats"], result["n_disagree"]
    pct_confirmed = 100.0 * (n_beats - n_disagree) / max(n_beats, 1)
    summary = (f"{result['n_methods']} other method(s) · "
               f"{pct_confirmed:.0f}% beats confirmed by ≥1 other · "
               f"{n_disagree} unconfirmed")
    return summary, pct_confirmed
