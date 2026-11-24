# -*- coding: utf-8 -*-
"""
ecg.services.detection_service
------------------------------
Detect Peaks without the UI: load the recording, crop it, read its Spike2
markers, prepare the signal (ecg.core.pipeline) and keep the candidates over
the threshold.

Moved unchanged from ecg.ui.signal_controller (the body of
SignalController.preview_worker). The reference outputs of tests/regression
check it is the same computation.

Nothing here may import from ecg.ui.
"""
from __future__ import annotations

from typing import Callable, Optional

from ecg.core.analysis import signal_crop_info
from ecg.core.detection import apply_threshold
from ecg.core.pipeline import prepare_signal
from ecg.io.loaders import load_mat_signal, read_spike2_markers


def load_and_detect(path: str, params: dict,
                    progress_cb: "Optional[Callable[[int, str], None]]" = None) -> dict:
    """Detect Peaks on the recording at *path* with the parameter snapshot
    *params* (FilterParams.to_dict() keys, plus sg_window_ms): the sampling
    rate read from the file when it has one (else params["fs"]), the time
    crop params["t_start"] / ["t_end"] (s, 0 = none), the prepared signal and
    its candidates, and the peaks over params["thresh"].

    Returns a plain dict (fs, signal_raw, dur_s, signal_crop, markers,
    detected_ch, detected_fs, ..., signal_flt, all_cands, all_proms,
    rpeaks_ok, rpeaks_rej, thresh_amp). Raises ValueError for a flat signal
    (wrong channel).
    """
    def _prog(pct: int, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(pct, msg)

    _prog(2, "Loading signal from file…")
    sig, detected_ch, _, detected_fs = load_mat_signal(
        path, params["channel"])

    # Determine effective fs
    if detected_fs is not None:
        fs = int(detected_fs)
    else:
        fs = params["fs"]

    # Estimation automatique min_rr supprimée.

    t0 = params["t_start"]
    t1 = params["t_end"]
    i0 = int(t0 * fs) if t0 > 0 else 0
    i1 = int(t1 * fs) if t1 > 0 else len(sig)
    # The load-time crop in file time: every time in the reports is
    # relative to its start (disclosed as the "Signal crop" row).
    signal_crop = signal_crop_info(t0, t1, len(sig) / fs if fs else None)
    sig = sig[i0:i1]
    # Spike2 marker channels in this signal's time base ({} otherwise).
    markers = read_spike2_markers(path, t0, t1)

    if sig.std() < 1e-10:
        raise ValueError("Signal is flat — wrong channel.")

    n_samples = len(sig)
    dur_s     = n_samples / fs
    _prog(5, f"Signal loaded — {dur_s:.0f} s  ({n_samples:,} samples)")

    # _compute_preview_bundle does all DSP — pure, no self writes
    def _prep_prog(pct: int, msg: str) -> None:
        _prog(5 + int(pct * 0.85), msg)

    signal_bundle = prepare_signal(sig, fs, params, _prep_prog)

    # Run threshold detection on the computed candidates (pure computation)
    thresh = params["thresh"]
    accepted, rejected, thresh_amp = apply_threshold(
        signal_bundle["signal_flt"],
        signal_bundle["all_cands"],
        signal_bundle["all_proms"],
        thresh,
        fs=fs,
    )

    _prog(100, "Done")

    return {
        # Signal identity
        "fs":              fs,
        "signal_raw":      sig,
        "dur_s":           dur_s,
        "signal_crop":     signal_crop,
        "markers":         markers,
        "detected_ch":     detected_ch,
        "detected_fs":     detected_fs,
        "requested_ch":    params["channel"],
        "fs_from_file":    detected_fs is not None,
        "thresh":          thresh,
        # Prepared signal bundle
        **signal_bundle,
        # Detection results
        "rpeaks_ok":       accepted,
        "rpeaks_rej":      rejected,
        "thresh_amp":      thresh_amp,
        "recommended_min_rr_ms": None,
    }
