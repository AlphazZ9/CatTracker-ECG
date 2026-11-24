# -*- coding: utf-8 -*-
"""
ecg.core.pipeline
-----------------
Signal preparation and R-peak candidate detection -- what Detect Peaks runs
before the threshold: normalisation, band-pass, notch, NeuroKit2 ecg_clean,
manual / automatic polarity, then one of the five detectors (Auto,
SG + Derivative, Wavelet, Envelope Max, ML).

Moved unchanged from ecg.ui.signal_controller (SignalController.
compute_preview_bundle -> prepare_signal, SignalController._preview_bundle
-> detect_candidates, the filtering part of compute_filter_preview_segment
-> filter_preview) so it runs without the UI, from a script or a worker.
The reference outputs of tests/regression check it is the same computation.

Nothing here may import from ecg.ui.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from ecg.core.detection import (
    fix_polarity,
    detect_peaks_sg_derivative, detect_peaks_wavelet, detect_peaks_envelope_max,
)
from ecg.core.filtering import bandpass, filter_description, notch, normalize
from ecg.core.ml_detector import NO_MODEL_STATUS, MLPeakModel, NoModelError, detect_peaks_ml
from ecg.core.models import FilterParams, MouseECG

# Detected here rather than imported from ecg.ui.theme (the same neurokit2
# module): see the note in ecg.core.analysis.
try:
    import neurokit2 as nk
except ImportError:
    nk = None

log = logging.getLogger("ecg")


def prepare_signal(
    sig_raw:     np.ndarray,
    fs:          int,
    params:      dict,
    progress_cb: "Optional[Callable[[int, str], None]]" = None,
) -> dict:
    """``detect_candidates`` plus ``filter_text``: the filters applied
    (filter_description at *fs*, i.e. the cut-offs bandpass really
    uses), stored as SignalState.filter_text so the reports state the
    filters the peaks were detected on -- and ``prepare_params``, the
    FilterParams snapshot they were prepared with (SignalState.
    prepare_params): a session restore re-prepares the signal from it
    rather than from widget values edited after detection."""
    bundle = detect_candidates(sig_raw, fs, params, progress_cb)
    bundle["filter_text"] = filter_description(params, fs)
    try:
        bundle["prepare_params"] = FilterParams.from_dict(params).to_dict()
        # sg_window_ms is a Recording entry, not a FilterParams field: kept
        # beside it so a session restore rebuilds the SG + Derivative
        # candidates with the window they were detected with.
        if params.get("sg_window_ms") is not None:
            bundle["prepare_params"]["sg_window_ms"] = params["sg_window_ms"]
    except (TypeError, ValueError):
        bundle["prepare_params"] = None
    return bundle


def detect_candidates(
    sig_raw:     np.ndarray,
    fs:          int,
    params:      dict,
    progress_cb: "Optional[Callable[[int, str], None]]" = None,
) -> dict:
    """Pure computation — no access to *self*, no Tkinter calls.

    Filters, normalises, fixes polarity, and finds all R-peak candidates.
    Returns a plain dict that ``SignalController.on_preview_done`` writes
    to the app state atomically on the main thread.

    This separation is the thread-safety contract:
    * Background workers call this method and return the bundle.
    * Only the main thread writes instance variables.
    """
    def _prog(pct: int, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)

    no_filter = params.get("no_filter", False)
    sig = sig_raw.copy()

    _prog(5, "Normalising raw signal…")
    signal_raw_norm = normalize(sig)

    if no_filter:
        log.info("_compute_preview_bundle: no_filter=True — skipping bandpass/notch/ecg_clean")
    else:
        _prog(10, "Bandpass filtering…")
        try:
            sig = bandpass(sig, fs, params["lp"], params["hp"])
        except Exception as exc:
            log.warning("bandpass skipped: %s", exc)

        if params["notch"]:
            _prog(25, "Notch filtering…")
            try:
                sig = notch(sig, fs)
            except Exception as exc:
                log.warning("notch skipped: %s — check fs vs notch frequency", exc)

        _prog(35, "NeuroKit2 clean…")
        try:
            assert nk is not None  # NK_AVAILABLE guard checked by caller
            sig = nk.ecg_clean(sig, sampling_rate=fs,
                               method=params["clean_method"])
        except Exception as exc:
            log.warning("nk.ecg_clean skipped: %s", exc)

    _prog(45, "Normalising filtered signal…")
    sig = normalize(np.asarray(sig, dtype=float))

    # ── Manual polarity override ─────────────────────────────────────
    # If the user has explicitly toggled "Inverser signal", flip the
    # signal before auto-polarity detection.  fix_polarity will then
    # find the (now correct) positive R peaks and not re-flip.
    if params.get("invert_signal", False):
        sig = -sig
        log.info("_compute_preview_bundle: user-requested signal inversion applied")

    # Raw/no-filter mode: if the user explicitly asked to detect on the
    # unfiltered signal and hasn't manually inverted it, honor that
    # choice by forcing fix_polarity to leave the signal as-is instead
    # of silently re-flipping it via the automatic polarity vote.
    force_pol = False if (no_filter and not params.get("invert_signal", False)) else None

    def _polarity_prog(pct: int, msg: str) -> None:
        _prog(45 + int(pct * 0.50), msg)

    # ── Detection method ─────────────────────────────────────────────
    det_method = params.get("detection_method", "auto")

    if "wavelet" in det_method.lower() or "cwt" in det_method.lower():
        # ── Wavelet (CWT) pipeline ────────────────────────────────────
        # fix_polarity AVANT la détection : le détecteur wavelet (comme SG)
        # travaille sur la dérivée positive → il faut que les R-peaks soient
        # positifs dans le signal, sinon seuls les artefacts positifs sont détectés.
        #
        # NOTE: unlike the SG-derivative/Envelope-Max/ML branches below,
        # this call is NOT guarded by `if not params.get("invert_signal")`
        # -- it always runs, even if the user already inverted manually.
        # That asymmetry is intentional-or-unconfirmed (not verified
        # against real double-inverted recordings) -- don't "fix" it by
        # assuming it should match the other three methods without
        # testing what Wavelet detection actually does on a
        # user-already-inverted signal first.
        _prog(48, "Polarity correction…")
        sig, inverted, _, _ = fix_polarity(sig, fs, params["min_rr_ms"], force_polarity=force_pol)
        _prog(50, "CWT — separating noise / QRS / J-wave…")
        try:
            peaks_wt, proms_wt, t_amp_wt = detect_peaks_wavelet(
                sig,
                fs=fs,
                min_rr_ms=params["min_rr_ms"],
                peak_distance_ms=params.get("peak_distance_ms", MouseECG.PEAK_DISTANCE_MS),
            )
        except ImportError:
            log.warning(
                "PyWavelets non installé — "
                "pip install PyWavelets  (fallback: auto)"
            )
            peaks_wt  = np.array([], dtype=int)
            proms_wt  = np.array([])
            t_amp_wt  = 0.0
        except Exception as exc:
            log.warning("Wavelet detection failed, falling back to auto: %s", exc)
            peaks_wt  = np.array([], dtype=int)
            proms_wt  = np.array([])
            t_amp_wt  = 0.0

        _prog(95, f"Wavelet candidates: {len(peaks_wt):,}")
        return {
            "signal_raw_norm": signal_raw_norm,
            "signal_flt":      sig,
            "no_filter_mode":  no_filter,
            "all_cands":       peaks_wt,
            "all_proms":       proms_wt if len(proms_wt) else np.ones(len(peaks_wt)),
            "inverted":        inverted,
        }

    if "sg" in det_method.lower() or "deriv" in det_method.lower():
        # ── SG + Derivative pipeline ──────────────────────────────────
        # fix_polarity automatique si l'utilisateur n'a pas déjà inversé
        # manuellement (évite la double inversion). Le détecteur SG
        # requiert des R positifs (upstroke = dérivée positive).
        if not params.get("invert_signal", False):
            _prog(48, "Polarity correction…")
            sig, inverted, _, _ = fix_polarity(sig, fs, params["min_rr_ms"], force_polarity=force_pol)
        else:
            inverted = False  # utilisateur a géré la polarité manuellement
        _prog(50, "SG derivative detection…")
        try:
            sg_window_ms = float(params.get("sg_window_ms", 20.0))
        except (TypeError, ValueError):
            sg_window_ms = 20.0

        try:
            peaks_sg, proms_sg, t_amp_sg = detect_peaks_sg_derivative(
                sig,
                fs=fs,
                sg_window_ms=sg_window_ms,
                min_rr_ms=params["min_rr_ms"],
                peak_distance_ms=params.get("peak_distance_ms", MouseECG.PEAK_DISTANCE_MS),
                target_fs=float(params.get("sg_target_fs", 10000)),
            )
        except Exception as exc:
            log.warning("SG+derivative detection failed, falling back to auto: %s", exc)
            peaks_sg = np.array([], dtype=int)
            proms_sg = np.array([])
            t_amp_sg = 0.0

        _prog(95, f"SG+Deriv candidates: {len(peaks_sg):,}")
        return {
            "signal_raw_norm": signal_raw_norm,
            "signal_flt":      sig,
            "no_filter_mode":  no_filter,
            "all_cands":       peaks_sg,
            "all_proms":       proms_sg if len(proms_sg) else np.ones(len(peaks_sg)),
            "inverted":        False,
        }

    if "envelope" in det_method.lower() or "max" in det_method.lower():
        # ── Envelope Max pipeline ─────────────────────────────────────
        # Détection par maximum local d'amplitude — robuste aux signaux
        # saturés (ADC clipping) et aux morphologies atypiques où la
        # dérivée SG est peu discriminante.
        # fix_polarity requis : le détecteur sélectionne les maxima → les
        # R-peaks doivent être des extrema positifs dans le signal.
        if not params.get("invert_signal", False):
            _prog(48, "Polarity correction…")
            sig, inverted, _, _ = fix_polarity(sig, fs, params["min_rr_ms"], force_polarity=force_pol)
        else:
            inverted = False
        _prog(55, "Envelope Max detection…")
        try:
            peaks_em, proms_em, t_amp_em = detect_peaks_envelope_max(
                sig,
                fs=fs,
                min_rr_ms=params["min_rr_ms"],
                peak_distance_ms=params.get("peak_distance_ms", MouseECG.PEAK_DISTANCE_MS),
            )
        except Exception as exc:
            log.warning("Envelope Max detection failed, falling back to auto: %s", exc)
            peaks_em = np.array([], dtype=int)
            proms_em = np.array([])
            t_amp_em = 0.0

        _prog(95, f"Envelope Max candidates: {len(peaks_em):,}")
        return {
            "signal_raw_norm": signal_raw_norm,
            "signal_flt":      sig,
            "no_filter_mode":  no_filter,
            "all_cands":       peaks_em,
            "all_proms":       proms_em if len(proms_em) else np.ones(len(peaks_em)),
            "inverted":        inverted,
        }

    if "ml" in det_method.lower() or "machine" in det_method.lower():
        # ── ML Detector pipeline ──────────────────────────────────────
        # Classifier-based detection, trained from the user's own
        # verified/corrected recordings (see ecg.core.ml_detector).
        # Same polarity requirement as the other detectors: features
        # (prominence, amplitude, upstroke slope...) all assume the
        # R-wave is a positive-going deflection.
        if not params.get("invert_signal", False):
            _prog(48, "Polarity correction…")
            sig, inverted, _, _ = fix_polarity(sig, fs, params["min_rr_ms"], force_polarity=force_pol)
        else:
            inverted = False
        detector_warning: Optional[str] = None
        ml_no_model = False

        def _ml_prog(pct: int, msg: str) -> None:
            # Rescale detect_peaks_ml's own 0-100 stage progress into the
            # 55-95 slice of the overall preview-bundle progress bar, so
            # its checkpoints (crucially, the candidate count) actually
            # reach the UI instead of sitting on one static message for
            # however long candidate generation + feature extraction take.
            _prog(55 + int(pct * 0.40), msg)

        try:
            model = MLPeakModel.load()
            peaks_ml, proms_ml, t_amp_ml = detect_peaks_ml(sig, fs, model, progress_cb=_ml_prog)
        except NoModelError as exc:
            # No trained model yet: the status bar points at the panel's
            # ML section, which opens and says what to do (the whole
            # NO_MODEL_MESSAGE does not fit the status bar).
            log.warning("ML detector: %s", exc)
            detector_warning = NO_MODEL_STATUS
            ml_no_model = True
            peaks_ml = np.array([], dtype=int)
            proms_ml = np.array([])
            t_amp_ml = 0.0
        except RuntimeError as exc:
            # Degenerate signal -- a clearly different situation from
            # "detection failed on this signal", surfaced distinctly so
            # the user isn't left guessing why 0 peaks were found.
            log.warning("ML detector: %s", exc)
            detector_warning = str(exc)
            peaks_ml = np.array([], dtype=int)
            proms_ml = np.array([])
            t_amp_ml = 0.0
        except Exception as exc:
            log.warning("ML detection failed, falling back to auto: %s", exc)
            peaks_ml = np.array([], dtype=int)
            proms_ml = np.array([])
            t_amp_ml = 0.0

        _prog(95, f"ML Detector candidates: {len(peaks_ml):,}")
        return {
            "signal_raw_norm": signal_raw_norm,
            "signal_flt":      sig,
            "no_filter_mode":  no_filter,
            "all_cands":       peaks_ml,
            "all_proms":       proms_ml if len(proms_ml) else np.ones(len(peaks_ml)),
            "inverted":        inverted,
            "detector_warning": detector_warning,
            "ml_no_model":     ml_no_model,
        }

    # ── Auto (NeuroKit2) pipeline — original path ─────────────────────
    sig_out, inverted, cands, proms = fix_polarity(
        sig, fs, params["min_rr_ms"], progress_cb=_polarity_prog, force_polarity=force_pol)

    _prog(95, f"Candidates found: {len(cands):,}")

    return {
        "signal_raw_norm": signal_raw_norm,
        "signal_flt":      sig_out,
        "no_filter_mode":  no_filter,
        "all_cands":       cands,
        "all_proms":       proms,
        "inverted":        inverted,
    }


def filter_preview(
    raw:          np.ndarray,
    fs:           float,
    t_start:      float,
    t_end:        float,
    t_last:       float,
    lp:           float,
    hp:           float,
    notch_on:     bool,
    clean_method: str,
) -> "Optional[tuple[int, int, np.ndarray, np.ndarray]]":
    """Live filter preview of *raw* over [t_start, t_end] s with the given
    filter settings (band-pass *lp*-*hp* Hz, notch, ecg_clean method).

    Filters only the visible window plus a 1 s margin on each side (cut
    back off afterwards) that absorbs the filtfilt edge transients; never
    the whole recording. *t_last* is the time of the last sample (s).

    Returns (i0, i1, raw_norm, filtered_norm): the samples [i0, i1) of the
    recording shown and their z-scored raw and filtered values, or None
    when the window (with margins) is shorter than 0.5 s. A filter stage
    that fails is skipped, as in the preview it shows.
    """
    margin = 1.0  # seconds — absorbs filtfilt edge transients
    lo_t = max(0.0, t_start - margin)
    hi_t = min(float(t_last), t_end + margin)
    lo_i = int(lo_t * fs)
    hi_i = int(hi_t * fs)
    if hi_i - lo_i < int(0.5 * fs):
        return None  # too short to filter meaningfully

    seg_raw = np.asarray(raw[lo_i:hi_i], dtype=float)

    seg = seg_raw.copy()
    try:
        seg = bandpass(seg, fs, lp, hp)
    except Exception as exc:
        log.debug("filter preview: bandpass skipped — %s", exc)
    if notch_on:
        try:
            seg = notch(seg, fs)
        except Exception as exc:
            log.debug("filter preview: notch skipped — %s", exc)
    try:
        if nk is not None:
            seg = nk.ecg_clean(seg, sampling_rate=fs, method=clean_method)
    except Exception as exc:
        log.debug("filter preview: ecg_clean skipped — %s", exc)

    seg      = normalize(np.asarray(seg, dtype=float))
    raw_norm = normalize(seg_raw)

    # Trim the margin back off — only the visible window is returned
    off0 = int(round((t_start - lo_t) * fs))
    off1 = off0 + int(round((t_end - t_start) * fs))
    off1 = min(off1, len(seg), len(raw_norm))
    off0 = min(off0, off1)
    return lo_i + off0, lo_i + off1, raw_norm[off0:off1], seg[off0:off1]
