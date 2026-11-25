# -*- coding: utf-8 -*-
"""
ecg.app.wave_landmarks
----------------------
The P / QRS / T landmarks of the wave template as the editors show them
(colours, short labels, grab radius) and the drag rules of a landmark's
search window. Moved unchanged from the Tk WaveTemplateMiniEditor.
"""
from __future__ import annotations

from typing import Any, Optional

from ecg.app.theme import BLUE_DARK, CORAL, TEAL


def hit_landmark(landmarks: "Any", keys: "Any", x_ms: float,
                 radius_ms: float) -> "tuple[Optional[str], str]":
    """(landmark, "center" | "lo" | "hi") whose line is nearest *x_ms*, within *radius_ms*."""
    best_key: "Optional[str]" = None
    best_d   = float("inf")
    best_mode = "center"
    for key in keys:
        if key not in landmarks:
            continue
        c, hw = landmarks[key]
        for val, mode in [(c, "center"), (c - hw, "lo"), (c + hw, "hi")]:
            d = abs(x_ms - val)
            if d < best_d and d < radius_ms:
                best_d    = d
                best_key  = key
                best_mode = mode
    return best_key, best_mode


def dragged_landmark(center: float, half: float, mode: str, x_ms: float) -> "tuple[float, float]":
    """(centre, half-window) once the line of *mode* is at *x_ms*."""
    if mode == "center":
        return round(x_ms, 1), half
    if mode == "lo":
        return center, round(max(2.0, center - x_ms), 1)
    return center, round(max(2.0, x_ms - center), 1)


class WaveLandmarks:
    """The landmarks' names, colours and short labels (WaveTemplateMiniEditor's)."""

    _KEYS   = ["P_peak", "QRS_onset", "Q_peak", "S_peak", "QRS_offset",
               "J_peak", "T_peak"]
    _COLS   = {
        "P_peak":  BLUE_DARK,
        "QRS_onset": "#FFB74D",
        "Q_peak":  "#EF9A9A", "S_peak":  CORAL,
        "QRS_offset": "#BA68C8",
        "J_peak":  TEAL,   # teal — J wave / early repolarization
        "T_peak":  "#66BB6A",
    }
    _LABELS = {
        "P_peak": "P peak",
        "QRS_onset": "QRS onset",
        "Q_peak": "Q peak",  "S_peak": "S peak",
        "QRS_offset": "J point",
        "J_peak": "J peak",
        "T_peak": "T peak",
    }
    # For the plots that draw these waves.
    WAVE_COLORS       = _COLS
    WAVE_SHORT_LABELS = _LABELS
    _GRAB_R_MS = 5.0   # grab radius (ms) for click-on-line

