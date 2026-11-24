"""Annotations suggested from the other channels of a recording: the
marker / keyboard channels (a key pressed at an injection, a stimulator's
markers) and the waveform channels carrying stimulus pulses (TTL). The user
picks which to add; nothing is added by itself.

A suggestion is a dict {t_start, t_end, label, source}: a train of pulses
or markers (closer than *gap_s*) spans the train, a lone marker lasts
*point_s* (an annotation has a start and an end; the start aligns the
time course and the event-locked analysis)."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import numpy as np

GAP_S = 2.0          # markers / pulses closer than this make one train
POINT_S = 1.0        # duration given to a lone marker
SAME_S = 0.5         # a suggestion this close to an annotation's start is there already


def _trains(times: np.ndarray, gap_s: float) -> "list[np.ndarray]":
    t = np.sort(np.asarray(times, dtype=np.float64))
    if not len(t):
        return []
    cut = np.flatnonzero(np.diff(t) > gap_s) + 1
    return np.split(t, cut)


def _span(tr: np.ndarray, point_s: float, t_max: "Optional[float]") -> "tuple[float, float]":
    a = float(tr[0])
    if len(tr) > 1:
        b = float(tr[-1] + np.median(np.diff(tr)))
    else:
        b = a + point_s
    if t_max is not None:
        b = min(b, float(t_max))
    if b <= a:
        b = a + 1e-3            # an annotation ends after it starts
    return round(a, 3), round(b, 3)


def _rate_text(tr: np.ndarray) -> str:
    if len(tr) < 3:
        return ""
    hz = 1.0 / float(np.median(np.diff(tr)))
    return f", {hz:.3g} Hz" if np.isfinite(hz) else ""


def from_events(events: "Mapping[str, Iterable[tuple[float, str]]]",
                gap_s: float = GAP_S, point_s: float = POINT_S,
                t_max: "Optional[float]" = None) -> "list[dict]":
    """Suggestions from marker channels {channel: [(time, code)]}: per channel
    and code, one per train."""
    out: "list[dict]" = []
    for ch, evs in events.items():
        by_code: "dict[str, list[float]]" = {}
        for t, code in evs:
            by_code.setdefault(str(code or ""), []).append(float(t))
        for code, times in by_code.items():
            name = f"{ch} '{code}'" if code else str(ch)
            for tr in _trains(np.asarray(times), gap_s):
                a, b = _span(tr, point_s, t_max)
                label = name if len(tr) == 1 else f"{name} ({len(tr)} markers{_rate_text(tr)})"
                out.append({"t_start": a, "t_end": b, "label": label, "source": str(ch),
                            "code": code})
    return sorted(out, key=lambda d: d["t_start"])


def pulse_onsets(sig: np.ndarray, fs: float) -> np.ndarray:
    """Rising edges (s) of a channel that is a train of pulses (two levels,
    mostly low), empty for any other signal (an ECG, a noisy line)."""
    x = np.asarray(sig, dtype=np.float64)
    if not np.isfinite(x).all():
        x = np.where(np.isfinite(x), x, np.nanmin(x) if np.isfinite(x).any() else 0.0)
    if len(x) < 10 or not fs > 0:
        return np.array([])
    # The high level is the max: pulses can be a tiny part of a long record.
    lo, hi = float(np.percentile(x, 1)), float(np.max(x))
    rng = hi - lo
    if not rng > 0:
        return np.array([])
    rel = (x - lo) / rng
    # Two levels: nearly every sample near the low or the high one.
    near = (rel < 0.2) | (rel > 0.8)
    high = rel > 0.5
    if near.mean() < 0.97 or not 0 < high.mean() < 0.5:
        return np.array([])
    up = np.flatnonzero(~high[:-1] & high[1:]) + 1
    if not len(up) or len(up) > len(x) / 4:
        return np.array([])
    return up / float(fs)


def from_waveforms(waves: "Mapping[str, tuple[np.ndarray, float]]", gap_s: float = GAP_S,
                   point_s: float = POINT_S, t_max: "Optional[float]" = None) -> "list[dict]":
    """Suggestions from waveform channels {channel: (samples, fs)}: one per
    train of pulses."""
    out: "list[dict]" = []
    for ch, (sig, fs) in waves.items():
        on = pulse_onsets(sig, fs)
        for tr in _trains(on, gap_s):
            a, b = _span(tr, point_s, t_max)
            n = len(tr)
            label = f"{ch} ({n} pulse{'s' if n != 1 else ''}{_rate_text(tr)})"
            out.append({"t_start": a, "t_end": b, "label": label, "source": str(ch),
                        "code": ""})
    return sorted(out, key=lambda d: d["t_start"])


def drop_existing(suggestions: "Iterable[dict]", annotations: "Iterable[Mapping[str, Any]]",
                  same_s: float = SAME_S) -> "list[dict]":
    """The suggestions no annotation starts at already (within *same_s*)."""
    starts = []
    for a in annotations or []:
        try:
            starts.append(float(a["t_start"]))
        except (KeyError, TypeError, ValueError):
            continue
    st = np.sort(np.asarray(starts))
    out = []
    for s in suggestions:
        t = float(s["t_start"])
        if len(st):
            i = np.searchsorted(st, t)
            near = [abs(st[j] - t) for j in (i - 1, i) if 0 <= j < len(st)]
            if near and min(near) <= same_s:
                continue
        out.append(s)
    return out
