# -*- coding: utf-8 -*-
"""
ecg.core.fast_dfa
-----------------
NeuroKit2's (multifractal) detrended fluctuation analysis, the same numbers
in a fraction of the time.

nk.fractal_dfa cuts the integrated RR series into overlapping windows with a
Python list of slices, fits each window's line with np.polyfit (vectorised),
then evaluates every line with np.polyval in a Python loop, one window at a
time. With the long-term scales of HRV DFA (12 to N / 10 beats: ~3 000
scales for an hour of mouse ECG) that loop is most of non-linear HRV's run
time (16 of 21 s for an hour). Here the windows are a strided view (the same
values) and the lines are evaluated for all windows at once by the same
Horner scheme as np.polyval (the same operations in the same order: the
same floats). np.polyfit, the fluctuation function, the slopes and the
singularity spectrum stay NeuroKit's. The detrended variances of each window
are kept for the last series seen (per thread): HRV's MFDFA then reuses those
of its DFA on the same windows instead of fitting them again.

``install()`` checks the replacement against NeuroKit's own result on a few
series first (exact equality of every fluctuation) and leaves NeuroKit
untouched if anything differs.

Nothing here may import from ecg.ui.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger("ecg")

_LOCK = threading.Lock()
_STATE = {"installed": None}     # None: not tried; True / False: result
# NeuroKit's own functions, kept when they are replaced (the tests compare).
ORIGINALS: "dict" = {}


# HRV DFA runs NeuroKit's DFA twice on the same windows (DFA alpha2, then
# MFDFA alpha2): the detrended variance of each window is the same in both.
# Kept for the last integrated series seen (a fingerprint of its bytes), per
# (window, order, overlap): ~3 MB for an hour of beats. One cache per thread:
# two analyses running at once (Analyze, Compare segments) never share it.
_TLS = threading.local()


def _state() -> dict:
    st = getattr(_TLS, "state", None)
    if st is None:
        st = _TLS.state = {"obj": None, "fp": None, "key": None, "pending": None,
                           "cache": {}}
    return st


class _Cached:
    """What trends() returns when the window's variances are cached: the
    fluctuation then uses them instead of recomputing the fit."""

    __slots__ = ("key",)

    def __init__(self, key: tuple) -> None:
        self.key = key


def _fingerprint(signal: np.ndarray) -> bytes:
    import hashlib
    a = np.ascontiguousarray(signal)
    return hashlib.blake2b(a.tobytes(), digest_size=16).digest() + str(a.dtype).encode()


def getwindow(signal: np.ndarray, n: int, window: int, overlap: bool = True) -> np.ndarray:
    """nk's _fractal_dfa_getwindow: the windows of *window* samples, starting
    every window // 2 (overlap) or back to back."""
    st = _state()
    if st["obj"] is not signal:
        fp = _fingerprint(signal)
        if fp != st["fp"]:
            st["cache"].clear()                  # another series: start over
        st["obj"], st["fp"] = signal, fp
    st["key"] = (st["fp"], int(window), bool(overlap))
    if overlap:
        starts = np.arange(0, n - window, window // 2)
        view = np.lib.stride_tricks.sliding_window_view(np.asarray(signal), window)
        return np.array(view[starts])
    segments = signal[: n - (n % window)]
    return segments.reshape((signal.shape[0] // window, window))


def trends(segments: np.ndarray, window: int, order: int = 1):
    """nk's _fractal_dfa_trends: each window's least-squares polynomial
    (np.polyfit, as NeuroKit), evaluated on 0..window-1 by np.polyval's Horner
    scheme for every window at once. When the windows' variances are cached
    (the same series and window seen before), a marker for fluctuation()."""
    st = _state()
    key = None if st["key"] is None else st["key"] + (int(order),)
    if key is not None and key in st["cache"]:
        return _Cached(key)
    st["pending"] = key                       # fluctuation() stores the variances under it
    x = np.arange(window)
    coefs = np.polyfit(x[:window], segments.T, order).T          # (n_windows, order + 1)
    y = np.zeros((coefs.shape[0], window), dtype=np.result_type(coefs, x))
    for k in range(coefs.shape[1]):                              # np.polyval: y = y * x + c
        y = y * x + coefs[:, k][:, None]
    return y


def fluctuation(segments, trends_, q=2):
    """nk's _fractal_dfa_fluctuation (NeuroKit 0.3.0), the variances of the
    detrended windows computed once per series and window."""
    from warnings import warn
    st = _state()
    if isinstance(trends_, _Cached):
        var = st["cache"][trends_.key]
    else:
        detrended = segments - trends_
        var = np.var(detrended, axis=1)
        key, st["pending"] = st["pending"], None
        if key is not None:
            st["cache"][key] = var
    var = var[var > 1e-08 * np.max(var)]
    if len(var) == 0:
        warn("All detrended segments have no variance. Retuning NaN.")
        return np.nan
    is0 = np.abs(q) < 0.1
    q_non0 = q[~is0].reshape(-1, 1)
    fluctuation_ = np.float_power(np.mean(np.float_power(var, q_non0 / 2), axis=1),
                                  1 / q_non0.T)
    if np.sum(is0) > 0:
        fluc0 = np.exp(0.5 * np.mean(np.log(var)))
        fluctuation_ = np.insert(fluctuation_, np.where(is0)[0], [fluc0])
    return fluctuation_


def _self_check(module, orig: tuple) -> bool:
    """The same fluctuations as NeuroKit's (*orig*: its three functions),
    bit for bit, on RR-like series, a series with flat stretches and a short
    one, for DFA and MFDFA."""
    rng = np.random.default_rng(0)
    series = [170 + 3 * rng.standard_normal(900),
              np.round(170 + 3 * rng.standard_normal(500)),
              np.concatenate([np.full(60, 120.0), 120 + rng.standard_normal(200)]),
              rng.standard_normal(40)]
    fast = (getwindow, trends, fluctuation)

    def use(fns) -> None:
        (module._fractal_dfa_getwindow, module._fractal_dfa_trends,
         module._fractal_dfa_fluctuation) = fns
    for x in series:
        n = len(x)
        scales = [np.arange(4, 12), np.arange(12, max(13, n // 10))]
        for scale in scales:
            if len(scale) < 2 or scale.max() >= n:      # NeuroKit refuses these
                continue
            for mf, q in ((False, "default"), (True, np.arange(-5, 6))):
                try:
                    use(orig)
                    _, a = module.fractal_dfa(x, multifractal=mf, q=q, scale=scale)
                    use(fast)                 # twice: the second run reads the cache
                    _, b = module.fractal_dfa(x, multifractal=mf, q=q, scale=scale)
                    _, c = module.fractal_dfa(x, multifractal=mf, q=q, scale=scale)
                finally:
                    use(orig)
                fa = np.asarray(a["Fluctuations"])
                for other in (b, c):
                    fb = np.asarray(other["Fluctuations"])
                    if fa.shape != fb.shape or not np.array_equal(fa, fb, equal_nan=True):
                        return False
    return True


def install() -> bool:
    """Swap NeuroKit's window cutting and trend evaluation for these, once;
    False (NeuroKit unchanged) if its internals are not where expected or a
    fluctuation of the self-check differs."""
    with _LOCK:
        if _STATE["installed"] is not None:
            return bool(_STATE["installed"])
        try:
            import warnings
            import importlib
            # The module (the package exports the function under its name).
            module = importlib.import_module("neurokit2.complexity.fractal_dfa")
            orig = (module._fractal_dfa_getwindow, module._fractal_dfa_trends,
                    module._fractal_dfa_fluctuation)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ok = _self_check(module, orig)
        except Exception as exc:
            log.debug("fast DFA not installed: %s", exc)
            ok, module = False, None
        if ok:
            ORIGINALS.update(module=module, getwindow=orig[0], trends=orig[1],
                             fluctuation=orig[2])
            module._fractal_dfa_getwindow = getwindow
            module._fractal_dfa_trends = trends
            module._fractal_dfa_fluctuation = fluctuation
        else:
            log.info("fast DFA not installed: NeuroKit's own DFA is used")
        _STATE["installed"] = ok
        return ok
