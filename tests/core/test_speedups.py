"""The speed-ups give the same numbers: parallel delineation, the entropy
neighbour count, the threshold caches, the DFA windows, the Savitzky-Golay
smoothing of the delineation, the batched per-beat quantities of Detect and
Events."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecg.core import analysis, delineation_pool, detection, fast_dfa, fast_entropy, filtering
from ecg.core.wave_template import WaveTemplate
from tests.helpers import make_synth_ecg

FS = 2000.0


def _recording(n_beats: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    sig, fs, peaks = make_synth_ecg(n_beats=n_beats, fs=FS, rr_ms=120.0, noise_sd=0.03,
                                    seed=seed, p_amp=0.15)
    # Some RR variability and a T-like bump after each R, so every landmark
    # has something to find and the P search start (previous T) matters.
    t = np.arange(len(sig))
    for p in peaks:
        sig += 0.12 * np.exp(-0.5 * ((t - p - 0.035 * fs) / (0.006 * fs)) ** 2)
    sig += 0.01 * rng.standard_normal(len(sig))
    return sig, peaks


@pytest.mark.parametrize("n_ens", [1, 32])
def test_parallel_delineation_is_the_sequential_one(monkeypatch, n_ens):
    sig, peaks = _recording(1800)
    kw = dict(wave_template=WaveTemplate(), rr_max_ms=1000.0, ensemble_beats=n_ens)
    monkeypatch.setenv("ECG_DELIN_SERIAL", "1")
    serial = analysis.analyse_intervals(sig, peaks, FS, np.diff(peaks) / 2.0, **kw)
    monkeypatch.delenv("ECG_DELIN_SERIAL")
    monkeypatch.setattr(delineation_pool, "PARALLEL_MIN_ROWS", 100)
    parallel = analysis.analyse_intervals(sig, peaks, FS, np.diff(peaks) / 2.0, **kw)
    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
    # No warm-up: every chunk starts from an unknown T peak, so the joints
    # are repaired in the parent -- still the sequential rows.
    monkeypatch.setattr(delineation_pool, "WARMUP", 0)
    repaired = analysis.analyse_intervals(sig, peaks, FS, np.diff(peaks) / 2.0, **kw)
    pd.testing.assert_frame_equal(serial, repaired, check_exact=True)


def test_chebyshev_counts_are_the_kdtree_ones():
    from sklearn.neighbors import KDTree
    rng = np.random.default_rng(3)
    for n, m in ((500, 2), (300, 3), (40, 1)):
        x = np.round(rng.standard_normal((n, m)), 2)        # ties on the radius
        for tol in (0.05, 0.2, 0.5):
            want = KDTree(x, metric="chebyshev").query_radius(x, tol, count_only=True)
            assert np.array_equal(fast_entropy.chebyshev_counts(x, tol), want.astype(float))


def test_non_linear_hrv_is_unchanged_by_the_fast_count():
    from neurokit2.complexity import entropy_approximate, utils_entropy
    rng = np.random.default_rng(5)
    rr = 170 + np.cumsum(rng.standard_normal(700)) * 0.3 + rng.standard_normal(700)
    peaks = np.r_[0, np.cumsum(np.round(rr / 1000 * FS))].astype(int) + 1000
    assert fast_entropy.install()          # once: later calls change nothing
    fast = utils_entropy._get_count
    original = fast.__wrapped__
    try:
        utils_entropy._get_count = entropy_approximate._get_count = original
        a = analysis.analyse_hrv_nonlinear(peaks, FS)       # NeuroKit's own count
    finally:
        utils_entropy._get_count = entropy_approximate._get_count = fast
    b = analysis.analyse_hrv_nonlinear(peaks, FS)
    assert {"HRV_SampEn", "HRV_ApEn", "HRV_MSEn", "HRV_RCMSEn"} <= set(a.columns)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_threshold_caches_give_the_peaks_of_a_fresh_computation():
    sig, peaks = _recording(500, seed=2)
    cands = np.sort(np.r_[peaks, peaks[:-1] + 90])           # R peaks and T-ish decoys
    proms = np.abs(sig[cands]) + 0.01 * np.arange(len(cands)) % 0.05
    results = {}
    for t in (0.5, 0.3, 0.8, 0.5):
        results.setdefault(t, []).append(detection.apply_threshold(sig, cands, proms, t, fs=FS))
        # A fresh computation: empty caches.
        detection._REF_CACHE.update(cands=None, proms=None, key=None, refs=None)
        detection._J_CACHE.update(signal=None, key=None, slope={}, noise={})
        fresh = detection.apply_threshold(sig, cands, proms, t, fs=FS)
        got = results[t][-1]
        assert np.array_equal(got[0], fresh[0]) and np.array_equal(got[1], fresh[1])
        assert got[2] == fresh[2]
    # New arrays (another recording, an inverted signal) are not served the
    # old cache.
    a = detection.apply_threshold(-sig, cands, proms.copy(), 0.5, fs=FS)
    detection._REF_CACHE.update(cands=None)
    detection._J_CACHE.update(signal=None)
    b = detection.apply_threshold(-sig, cands, proms.copy(), 0.5, fs=FS)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# ── DFA ──────────────────────────────────────────────────────

def _rr_series(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 170 + np.cumsum(rng.standard_normal(n)) * 0.3 + rng.standard_normal(n)


def _neurokit_dfa():
    """NeuroKit's own DFA functions in place, for a reference run."""
    import contextlib
    assert fast_dfa.install()
    o = fast_dfa.ORIGINALS
    mod = o["module"]

    @contextlib.contextmanager
    def ctx():
        fast = (mod._fractal_dfa_getwindow, mod._fractal_dfa_trends,
                mod._fractal_dfa_fluctuation)
        (mod._fractal_dfa_getwindow, mod._fractal_dfa_trends,
         mod._fractal_dfa_fluctuation) = o["getwindow"], o["trends"], o["fluctuation"]
        try:
            yield
        finally:
            (mod._fractal_dfa_getwindow, mod._fractal_dfa_trends,
             mod._fractal_dfa_fluctuation) = fast
    return mod, ctx


def test_fast_dfa_gives_neurokits_fluctuations():
    mod, neurokit = _neurokit_dfa()
    for seed, n in ((1, 2500), (2, 300)):
        x = _rr_series(n, seed)
        for scale in (np.arange(4, 12), np.arange(12, n // 10)):
            for mf, q in ((False, "default"), (True, np.arange(-5, 6))):
                with neurokit():
                    _, want = mod.fractal_dfa(x, multifractal=mf, q=q, scale=scale)
                for _ in range(2):              # the second run reads the variance cache
                    _, got = mod.fractal_dfa(x, multifractal=mf, q=q, scale=scale)
                    assert np.array_equal(np.asarray(want["Fluctuations"]),
                                          np.asarray(got["Fluctuations"]), equal_nan=True)
    # Another series of the same length: not served the first one's variances.
    a, b = _rr_series(400, 7), _rr_series(400, 8)
    mod.fractal_dfa(a, scale=np.arange(4, 12))
    _, got = mod.fractal_dfa(b, scale=np.arange(4, 12))
    with neurokit():
        _, want = mod.fractal_dfa(b, scale=np.arange(4, 12))
    assert np.array_equal(np.asarray(want["Fluctuations"]), np.asarray(got["Fluctuations"]))


def test_non_linear_hrv_is_unchanged_by_the_fast_dfa():
    _mod, neurokit = _neurokit_dfa()
    rr = _rr_series(3000, 11)
    peaks = np.r_[0, np.cumsum(np.round(rr / 1000 * FS))].astype(int) + 1000
    with neurokit():
        a = analysis.analyse_hrv_nonlinear(peaks, FS)
    b = analysis.analyse_hrv_nonlinear(peaks, FS)
    assert {"HRV_DFA_alpha1", "HRV_DFA_alpha2"} <= set(a.columns)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_fast_dfa_caches_per_thread():
    from concurrent.futures import ThreadPoolExecutor
    mod, neurokit = _neurokit_dfa()
    series = [_rr_series(1500, s) for s in (21, 22, 23, 24)]
    scale = np.arange(12, 150)
    with neurokit():
        want = [np.asarray(mod.fractal_dfa(x, scale=scale)[1]["Fluctuations"]) for x in series]

    def run(x):
        out = None
        for _ in range(3):
            out = np.asarray(mod.fractal_dfa(x, scale=scale)[1]["Fluctuations"])
        return out
    with ThreadPoolExecutor(4) as ex:
        got = list(ex.map(run, series * 2))
    for w, g in zip(want * 2, got):
        assert np.array_equal(w, g)


# ── Savitzky-Golay smoothing of the delineation ──────────────

def test_savgol_smooth_is_scipys_filter():
    from scipy.signal import savgol_filter
    rng = np.random.default_rng(4)
    for n in (12, 25, 60, 400):
        x = rng.standard_normal(n)
        for wl, po in ((11, 3), (7, 2), (5, 3), (21, 3)):
            if wl > n:
                continue
            want = savgol_filter(x, wl, po, mode="interp")
            got = filtering.savgol_smooth(x, wl, po)
            h = wl // 2
            # The convolution itself: the same floats; the polynomial fits
            # of the two ends: the same values to rounding.
            assert np.array_equal(want[h:n - h], got[h:n - h])
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    # What it does not take is handed to SciPy unchanged.
    x = rng.standard_normal(30)
    for args in ((x, 6, 2), (x.astype(np.float32), 7, 2), (x, 7, 2, "nearest")):
        want = savgol_filter(*args[:3], mode=args[3] if len(args) > 3 else "interp")
        assert np.array_equal(filtering.savgol_smooth(*args), want)
    with pytest.raises(ValueError):                 # as SciPy: window longer than x
        filtering.savgol_smooth(x[:5], 7, 2)


# ── Detect: per-beat quantities in batches ───────────────────

def _signals():
    rng = np.random.default_rng(9)
    n = 6000
    base = np.cumsum(rng.standard_normal(n)) * 0.05 + rng.standard_normal(n)
    with_nan = base.astype(np.float32)
    with_nan[rng.integers(0, n, 60)] = np.nan
    return {"float": base, "plateaus": np.round(base * 2) / 2,
            "adc": np.round(base * 3).astype(np.int16), "nan": with_nan,
            "flat": np.zeros(n)}


def test_batched_r_snapping_is_the_per_window_one():
    rng = np.random.default_rng(10)
    for name, x in _signals().items():
        approx = rng.integers(-15, len(x) + 15, 300).tolist()
        deriv = np.gradient(np.asarray(x, float))
        for pre, post, d in ((8, 4, deriv), (12, None, None), (0, 0, None), (1, 2, deriv)):
            want = [detection._snap_to_r(a, x, FS, pre, post, d) for a in approx]
            got = detection._snap_to_r_many(approx, x, FS, pre, post, d)
            assert got.tolist() == want, (name, pre, post)


def test_batched_slopes_and_noise_are_the_per_peak_ones():
    rng = np.random.default_rng(11)
    for name, x in _signals().items():
        peaks = np.sort(rng.choice(len(x), 200, replace=False))
        peaks[:3] = (0, 1, 2)
        want = [detection._upstroke_slope_at(int(p), x, FS) for p in peaks]
        assert np.array_equal(detection._upstroke_slopes(peaks, x, FS), want, equal_nan=True), name
        lo = rng.integers(0, len(x) - 50, 200)
        hi = lo + rng.integers(5, 50, 200)
        want = [float(np.std(x[a:b])) for a, b in zip(lo, hi)]
        got = detection._window_stds(x, lo, hi).tolist()
        assert np.array_equal(got, want, equal_nan=True), name


def test_local_reference_proms_is_the_two_pointer_median():
    rng = np.random.default_rng(12)
    cands = np.sort(rng.choice(200_000, 900, replace=False))
    cands = np.sort(np.r_[cands, cands[::9]])                 # duplicates too
    for proms in (rng.random(len(cands)), np.round(rng.random(len(cands)), 1)):
        for win_s in (0.002, 0.5, 2.0, 30.0):
            half = int(win_s / 2.0 * FS)
            want = []
            for c in cands:
                sel = proms[(cands >= c - half) & (cands <= c + half)]
                want.append(float(np.median(sel)) if len(sel) >= 4 else float(np.median(proms)))
            detection._REF_CACHE.update(cands=None)
            got = detection._local_reference_proms(cands, proms, FS, win_s)
            assert got.tolist() == want, win_s


# ── Events: rolling medians ──────────────────────────────────

def test_centred_medians_are_the_per_interval_ones():
    rng = np.random.default_rng(13)
    for n in (0, 1, 4, 5, 10, 11, 12, 500):
        for rr in (170 + 5 * rng.standard_normal(n), np.round(170 + 5 * rng.standard_normal(n))):
            for half, excl in ((2, False), (5, True), (5, False), (2, True)):
                want = []
                for i in range(n):
                    lo, hi = max(0, i - half), min(n, i + half + 1)
                    w = np.r_[rr[lo:i], rr[i + 1:hi]] if excl else rr[lo:hi]
                    want.append(float(np.median(w)) if len(w) else float(rr[i]))
                got = detection._centred_medians(rr, half, exclude_centre=excl)
                assert got.tolist() == want, (n, half, excl)
