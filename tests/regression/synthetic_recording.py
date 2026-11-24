"""
tests.regression.synthetic_recording
------------------------------------
A deterministic synthetic mouse ECG, saved as a Spike2-style MATLAB file, for
the reference ("golden master") outputs of tests/regression.

It is not a physiological model. It holds, on purpose, the situations the
pipeline treats specially, so that the reference outputs cover them:

* a mouse P-QRS-T beat (P, Q, R, S, J/T and a late negative T phase) at about
  520 bpm, with LF (0.8 Hz) and respiratory HF (2.5 Hz) modulation of the RR;
* isolated premature beats, a premature couplet and a pause (missed beat);
* baseline wander, white noise, 50 Hz mains and a 1.5 s EMG-like noise burst;
* a Spike2-style struct (``values`` + ``interval`` = 1/fs), so the sampling
  rate is read from the file like a real Spike2 export.

The file is written once and committed (tests/fixtures/). This module only
documents how it was made and lets it be rebuilt; the reference outputs are
tied to the committed file (its SHA-256 is checked), not to this code.

    python -m tests.regression.synthetic_recording tests/fixtures/synthetic_mouse_ecg.mat
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FS = 2000            # Hz
DURATION_S = 150.0   # s
SEED = 20261124

# Beat shape: (centre ms relative to R, width ms, amplitude) Gaussian bumps.
_WAVES = (
    (-35.0, 5.0, 0.08),    # P
    (-5.0, 1.5, -0.10),    # Q
    (0.0, 2.0, 1.00),      # R
    (5.0, 2.0, -0.35),     # S
    (18.0, 6.0, 0.15),     # J / T
    (40.0, 10.0, -0.05),   # late negative T phase
)

# Beat indices with a special rhythm.
PREMATURE_SINGLE = (300, 700)      # coupling 0.70 x RR, then compensatory pause
PREMATURE_COUPLET = (900,)         # two premature beats in a row
PAUSE = (1100,)                    # one beat missing (RR doubled)
NOISE_BURST_S = (95.0, 96.5)       # EMG-like burst


def _rr_series(rng: np.random.Generator, n: int) -> np.ndarray:
    """Normal-sinus RR intervals (ms) with LF and HF modulation."""
    t = np.cumsum(np.full(n, 0.115))
    rr = (115.0
          + 4.0 * np.sin(2 * np.pi * 0.8 * t)
          + 3.0 * np.sin(2 * np.pi * 2.5 * t)
          + rng.normal(0.0, 1.5, n))
    return rr


def beat_times_s(rng: np.random.Generator) -> np.ndarray:
    """R-peak times (s) including the premature beats and the pause."""
    rr = _rr_series(rng, int(DURATION_S / 0.1) + 50)
    times = []
    t = 0.2
    i = 0
    while t < DURATION_S - 0.2:
        times.append(t)
        step = rr[i] / 1000.0
        if i in PREMATURE_SINGLE:
            times.append(t + 0.70 * step)
            step *= 2.0                     # compensatory pause
        elif i in PREMATURE_COUPLET:
            times.append(t + 0.70 * step)
            times.append(t + 1.40 * step)
            step *= 2.3
        elif i in PAUSE:
            step *= 2.0
        t += step
        i += 1
    return np.array([x for x in times if x < DURATION_S - 0.2])


def make_signal() -> "tuple[np.ndarray, np.ndarray]":
    """(signal in mV as float32, true R-peak sample indices)."""
    rng = np.random.default_rng(SEED)
    n = int(DURATION_S * FS)
    t = np.arange(n) / FS
    sig = np.zeros(n)
    beats = beat_times_s(rng)
    half = int(0.08 * FS)
    k = np.arange(-half, half + 1)
    shape = np.zeros(len(k))
    for centre, width, amp in _WAVES:
        shape += amp * np.exp(-0.5 * ((k / FS * 1000.0 - centre) / width) ** 2)
    r_idx = np.round(beats * FS).astype(int)
    for c in r_idx:
        lo, hi = c - half, c + half + 1
        if lo < 0 or hi > n:
            continue
        sig[lo:hi] += shape
    sig += 0.10 * np.sin(2 * np.pi * 0.3 * t) + 0.05 * np.sin(2 * np.pi * 1.1 * t)
    sig += 0.01 * np.sin(2 * np.pi * 50.0 * t)
    sig += rng.normal(0.0, 0.02, n)
    b0, b1 = (int(x * FS) for x in NOISE_BURST_S)
    sig[b0:b1] += rng.normal(0.0, 0.30, b1 - b0)
    return sig.astype(np.float32), r_idx


def write(path: "str | Path") -> Path:
    """Write the recording as a Spike2-style v5 .mat file."""
    import scipy.io
    path = Path(path)
    sig, _ = make_signal()
    scipy.io.savemat(path, {"ECG": {"values": sig.reshape(-1, 1),
                                    "interval": 1.0 / FS,
                                    "title": "ECG", "units": "mV"}},
                     do_compression=True)
    return path


if __name__ == "__main__":
    out = write(sys.argv[1] if len(sys.argv) > 1 else "synthetic_mouse_ecg.mat")
    print(out)
