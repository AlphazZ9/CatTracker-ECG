"""ecg.core.windowed_hrv called directly (no UI): the epoch grid, the
skipping floors, the optional Epochs columns and the progress callback. The
numbers themselves are held by tests/regression and by
tests/ui/test_epoch_sliding_characterisation.py (through the controller)."""
import numpy as np
import pytest

from ecg.core import analysis
from ecg.core.models import MouseECG
from ecg.core.windowed_hrv import (EPOCH_OPTIONAL, epoch_grid, epoch_hrv, sliding_hrv)
from tests.helpers import rpeaks_from_rr

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")

FS = 2000.0
RR = (MouseECG.RR_MIN_MS, MouseECG.RR_MAX_MS)


def _peaks(seconds: float = 120.0) -> np.ndarray:
    rng = np.random.default_rng(3)
    n = int(seconds / 0.12)
    return rpeaks_from_rr(list(120 + rng.normal(0, 3, n)), FS)


def test_epoch_grid_spans_first_to_last_peak():
    rp = _peaks()
    step, starts, dur = epoch_grid(rp, FS, 30.0, 10.0)
    assert step == 20.0
    assert starts[0] == rp[0] / FS
    assert np.allclose(np.diff(starts), 20.0)
    assert starts[-1] + 30.0 <= rp[0] / FS + dur + step * 0.5
    assert epoch_grid(rp, FS, 30.0, 29.5)[0] == 1.0          # step floored at 1 s


def test_epochs_keep_optional_columns_only_when_ticked():
    rp = _peaks()
    _, starts, _ = epoch_grid(rp, FS, 30.0, 0.0)
    base = epoch_hrv(rp, FS, starts, 30.0, *RR, None, None, False, False, set())
    assert not set(EPOCH_OPTIONAL) & set(base["rows"][0])
    sd = epoch_hrv(rp, FS, starts, 30.0, *RR, None, None, True, False, {"SD1", "SD2"})
    assert {"SD1", "SD2"} <= set(sd["rows"][0]) and "LF_nu" not in sd["rows"][0]
    for a, b in zip(base["rows"], sd["rows"]):
        assert a["RMSSD"] == b["RMSSD"] and a["HR_mean"] == b["HR_mean"]


def test_windows_below_the_floors_are_skipped_not_filled():
    rp = _peaks(20.0)
    out = sliding_hrv(rp, FS, 3.0, 1.0, *RR, None, None, False, False, min_beats=10_000)
    assert out["rows"] == [] and out["n_skipped"] > 0


def test_progress_is_reported_and_optional():
    rp = _peaks()
    seen = []
    sliding_hrv(rp, FS, 30.0, 5.0, *RR, None, None, False, False,
                progress_cb=lambda p, m: seen.append((p, m)))
    assert seen and seen[0][1].startswith("Window 1/") and all(0 < p <= 100 for p, _ in seen)
    assert sliding_hrv(rp, FS, 30.0, 5.0, *RR, None, None, False, False)["rows"]
