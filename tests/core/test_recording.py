"""ecg.core.recording: the recorded signal cannot be overwritten, and the
Recording helpers state their units (s, ms, sample indices)."""
import numpy as np
import pytest

from ecg.core.recording import Recording, SignalState


def test_raw_signal_is_read_only_without_changing_the_callers_array():
    data = np.arange(10.0)
    sig = SignalState(raw=data, fs=1000)
    with pytest.raises(ValueError):
        sig.raw[0] = 1.0
    data[0] = 42.0                        # the caller keeps its own writable array
    assert sig.raw[0] == 42.0 and data.flags.writeable


def test_reassigning_raw_keeps_it_read_only():
    sig = SignalState()
    sig.raw = np.zeros(5)
    assert not sig.raw.flags.writeable
    sig.raw = None
    assert sig.raw is None


def test_recording_units():
    rec = Recording(signal=SignalState(raw=np.zeros(4000), fs=2000))
    rec.detection.rpeaks_ok = np.array([100, 340, 580])
    assert rec.n_samples == 4000 and rec.duration_s == 2.0
    assert np.allclose(rec.peak_times_s(), [0.05, 0.17, 0.29])        # s
    assert np.allclose(rec.rr_ms(), [120.0, 120.0])                  # ms
    assert rec.time_axis()[-1] == pytest.approx(3999 / 2000)
    assert len(Recording().rr_ms()) == 0 and Recording().duration_s == 0.0
