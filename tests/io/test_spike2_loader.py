"""Native Spike2 (.smr) files: recognised by extension or header, the ECG
channel chosen by name or by how much it looks like a mouse ECG, and
resampled to a round integer rate by an exact ratio (the app works in
integer Hz; truncating 1028.8 Hz to 1028 would shift times by 2.8 s/h)."""
from __future__ import annotations

import numpy as np
import pytest

from ecg.io import loaders


def test_integer_rate_keeps_whole_rates_and_rounds_spike2_ones():
    assert loaders.integer_rate(1000.0) == (1000, 1, 1)
    # 972 µs and 108 µs intervals: exact small ratios.
    assert loaders.integer_rate(1 / 972e-6) == (1000, 243, 250)
    fs_out, up, down = loaders.integer_rate(1 / 108e-6)
    assert fs_out == 9200 and abs(1 / 108e-6 * up / down - 9200) < 1e-6


def test_spike2_files_are_recognised(tmp_path):
    ced = tmp_path / "rec.dat"
    ced.write_bytes(b"\x05\x00(C) CED 87" + bytes(100))
    mat = tmp_path / "rec.mat"
    mat.write_bytes(b"MATLAB 5.0 MAT-file" + bytes(100))
    assert loaders.is_spike2_file(str(ced))
    assert loaders.is_spike2_file(str(tmp_path / "x.smr"))     # by extension
    assert not loaders.is_spike2_file(str(mat))


def test_smrx_gets_a_clear_message(tmp_path):
    f = tmp_path / "rec.smrx"
    f.write_bytes(bytes(64))
    with pytest.raises(ValueError, match="32-bit .smr"):
        loaders.load_mat_signal(str(f), "ECG")


def _ecg(fs: float, seconds: float, rr_s: float = 0.135) -> np.ndarray:
    n = int(fs * seconds)
    x = 0.02 * np.random.default_rng(0).standard_normal(n)
    k = np.arange(-int(0.01 * fs), int(0.01 * fs) + 1)
    pulse = np.exp(-0.5 * (k / (0.002 * fs)) ** 2)
    for b in (np.arange(0.1, seconds - 0.1, rr_s) * fs).astype(int):
        x[b + k] += pulse
    return x


class _FakeReader:
    """What load_spike2_signal reads from neo's Spike2RawIO."""

    def __init__(self, chans: "list[tuple[str, float, np.ndarray]]") -> None:
        self.data = chans
        self.header = {
            "signal_streams": np.array([(f"s{i}", str(i), "") for i in range(len(chans))],
                                       dtype=[("name", "U8"), ("id", "U8"), ("x", "U2")]),
            "signal_channels": np.array(
                [(n, str(i), fs, str(i)) for i, (n, fs, _x) in enumerate(chans)],
                dtype=[("name", "U16"), ("id", "U8"), ("sampling_rate", "f8"),
                       ("stream_id", "U8")]),
            "event_channels": np.array([("STIM", "1")], dtype=[("name", "U8"), ("id", "U8")]),
        }

    def get_signal_size(self, _b, _s, si):
        return len(self.data[si][2])

    def get_analogsignal_chunk(self, _b, _s, i0, i1, stream_index, channel_indexes):
        return self.data[stream_index][2][i0:i1][:, None]

    def rescale_signal_raw_to_float(self, raw, dtype, stream_index, channel_indexes):
        return raw.astype(dtype)


@pytest.fixture
def fake_smr(monkeypatch, tmp_path):
    fs_ecg = 1 / 972e-6
    reader = _FakeReader([
        ("FIELDrec", 2000.0, 0.01 * np.random.default_rng(1).standard_normal(2000 * 60)),
        ("Recording", fs_ecg, _ecg(fs_ecg, 60)),
    ])
    monkeypatch.setattr(loaders, "_spike2_reader", lambda _p: reader)
    path = tmp_path / "rec.smr"
    path.write_bytes(b"\x05\x00(C) CED 87")
    return str(path), fs_ecg


def test_the_ecg_like_channel_is_chosen_and_resampled(fake_smr):
    path, fs_ecg = fake_smr
    sig, ch, keys, fs = loaders.load_mat_signal(path, "ECG")
    assert ch == "Recording" and keys == ["FIELDrec", "Recording"]
    assert fs == 1000.0
    # Same duration at the new rate: nothing lost or stretched.
    assert len(sig) / fs == pytest.approx(60, abs=0.01)
    assert sig.dtype == np.float64


def test_a_named_channel_wins(fake_smr):
    path, _ = fake_smr
    _sig, ch, _keys, fs = loaders.load_mat_signal(path, "fieldrec")
    assert ch == "FIELDrec" and fs == 2000.0


def test_listing_shows_channels_and_markers(fake_smr):
    path, _ = fake_smr
    text = loaders.list_channels(path)
    assert "Spike2" in text and "Recording" in text and "STIM" in text
