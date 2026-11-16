"""Regression tests for audit bug #7 (2026-11-13): a multi-column .mat
matrix must be split into one channel per column, never flattened into an
interleaved trace."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.io

from ecg.io import loaders
from ecg.io.loaders import _flatten_mat_dict, _split_channels, load_mat_signal


def _two_channels(n: int = 20000):
    rng = np.random.default_rng(0)
    ecg = np.sin(np.linspace(0, 200 * np.pi, n)) + 0.01 * rng.standard_normal(n)
    other = 5.0 + rng.standard_normal(n)
    return ecg, other


def test_split_channels_columns_become_channels():
    ecg, other = _two_channels(500)
    out = _split_channels("data", np.column_stack([ecg, other]), time_axis_on_tie=0)
    assert set(out) == {"data[0]", "data[1]"}
    np.testing.assert_array_equal(out["data[0]"], ecg)
    np.testing.assert_array_equal(out["data[1]"], other)


def test_split_channels_rows_when_time_is_axis_1():
    ecg, other = _two_channels(500)
    out = _split_channels("data", np.vstack([ecg, other]), time_axis_on_tie=0)
    np.testing.assert_array_equal(out["data[0]"], ecg)
    np.testing.assert_array_equal(out["data[1]"], other)


@pytest.mark.parametrize("shape", [(500,), (500, 1), (1, 500)])
def test_split_channels_vectors_unchanged(shape):
    arr = np.arange(500, dtype=float).reshape(shape)
    out = _split_channels("ecg", arr, time_axis_on_tie=0)
    assert list(out) == ["ecg"]
    np.testing.assert_array_equal(out["ecg"], np.arange(500, dtype=float))


def test_flatten_mat_dict_keeps_1d_names():
    flat = _flatten_mat_dict({"__header__": b"x", "ecg": np.arange(300.0)})
    assert list(flat) == ["ecg"]


def test_load_mat_signal_v5_matrix_not_interleaved(tmp_path):
    ecg, other = _two_channels()
    path = tmp_path / "m.mat"
    scipy.io.savemat(str(path), {"data": np.column_stack([ecg, other]),
                                 "fs": 1000.0})
    sig, ch, keys, fs = load_mat_signal(str(path), "data[0]")
    assert ch == "data[0]"
    assert len(sig) == len(ecg)
    np.testing.assert_allclose(sig, ecg)
    assert "data[1]" in keys
    assert fs == 1000

    # Auto-selection (preferred name absent) still yields a real column.
    sig2, ch2, _, _ = load_mat_signal(str(path), "missing")
    assert ch2 in ("data[0]", "data[1]")
    assert len(sig2) == len(ecg)


def test_load_mat_signal_v5_vector_unchanged(tmp_path):
    ecg, _ = _two_channels()
    path = tmp_path / "v.mat"
    scipy.io.savemat(str(path), {"ecg": ecg.reshape(-1, 1)})
    sig, ch, _, _ = load_mat_signal(str(path), "ecg")
    assert ch == "ecg"
    np.testing.assert_allclose(sig, ecg)


@pytest.mark.skipif(not loaders.H5_AVAILABLE, reason="h5py not installed")
def test_hdf5_matrix_split_per_channel(tmp_path):
    import h5py
    ecg, other = _two_channels()
    path = tmp_path / "h.h5"
    with h5py.File(path, "w") as f:
        # MATLAB v7.3 stores an (n, 2) matrix transposed: (2, n) in h5py.
        f.create_dataset("data", data=np.vstack([ecg, other]))
    flat = loaders._flatten_hdf5_file(str(path))
    assert set(flat) == {"data[0]", "data[1]"}
    np.testing.assert_allclose(flat["data[0]"], ecg)
    np.testing.assert_allclose(flat["data[1]"], other)
