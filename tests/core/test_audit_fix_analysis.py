"""
Audit 2026-11-13, bug #2: with permissive bounds, a beat whose PR / QRS /
QT lies between the strict and the widened limit keeps the interval, so it
must also keep the landmarks that interval is measured from.
"""
import numpy as np
import pytest

from ecg.core import analysis
from ecg.core import delineation_pool
from ecg.core.models import MouseECG
from ecg.core.wave_template import WaveTemplate

FS = 2000.0

# Landmarks (ms from R) giving PR = 90, QRS = 40, QT = 150 ms: each above
# its strict maximum (80 / 35 / 110) and below the permissive one (x2).
_DET = {"P_onset": -98.0, "P_peak": -90.0, "QRS_onset": -8.0, "Q_peak": -4.0,
        "S_peak": 6.0, "QRS_offset": 32.0, "J_peak": 36.0, "T_peak": 80.0,
        "T_offset": 142.0}


@pytest.fixture(autouse=True)
def _no_template_writes(monkeypatch):
    def _refuse(self, path=None):
        raise AssertionError("WaveTemplate.save() must not run in tests")
    monkeypatch.setattr(WaveTemplate, "save", _refuse)


def _run(monkeypatch, permissive):
    monkeypatch.setattr(delineation_pool, "detect_waves_on_beat",
                        lambda beat, bt, wt, **kw: dict(_DET))
    monkeypatch.setenv("ECG_DELIN_SERIAL", "1")
    rp = np.arange(20) * int(0.2 * FS) + int(0.3 * FS)
    sig = np.zeros(rp[-1] + int(0.3 * FS))
    wt = WaveTemplate()
    wt.confirmed = True
    return analysis.analyse_intervals(sig, rp, FS, np.diff(rp) / FS * 1000.0,
                                      wave_template=wt,
                                      permissive_bounds=permissive)


def test_fixture_intervals_sit_between_strict_and_permissive_bounds():
    assert MouseECG.PR_ABS_MAX < 90 < 2 * MouseECG.PR_ABS_MAX
    assert MouseECG.QRS_ABS_MAX < 40 < 2 * MouseECG.QRS_ABS_MAX
    assert MouseECG.QT_ABS_MAX < 150 < 2 * MouseECG.QT_ABS_MAX


def test_permissive_bounds_keep_the_landmarks_of_kept_intervals(monkeypatch):
    df = _run(monkeypatch, permissive=True)
    assert len(df) > 0
    np.testing.assert_allclose(df["PR_ms"], 90.0)
    np.testing.assert_allclose(df["QRS_ms"], 40.0)
    np.testing.assert_allclose(df["QT_ms"], 150.0)
    for col, ms in [("P_onset_ms", -98.0), ("P_peak_ms", -90.0),
                    ("Q_peak_ms", -4.0), ("S_peak_ms", 6.0),
                    ("QRS_offset_ms", 32.0), ("T_peak_ms", 80.0),
                    ("T_offset_ms", 142.0), ("QRS_onset_ms", -8.0)]:
        np.testing.assert_allclose(df[col], ms, err_msg=col)
    for col in ["P_onset_s", "P_peak_s", "Q_peak_s", "S_peak_s",
                "QRS_offset_s", "T_peak_s", "T_offset_s"]:
        assert df[col].notna().all(), col


def test_strict_bounds_still_blank_intervals_and_landmarks(monkeypatch):
    df = _run(monkeypatch, permissive=False)
    assert len(df) > 0
    assert df[["PR_ms", "QRS_ms", "QT_ms"]].isna().all().all()
    for col in ["P_onset_ms", "P_peak_ms", "P_onset_s", "P_peak_s",
                "Q_peak_ms", "S_peak_ms", "QRS_offset_ms", "QRS_offset_s",
                "T_peak_ms", "T_offset_ms", "T_peak_s", "T_offset_s",
                "QRS_onset_ms", "QRS_onset_s"]:
        assert df[col].isna().all(), col
