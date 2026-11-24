"""Save Session / restore in the Qt interface: the session file is the Tk
one (same writer), and restoring it gives back the peaks, the hand edits,
the noisy regions, the settings and the results exactly."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecg.core import analysis
from ecg.io.session import load_session
from ecg.qt.main_window import MainWindow

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")


@pytest.fixture
def isolated_sessions(tmp_path, monkeypatch):
    """Session files and the registry in a temporary folder."""
    import ecg.io.session as io_session
    monkeypatch.setattr(io_session, "SESSION_DIR", tmp_path / "sessions")
    import ecg.io.db as db
    monkeypatch.setattr(db, "_DB_PATH", tmp_path / "registry.db")
    monkeypatch.setattr(db, "_schema_ready", False)
    monkeypatch.setattr(db, "_schema_retry_at", 0.0)
    return tmp_path


def _window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    return w


def test_save_then_restore_gives_the_same_state(qtbot, isolated_sessions):
    w = _window(qtbot)
    w.open_path(str(FIXTURE), check_session=False)
    w.detection.cb_method.setCurrentIndex(0)
    w.settings.sp_maxrr.setValue(250.0)
    w.detection.detect()
    qtbot.waitUntil(lambda: w.recording.detection.rpeaks_ok is not None, timeout=60000)
    d, a = w.recording.detection, w.recording.analysis
    # a hand edit and a noisy region
    t_excl = d.rpeaks_ok[100] / w.recording.signal.fs
    w.detection.btn_edit.setChecked(True)
    w.detection._clicked(float(t_excl), "left")
    a.noise_regions.append({"t_start": 30.0, "t_end": 35.0, "note": "test"})
    a.annotations.append({"t_start": 50.0, "t_end": 60.0, "label": "drug", "color": "#E53935"})
    w.analyze()
    qtbot.waitUntil(lambda: w.statusBar().currentMessage() == "Analysis done", timeout=180000)
    before_peaks = d.rpeaks_ok.copy()
    before_excl = set(d.manual_excluded)
    before = {k: v.copy() for k, v in a.results.items() if isinstance(v, pd.DataFrame)}
    w.session.save()
    qtbot.waitUntil(lambda: w.session.saved_at is not None, timeout=60000)
    state = load_session(str(FIXTURE))
    assert state is not None and state["filter_params"]["max_rr_ms"] == 250.0
    assert state["noise_regions"] == [{"t_start": 30.0, "t_end": 35.0, "note": "test"}]
    w.jobs.cancel_all()
    w.jobs.wait(5000)

    w2 = _window(qtbot)
    w2.open_path(str(FIXTURE), check_session=False)
    w2.session.saved_at = None
    w2.session.restore(state, str(FIXTURE))
    qtbot.waitUntil(lambda: w2.session.saved_at is not None, timeout=60000)
    d2, a2 = w2.recording.detection, w2.recording.analysis
    np.testing.assert_array_equal(d2.rpeaks_ok, before_peaks)
    assert set(d2.manual_excluded) == before_excl
    assert a2.noise_regions == a.noise_regions and a2.annotations == a.annotations
    assert w2.settings.sp_maxrr.value() == 250.0
    assert w2.detection.cb_method.currentIndex() == 0
    assert w2.detection.btn_noise.text() == "Noisy regions (1)"
    for key, df in before.items():
        if key in a2.results:
            # The session JSON (the Tk format) does not keep a frame's index.
            pd.testing.assert_frame_equal(a2.results[key].reset_index(drop=True),
                                          df.reset_index(drop=True), check_exact=False,
                                          rtol=1e-12, check_dtype=False)
    assert w2.btn_analyze.isEnabled()
    w2.jobs.cancel_all()
    w2.jobs.wait(5000)
