"""Review artifacts in Qt: the candidates of detect_rr_artifacts with the
Tk review's parameters, and the decisions applied as in Tk."""
from pathlib import Path

import numpy as np
import pytest

from ecg.core import analysis
from ecg.core.detection import apply_artifact_decisions, detect_rr_artifacts
from ecg.qt.dialogs import artifact_review as ar
from ecg.qt.main_window import MainWindow

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")


def test_review_removes_the_chosen_beats(qtbot, monkeypatch):
    w = MainWindow()
    qtbot.addWidget(w)
    w.open_path(str(FIXTURE), check_session=False)
    w.detection.detect()
    qtbot.waitUntil(lambda: w.recording.detection.rpeaks_ok is not None, timeout=60000)
    rec = w.recording
    rp = rec.detection.rpeaks_ok.copy()
    # an extra peak 20 ms after a real one: a duplicate candidate
    extra = int(rp[200] + 0.02 * rec.signal.fs)
    rp = np.sort(np.append(rp, extra))
    rec.detection.rpeaks_ok = rp.copy()
    _lo, hi = w.figures.analysis_ctrl.hrv_rr_bounds()
    ref = detect_rr_artifacts(rp, rec.signal.fs, rr_min_ms=float(w.settings.sp_minrr.value()),
                              rr_max_ms=hi, window_beats=11, dev_threshold=0.20,
                              signal=rec.signal.filtered)
    assert ref
    seen = {}

    def fake_exec(self):
        seen["n"] = len(self.candidates)
        self._set_all("remove")
        return True
    monkeypatch.setattr(ar.ArtifactReviewDialog, "exec", fake_exec)
    w.detection.btn_review.click()
    qtbot.waitUntil(lambda: rec.analysis.artifact_candidates is not None
                    and len(rec.analysis.artifact_candidates) > 0, timeout=60000)
    assert seen["n"] == len(ref)
    expected, _report = apply_artifact_decisions(
        rp, [dict(c, decision="remove") for c in ref], fs=rec.signal.fs)
    np.testing.assert_array_equal(rec.detection.rpeaks_ok, expected)
    assert rec.analysis.artifact_report is not None
    w.jobs.cancel_all()
    w.jobs.wait(5000)
