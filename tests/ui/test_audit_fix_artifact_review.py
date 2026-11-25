"""
Audit bug #13: a peak edit or threshold nudge re-runs run_detection() from
the full candidate set, which used to bring back every beat an Artifact
Review (or auto-correction) removed. The removals still in effect must be
re-applied; beats that are already back (fresh detection, hand-added) must
not be dropped again.
"""
from types import SimpleNamespace

import numpy as np

from ecg.core import peak_editing as dc_module  # where run_detection applies the threshold
from ecg.app.analysis_controller import AnalysisController
from ecg.app.detection_controller import DetectionController
from ecg.app.state import AnalysisState, DetectionState, SessionState, SignalState, UIState

FS = 2000.0


def _app(candidates, rpeaks_ok, report):
    app = SimpleNamespace(
        analysis=AnalysisState(artifact_report=report),
        detection=DetectionState(rpeaks_ok=rpeaks_ok),
        signal=SignalState(fs=FS),
        ui=UIState(), session=SessionState(),
        after=lambda ms, fn=None: None,
        lbl_npeaks=None, lbl_thr_hint=None, btn_check_agreement=None,
        plot_ctrl=SimpleNamespace(refresh_live_rr=lambda: None),
    )
    app.signal.filtered = np.zeros(int(candidates[-1]) + 10)
    app.detection.all_candidates = candidates
    app.detection.all_prominences = np.ones(len(candidates))
    app.analysis_ctrl = AnalysisController(app)
    ctrl = DetectionController(app)
    ctrl.update_signal_quality = lambda accepted: None
    return app, ctrl


def _setup(monkeypatch):
    cands = np.arange(1, 41) * 240            # 40 beats, 120 ms apart
    removed = [int(cands[10]), int(cands[25])]
    reviewed = cands[~np.isin(cands, removed)]
    report = {"removed_samples": removed, "removed_types": ["ectopic", "duplicate"],
              "break_samples": [removed[0]]}
    monkeypatch.setattr(dc_module, "apply_threshold",
                        lambda *a, **k: (cands.copy(), np.array([], int), 0.5))
    app, ctrl = _app(cands, reviewed.copy(), report)
    return app, ctrl, cands, removed


def test_manual_exclusion_keeps_review_removals(monkeypatch):
    app, ctrl, cands, removed = _setup(monkeypatch)
    app.detection.manual_excluded.add(int(cands[3]))
    n = ctrl.run_detection(0.5)
    rp = app.detection.rpeaks_ok
    assert not np.isin(removed, rp).any()
    assert int(cands[3]) not in rp
    assert n == len(cands) - 3


def test_threshold_nudge_keeps_review_removals(monkeypatch):
    app, ctrl, cands, removed = _setup(monkeypatch)
    ctrl.run_detection(0.4)
    ctrl.run_detection(0.6)
    assert not np.isin(removed, app.detection.rpeaks_ok).any()
    assert len(app.detection.rpeaks_ok) == len(cands) - 2


def test_hand_added_beat_is_not_dropped(monkeypatch):
    app, ctrl, cands, removed = _setup(monkeypatch)
    app.detection.manual_added.add(removed[0])
    ctrl.run_detection(0.5)
    rp = app.detection.rpeaks_ok
    assert removed[0] in rp and removed[1] not in rp
    # and once back it stays back (no longer an active removal)
    app.detection.manual_added.clear()
    ctrl.run_detection(0.5)
    assert removed[0] in app.detection.rpeaks_ok


def test_beat_already_back_is_not_dropped(monkeypatch):
    # e.g. a fresh Detect Peaks brought every candidate back: the old
    # report's removals are no longer in effect and must not be re-applied
    app, ctrl, cands, removed = _setup(monkeypatch)
    app.detection.rpeaks_ok = cands.copy()
    ctrl.run_detection(0.5)
    assert len(app.detection.rpeaks_ok) == len(cands)
