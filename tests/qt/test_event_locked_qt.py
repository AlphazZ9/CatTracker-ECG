"""Time course > Event-locked in Qt: the Tk EventLockedController on the
window's host; its result is compute_event_locked of the Autonomic result
around the annotations, and it is dropped when they change."""
from pathlib import Path

import pandas as pd
import pytest

from ecg.core import analysis
from ecg.qt.main_window import MainWindow
from ecg.ui.event_locked_view import (ALL_LABELS, compute_event_locked,
                                      other_annotation_onsets, select_events)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")


def test_event_locked(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    w.open_path(str(FIXTURE), check_session=False)
    w.detection.detect()
    qtbot.waitUntil(lambda: w.recording.detection.rpeaks_ok is not None, timeout=60000)
    a = w.recording.analysis
    a.annotations[:] = [{"t_start": 50.0, "t_end": 55.0, "label": "drug", "color": "#E53935"},
                        {"t_start": 100.0, "t_end": 105.0, "label": "drug", "color": "#E53935"}]
    w.detection.annotations_changed()
    tc = w.time_course
    assert [tc.cb_evt_label.itemText(i) for i in range(tc.cb_evt_label.count())] == \
        [ALL_LABELS, "drug"]
    tc.btn_auto.click()
    qtbot.waitUntil(lambda: a.autonomic_result is not None, timeout=120000)
    tc.sp_evt_pre.setValue(20.0)
    tc.sp_evt_post.setValue(20.0)
    tc.btn_evt.click()
    assert a.event_locked is not None
    events = select_events(a.annotations, ALL_LABELS)
    ref = compute_event_locked(a.autonomic_result, events, 20.0, 20.0,
                               spwvd=a.autonomic_spwvd, label_filter=ALL_LABELS,
                               other_onsets=other_annotation_onsets(a.annotations, events))
    pd.testing.assert_frame_equal(a.event_locked.table, ref.table)
    assert tc.tbl_evt.rowCount() == 2
    assert w.session.ctrl.collect_session_state()["event_locked_params"]["pre_s"] == 20.0
    # moving an event drops the result
    a.annotations[0]["t_start"] = 60.0
    w.detection.annotations_changed()
    assert a.event_locked is None and "Annotations changed" in w.figures.lbl_evt_status.text()
    w.jobs.cancel_all()
    w.jobs.wait(5000)
