"""Compare segments in Qt: the default two halves, the Tk tab's checks,
and the statistics of each segment as compute_segment_stats gives them
with the analysis's RR bounds, breaks and bands (the Tk worker's call)."""
from pathlib import Path

import pytest

from ecg.core import analysis
from ecg.core.analysis import compute_segment_stats
from ecg.core.segment_plan import PAIRWISE
from ecg.io.segment_export import comparison_tsv
from ecg.qt.main_window import MainWindow
from tests.regression import golden

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    w.open_path(str(FIXTURE), check_session=False)
    w.detection.detect()
    qtbot.waitUntil(lambda: w.recording.detection.rpeaks_ok is not None, timeout=60000)
    yield w
    w.jobs.cancel_all()
    w.jobs.wait(5000)


def _same(a: dict, b: dict) -> None:
    # compare_segments adds the block summary (n_blocks, hr_block_*) to the
    # segments it compares, in Qt as in Tk.
    assert set(b) <= set(a)
    ref = golden.flatten({"seg": b})
    new = golden.flatten({"seg": {k: a[k] for k in b}})
    assert golden.compare(ref, new, rtol=0.0, atol=0.0) == []

def test_default_rows_are_the_two_halves(window):
    rows = window.compare.read_rows()
    dur = len(window.recording.signal.filtered) / window.recording.signal.fs
    assert rows == [("Segment A", 0.0, float(int(dur / 2))),
                    ("Segment B", float(int(dur / 2)), float(int(dur)))]


def test_bad_rows_are_refused(window):
    t = window.compare
    t.set_rows([("A", 10.0, 5.0)])
    t.run()
    assert "start ≥ end" in t.lbl_status.text()
    t.set_rows([("A", 0.0, 1e6)])
    t.run()
    assert "outside recording" in t.lbl_status.text()


def test_compare_gives_compute_segment_stats(qtbot, window):
    t = window.compare
    window.settings.sp_maxrr.setValue(300.0)
    t.set_rows([("Pre", 0.0, 50.0), ("Mid", 50.0, 100.0), ("Post", 100.0, 150.0)])
    t.run()
    qtbot.waitUntil(lambda: bool(t.segs), timeout=60000)
    rec, ctl = window.recording, window.figures.analysis_ctrl
    vlf, lf, hf = ctl.get_freq_bands()
    lo, hi = ctl.hrv_rr_bounds()
    assert hi == 300.0
    for seg, (lbl, a, b) in zip(t.segs, t.read_rows()):
        ref = compute_segment_stats(rec.signal.filtered, rec.detection.rpeaks_ok, rec.signal.fs,
                                    a, b, lbl, lf_band=lf, hf_band=hf, vlf_band=vlf,
                                    rr_min_ms=lo, rr_max_ms=hi,
                                    breaks=ctl.hrv_breaks(rec.detection.rpeaks_ok),
                                    block_s=t.sp_block.value())
        _same(seg, ref)
    res = rec.analysis.last_comparison
    assert res is not None and res.comps == [None, 0, 0]
    assert t.table.rowCount() > 20 and comparison_tsv(res)
    # pairwise: each against the previous one, without recomputing the segments
    segs = t.segs
    t._ref_picked(list(t._ref_opts.values()).index(PAIRWISE))
    assert t.segs is segs and rec.analysis.last_comparison.comps == [None, 0, 1]
    t.cb_view.setCurrentIndex(1)                  # Poincaré draws
    window.detection.apply_threshold()            # new peaks: results dropped
    assert not t.segs and rec.analysis.last_comparison is None
