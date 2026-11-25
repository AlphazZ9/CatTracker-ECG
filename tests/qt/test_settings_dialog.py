"""The Settings dialog: its defaults are the Tk ones, and its fields reach
the Detect Peaks parameters and the HRV RR bounds through the same reader
(FilterParams.from_widgets / AnalysisController.hrv_rr_bounds)."""
import pytest

from ecg.core.models import FilterParams
from ecg.qt.main_window import MainWindow


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w.jobs.cancel_all()
    w.jobs.wait(5000)


def test_defaults_are_the_tk_defaults(window):
    expected = FilterParams(detection_method="sg_derivative").to_dict()
    params = window.detection.params()
    assert {k: params[k] for k in expected} == expected
    assert params["sg_window_ms"] == 20.0


def test_fields_reach_the_parameters(window):
    s = window.settings
    s.ed_channel.setText("ECG2")
    s.sp_t0.setValue(10.0)
    s.sp_t1.setValue(90.0)
    s.sp_lp.setValue(5.0)
    s.sp_hp.setValue(250.0)
    s.chk_notch.setChecked(True)
    s.cb_clean.setCurrentText("elgendi2010")
    s.sp_minrr.setValue(60.0)
    s.sp_maxrr.setValue(300.0)
    s.chk_invert.setChecked(True)
    s.chk_artifact.setChecked(True)
    s.sp_sg_win.setValue(15.0)
    s.sp_sg_fs.setValue(8000)
    window.detection.chk_filter.setChecked(False)
    p = window.detection.params()
    assert (p["channel"], p["t_start"], p["t_end"], p["lp"], p["hp"]) == ("ECG2", 10.0, 90.0,
                                                                         5.0, 250.0)
    assert p["notch"] and p["invert_signal"] and p["artifact_correction"]
    assert p["no_filter"] and p["clean_method"] == "elgendi2010"
    assert (p["min_rr_ms"], p["max_rr_ms"], p["sg_target_fs"], p["sg_window_ms"]) == (
        60.0, 300.0, 8000, 15.0)
    assert window.figures.analysis_ctrl.hrv_rr_bounds() == (60.0, 300.0)
