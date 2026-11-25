"""The Qt wave-template editor: the Tk editor's drag rules and provenance,
and Apply writing to this recording's template only."""
from types import SimpleNamespace

from PySide6 import QtWidgets

from ecg.core.wave_template import WaveTemplate, _mouse_demo_beat
from ecg.qt.dialogs.wave_editor import WaveTemplateEditor


def test_drag_calibrate_and_apply(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: 0))
    wt = WaveTemplate()
    bt, mb = _mouse_demo_beat()
    dlg = WaveTemplateEditor(None, wt, mean_beat=mb, beat_time=bt)
    qtbot.addWidget(dlg)
    c, hw = dlg.lm["T_peak"]
    # press on the T centre, drag it 3 ms later
    dlg._press(SimpleNamespace(inaxes=True, xdata=c, button=1))
    dlg._motion(SimpleNamespace(xdata=c + 3.0))
    dlg._release()
    assert dlg.lm["T_peak"] == [round(c + 3.0, 1), hw]
    assert dlg.src == "edited"
    # auto-calibrate: WaveTemplate.calibrate_from_beat
    ref = WaveTemplate()
    ref.calibrate_from_beat(bt, mb)
    dlg.auto_calibrate()
    assert {k: tuple(v) for k, v in dlg.lm.items()} == \
        {k: tuple(v) for k, v in ref.landmarks.items()}
    assert dlg.src == "auto-calibrated"
    path = tmp_path / "t.json"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    dlg.save_file()
    assert WaveTemplate.load(str(path)).landmarks == dlg.pending().landmarks
    dlg.apply()
    assert dlg.saved and wt.confirmed and wt.source == "auto-calibrated"
    assert wt.landmarks == dlg.pending().landmarks
