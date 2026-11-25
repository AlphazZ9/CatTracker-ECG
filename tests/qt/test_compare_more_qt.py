"""Compare segments in Qt beyond the comparison: the configurations
relative to the stimulus, the Prism exports (new file, template filled),
the figures, Add to group table with the group Poincaré, and the setup kept
in sessions -- the Tk tab's helpers and files."""
from pathlib import Path

import pytest
from PySide6 import QtWidgets

from ecg.core import analysis
from ecg.core import segment_plan as sp
from ecg.io import group_data as gd
from ecg.io import prism_template as pt
from ecg.io.session import load_session
from ecg.qt.main_window import MainWindow

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")
YES = QtWidgets.QMessageBox.StandardButton.Yes


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "TEMPLATES_PATH", tmp_path / "templates.json")
    monkeypatch.setattr(pt, "PREFS_PATH", tmp_path / "prism.json")
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: YES))
    w = MainWindow()
    qtbot.addWidget(w)
    w.open_path(str(FIXTURE), check_session=False)
    w.detection.detect()
    qtbot.waitUntil(lambda: w.recording.detection.rpeaks_ok is not None, timeout=60000)
    yield w
    w.jobs.cancel_all()
    w.jobs.wait(5000)


def _compare(qtbot, t, rows):
    t.set_rows(rows)
    t.run()
    qtbot.waitUntil(lambda: bool(t.segs), timeout=60000)


def test_configurations_relative_to_the_stimulus(window):
    t = window.compare
    t.set_rows([("Pre", 40.0, 60.0), ("Post", 60.0, 80.0)])
    t.ed_stim.setText("60")
    t.cfg_save("stim 20 s")
    assert [c.name for c in sp.user_templates(sp.TEMPLATES_PATH)] == ["stim 20 s"]
    assert t.cb_cfg.currentText() == "stim 20 s" and t.btn_cfg_apply.isEnabled()
    t.ed_stim.setText("100,5")                     # a comma is read as the point
    t.cfg_apply()
    assert t.read_rows() == [("Pre", 80.5, 100.5), ("Post", 100.5, 120.5)]
    assert "applied at 100.5 s" in t.lbl_cfg.text()
    t.ed_stim.setText("140")                       # Post cut at the end (150 s)
    t.cfg_apply()
    assert t.read_rows()[1] == ("Post", 140.0, 150.0) and "cut to fit: Post" in t.lbl_cfg.text()
    t.ed_stim.setText("")
    t.cfg_apply()
    assert "Type the stim time" in t.lbl_cfg.text()
    # empty Stim at on save: the start of the Post window is offered
    t.set_rows([("Pre", 10.0, 30.0), ("Post", 30.0, 50.0)])
    t.cfg_save("second")
    assert t.ed_stim.text() == "30" and len(sp.user_templates(sp.TEMPLATES_PATH)) == 2
    t.cb_cfg.setCurrentText("second")
    t.cfg_delete()
    assert [c.name for c in sp.user_templates(sp.TEMPLATES_PATH)] == ["stim 20 s"]
    # times keep their milliseconds (not 6 significant digits)
    t.set_rows([("A", 12345.678, 12400.0)])
    assert t.rows.item(0, 1).text() == "12345.678"


def test_prism_exports_and_figures(qtbot, window, tmp_path):
    t = window.compare
    _compare(qtbot, t, [("Pre", 0.0, 75.0), ("Post", 75.0, 150.0)])
    assert t.btn_prism.isEnabled() and t.btn_figures.isEnabled()
    new = tmp_path / "new.pzfx"
    t.export_prism_new(str(new))
    assert new.exists() and "Prism:" in t.lbl_exp.text()
    filled = tmp_path / "filled.pzfx"
    t.export_prism_template(template=str(new), out=str(filled))
    assert filled.exists() and "Prism template new.pzfx" in t.lbl_exp.text()
    t.export_figures(str(tmp_path / "figs.pdf"))
    assert (tmp_path / "figs_tachograms.pdf").exists()
    assert (tmp_path / "figs_poincare.pdf").exists()


def test_add_to_group_and_the_group_poincare(qtbot, window, tmp_path):
    t = window.compare
    _compare(qtbot, t, [("Pre", 0.0, 75.0), ("Post", 75.0, 150.0)])
    table = str(tmp_path / "group.xlsx")
    t.add_to_group(table)
    assert "added to group.xlsx" in t.lbl_exp.text()
    # the synthetic recording's segments are flagged "to check": left out by default
    assert "Only 0 animals" in t.lbl_group.text() and "1 to check" in t.lbl_group.text()
    window.group.chk_leave.setChecked(False)
    t.group_refresh()
    assert "Only 1 animal " in t.lbl_group.text()            # one animal so far
    window.recording_sheet.ed_subject.setText("second")
    t.add_to_group(table)
    sheets = gd.read_sheets(table, only=("Animals", "Group"))
    assert len(sheets[gd.SHEET_SEGMENTS]) == 2
    assert t.lbl_group.text().startswith("n = 2 animals")
    assert t.group_shown is not None and t.group_shown[1:3] == ("Pre", "Post")
    assert t.btn_group_fig.isEnabled()
    t.save_group_figure(str(tmp_path / "group.png"))
    assert (tmp_path / "group.png").exists()
    t.open_group_tab()
    assert window.tabs.currentWidget() is window.group and window.group.path == table


def test_the_setup_is_kept_in_the_session(qtbot, window):
    t = window.compare
    _compare(qtbot, t, [("Pre", 10.0, 70.0), ("Post", 70.0, 130.0)])
    t.ed_stim.setText("70")
    t.cb_view.setCurrentIndex(1)
    window.session.save()
    qtbot.waitUntil(lambda: window.session.saved_at is not None, timeout=60000)
    state = load_session(str(FIXTURE))
    setup = state["compare_segments"]
    assert setup["rows"] == [["Pre", 10.0, 70.0], ["Post", 70.0, 130.0]]
    assert setup["stim"] == "70" and setup["view"] == "poincare" and setup["ran"]

    w2 = MainWindow()
    qtbot.addWidget(w2)
    w2.open_path(str(FIXTURE), check_session=False)
    w2.session.saved_at = None
    w2.session.restore(state, str(FIXTURE))
    qtbot.waitUntil(lambda: bool(w2.compare.segs), timeout=60000)
    t2 = w2.compare
    assert t2.read_rows() == [("Pre", 10.0, 70.0), ("Post", 70.0, 130.0)]
    assert t2.ed_stim.text() == "70" and t2.cb_view.currentIndex() == 1
    assert [s["n_beats"] for s in t2.segs] == [s["n_beats"] for s in t.segs]
    w2.jobs.cancel_all()
    w2.jobs.wait(5000)
