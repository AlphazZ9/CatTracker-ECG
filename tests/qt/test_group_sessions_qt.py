"""Group in Qt: From saved sessions (a table held in memory, the Tk
project_sheets) and the Prism exports (new file, template filled)."""
import numpy as np
from PySide6 import QtWidgets

from ecg.io import db
from ecg.io import group_compare as gc
from ecg.io import group_data as gd
from ecg.io import prism_template as pt
from ecg.qt.main_window import MainWindow
from ecg.app import group_helpers as tg


def _saved(tmp_path, name, subject, sex, project, analysed=True):
    from ecg.io.loaders import _serialise_results
    from ecg.io.session import _file_fingerprint, save_session
    fp = tmp_path / name
    fp.write_bytes(b"ecg")
    rng = np.random.default_rng(len(name))
    rr = 120 + rng.normal(0, 2, 2000)
    results = {"rr_ms": rr, "rr_t_s": np.cumsum(rr) / 1000,
               "rr_valid": np.ones(len(rr), bool), "hr": {}}
    state = {"recording_fields": {"subject": subject, "sex": sex, "project_name": project},
             "annotations": [{"t_start": 100.0, "t_end": 101.0, "label": "Stim"}]}
    if analysed:
        state["results"] = _serialise_results(results)
    sp = save_session(str(fp), state)
    db.upsert_recording(str(fp), _file_fingerprint(str(fp)), str(sp), project_name=project)
    return str(fp)


def test_group_from_saved_sessions_then_prism(qtbot, tmp_path, monkeypatch):
    _saved(tmp_path, "a 1146.smr", "1146", "F", "BNST")
    _saved(tmp_path, "b 206.smr", "206", "m", "BNST")
    _saved(tmp_path, "c 300.smr", "300", "F", "BNST", analysed=False)
    monkeypatch.setattr(pt, "PREFS_PATH", tmp_path / "prism.json")
    w = MainWindow()
    qtbot.addWidget(w)
    g = w.group
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem",
                        staticmethod(lambda *a, **k: ("BNST", True)))
    g.from_sessions()
    assert g.path == "" and g.name == "BNST"
    assert not list(tmp_path.glob("*.xlsx"))                 # nothing written
    ref, added, skipped = tg.project_sheets("BNST")
    assert sorted(added) == ["1146", "206"] and len(skipped) == 1
    for name, df in ref.items():
        assert g.sheets[name].equals(df)
    assert "BNST (not saved)" in g.lbl_file.text() and "2 animals" in g.lbl_file.text()
    assert "left out: c 300.smr" in g.lbl_msg.text()
    g.cb_by.setCurrentText(gc.BY_SEX)
    g.cb_an.setCurrentText(gc.AN_WHOLE)
    assert g.res is not None and g.btn_prism.isEnabled()
    new = tmp_path / "group.pzfx"
    g.prism_new(str(new))
    assert new.exists() and "Prism:" in g.lbl_out.text()
    filled = tmp_path / "filled.pzfx"
    g.prism_fill(template=str(new), out=str(filled))
    assert filled.exists() and "Prism template group.pzfx" in g.lbl_out.text()
    # Save as Excel… keeps the in-memory group
    out = tmp_path / "BNST.xlsx"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "")))
    g.save_as()
    assert g.path == str(out) and len(gd.animals_frame(gd.read_sheets(str(out)))) == 2


def test_no_project_yet_says_how(qtbot, monkeypatch):
    seen = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: seen.append(a[2])))
    w = MainWindow()
    qtbot.addWidget(w)
    w.group.from_sessions()
    assert seen and "No saved session has a project name yet" in seen[0]
