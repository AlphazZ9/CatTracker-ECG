"""Qt preferences and small commands: light / dark and the text size kept
between runs (the Tk theme file read once), Reset to mouse ECG defaults,
Clear session cache and the keyboard shortcuts."""
import json
from pathlib import Path

from PySide6 import QtWidgets

from ecg.core.models import FilterParams
from ecg.qt import theme
from ecg.qt.main_window import SHORTCUTS_TEXT, MainWindow

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"


def test_prefs_round_trip_and_the_tk_theme_file(qtbot):
    assert theme.load_prefs() == {"theme": None, "font_scale": 1.0}
    theme.LEGACY_THEME_PATH.parent.mkdir(parents=True, exist_ok=True)
    theme.LEGACY_THEME_PATH.write_text(json.dumps({"is_dark": True, "font_scale": 1.2}))
    assert theme.load_prefs() == {"theme": "Dark", "font_scale": 1.2}
    w = MainWindow()
    qtbot.addWidget(w)
    app = QtWidgets.QApplication.instance()
    theme.apply(app, "Light", 1.0)
    w.set_text_size(1.3)
    assert theme.font_scale() == 1.3 and "13.0pt" in app.styleSheet()
    w.toggle_theme()
    assert theme.current().is_dark
    assert theme.load_prefs() == {"theme": "Dark", "font_scale": 1.3}   # ours wins now
    w.set_text_size(9)                                                  # kept in range
    assert theme.font_scale() == theme.FONT_SCALE_RANGE[1]
    theme.apply(app, "Light", 1.0)


def test_reset_to_defaults(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    st = w.settings
    st.ed_channel.setText("Ch2")
    st.sp_hp.setValue(300.0)
    st.sp_maxrr.setValue(250.0)
    st.sp_sg_win.setValue(35.0)
    st.reset_defaults()
    fp = FilterParams()
    assert st.ed_channel.text() == fp.channel and st.sp_hp.value() == int(fp.hp)
    assert st.sp_maxrr.value() == fp.max_rr_ms and st.sp_sg_win.value() == 20.0
    p = w.detection.params()
    assert p["lp"] == fp.lp and p["max_rr_ms"] == fp.max_rr_ms


def test_clear_session_cache_and_shortcuts(qtbot, monkeypatch):
    from ecg.io.session import save_session, session_exists
    msgs = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: msgs.append(a[2])))
    w = MainWindow()
    qtbot.addWidget(w)
    w.open_path(str(FIXTURE), check_session=False)
    save_session(str(FIXTURE), {"annotations": []})
    assert session_exists(str(FIXTURE))
    w.clear_session_cache()
    assert not session_exists(str(FIXTURE))
    assert w.statusBar().currentMessage() == "Session cache deleted."
    w.clear_session_cache()
    assert w.statusBar().currentMessage() == "No session cache to delete."
    w.show_shortcuts()
    assert msgs == [SHORTCUTS_TEXT] and "Ctrl+M" in SHORTCUTS_TEXT
