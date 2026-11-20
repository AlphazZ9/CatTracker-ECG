# -*- coding: utf-8 -*-
"""Opening a file fills Subject ID and Sex from its name while they hold what
the app put there; a typed value is never replaced; a session saves and
restores the sex like the subject."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import ecg
from ecg.ui.session_controller import SessionController
from ecg.ui.signal_controller import SignalController

UI_DIR = Path(ecg.__file__).parent / "ui"


class _Entry:
    def __init__(self, text: str = "") -> None:
        self.text = text

    def get(self) -> str:
        return self.text

    def delete(self, *_a) -> None:
        self.text = ""

    def insert(self, _i, val: str) -> None:
        self.text = val + self.text


def _app(subject: str = "subject_01", sex: str = ""):
    shown = []
    app = SimpleNamespace(ent_subject=_Entry(subject), ent_sex=_Entry(sex),
                          _subject_autofill=None, _sex_autofill=None,
                          _update_topbar_filename=lambda: shown.append(app.ent_subject.text))
    return app, shown


def _open(app, name: str) -> "tuple[str, str]":
    SignalController(app).autofill_subject_from_name(rf"C:\data\{name}")
    return app.ent_subject.text, app.ent_sex.text


def test_the_placeholder_is_filled_from_the_name():
    app, shown = _app()
    assert _open(app, "Stim BNST - 1146 F.smr") == ("1146", "F")
    assert shown == ["1146"]                         # the file chip shows it
    app, _ = _app(subject="", sex="")
    assert _open(app, "Baseline_STIM_BNST_206.mat") == ("206", "")


def test_an_auto_filled_value_follows_the_next_file():
    app, _ = _app()
    assert _open(app, "Stim BNST - 1146 F.smr") == ("1146", "F")
    assert _open(app, "Baseline_STIM_BNST_206.mat") == ("206", "")
    assert _open(app, "M_300.mat") == ("300", "M")
    # A name without a number: what came from the previous name goes.
    assert _open(app, "baseline.mat") == ("subject_01", "")
    assert _open(app, "F1146.smr") == ("1146", "F")


def test_a_typed_value_is_never_replaced():
    app, _ = _app(subject="mouse_A", sex="M")
    assert _open(app, "Stim BNST - 1146 F.smr") == ("mouse_A", "M")
    # Typed after an auto-fill: kept for the next file too.
    app, _ = _app()
    _open(app, "Stim BNST - 1146 F.smr")
    app.ent_subject.text, app.ent_sex.text = "1146b", "M"
    assert _open(app, "Baseline_STIM_BNST_206.mat") == ("1146b", "M")
    # Each field on its own: a typed subject, an empty sex filled.
    app, _ = _app(subject="A7", sex="")
    assert _open(app, "Stim 1146 F.mat") == ("A7", "F")


def test_a_saved_session_restores_its_own_but_not_a_placeholder():
    app, _ = _app()
    _open(app, "Stim BNST - 1146 F.smr")
    ctl = SessionController(app)
    # Saved before the Sex field, with the placeholder: the name's values stay.
    ctl.apply_recording_fields({"subject": "subject_01"})
    assert (app.ent_subject.text, app.ent_sex.text) == ("1146", "F")
    ctl.apply_recording_fields({"subject": "subject_01", "sex": ""})
    assert (app.ent_subject.text, app.ent_sex.text) == ("1146", "F")
    # Saved with values: they win, and the next file may replace them.
    ctl.apply_recording_fields({"subject": "1146-L", "sex": "M"})
    assert (app.ent_subject.text, app.ent_sex.text) == ("1146-L", "M")
    assert _open(app, "Baseline_STIM_BNST_206.mat") == ("206", "")


def test_session_round_trip_of_the_sex():
    src, _ = _app(subject="1146", sex="F")
    src.ent_sg_window_ms = _Entry("20")
    fields = SessionController(src).recording_fields()
    assert fields["sex"] == "F" and fields["subject"] == "1146"
    dst, _ = _app()
    dst.ent_sg_window_ms = _Entry("15")
    SessionController(dst).apply_recording_fields(fields)
    assert (dst.ent_subject.text, dst.ent_sex.text) == ("1146", "F")
    assert "ent_sex" in SessionController._SNAPSHOT_ENTRIES


def test_load_path_fills_before_the_session_restore():
    src = (UI_DIR / "signal_controller.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "load_path")
    calls = [n.func.attr for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert calls.index("autofill_subject_from_name") < calls.index("_try_restore_session")


def test_add_to_group_table_writes_the_sex():
    src = (UI_DIR / "compare_segments.py").read_text(encoding="utf-8")
    call = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "group_row")
    assert "sex" in {k.arg for k in call.keywords}


def test_a_kept_typed_value_the_name_contradicts_is_reported():
    app, _ = _app(subject="Typed7", sex="M")
    shown = []
    app._set_status = lambda text, color=None: shown.append(text)
    assert _open(app, "F_400.mat") == ("Typed7", "M")
    assert shown and "Typed7" in shown[-1] and "400" in shown[-1]
    assert "sex M" in shown[-1] and "says F" in shown[-1]
    # Taken once, as a suffix of the loading status line.
    ctl = SignalController(app)
    assert "Typed7" in ctl._take_subject_kept_note()
    assert ctl._take_subject_kept_note() == ""
    # Nothing to say when the name agrees, or gives nothing.
    shown.clear()
    _open(app, "Typed7 M.mat")
    _open(app, "baseline.mat")
    assert shown == [] and app._subject_kept_note == ""


def test_an_old_session_with_the_channel_digits_yields_to_the_name():
    app, _ = _app()
    app.signal = SimpleNamespace(channel="Baseline_STIM_BNST_206_Ch6")
    _open(app, "Baseline_STIM_BNST_206.mat")
    ctl = SessionController(app)
    # 2066: what the old subject-from-channel rule saved.
    ctl.apply_recording_fields({"subject": "2066", "sex": ""})
    assert app.ent_subject.text == "206"
    # Any other saved subject still wins.
    ctl.apply_recording_fields({"subject": "206-b"})
    assert app.ent_subject.text == "206-b"
