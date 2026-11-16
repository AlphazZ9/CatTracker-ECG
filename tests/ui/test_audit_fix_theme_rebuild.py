"""Audit 2026-11-13 #10 / #19: after a UI rebuild (theme toggle) every toggle
button and switch shows the live state it mirrors, not its built-in default.

Pure tests: restore_ui_state runs on a fake app whose widgets were "rebuilt"
in their default OFF look while the application state survived.
"""
from __future__ import annotations

from types import SimpleNamespace

from ecg.ui.session_controller import SessionController


class _Switch:
    def __init__(self, on: bool = False):
        self.on = on

    def get(self):
        return int(self.on)

    def select(self):
        self.on = True

    def deselect(self):
        self.on = False


class _Button:
    def __init__(self, text: str):
        self.text = text
        self.style: dict = {}

    def configure(self, text=None, **kw):
        if text is not None:
            self.text = text
        self.style.update(kw)


class _Label:
    def __init__(self):
        self.packed = False

    def pack(self, **_kw):
        self.packed = True

    def pack_forget(self):
        self.packed = False


class _App(SimpleNamespace):
    """Any method restore_ui_state calls that the test doesn't care about is
    a no-op; any widget it doesn't care about is missing (None)."""

    def __getattr__(self, name):
        if name.startswith("_"):
            return lambda *a, **k: None
        raise AttributeError(name)


def _rebuilt_app(*, edit_mode=False, free=False, verified=False, preview=False):
    toggles = []
    app = _App(
        detection=SimpleNamespace(edit_mode=edit_mode, edit_free_placement=free),
        session=SimpleNamespace(verified_for_training=verified),
        ui=SimpleNamespace(filter_preview_on=preview, nav_pos=0.0,
                           rrhr_strip_visible=True, show_spike_markers=True),
        # Freshly rebuilt widgets: all in their default OFF look.
        btn_edit_mode=_Button("Edit Peaks"), lbl_edit_hint=_Label(),
        btn_free_placement=_Button("Free Placement"),
        sw_verified_training=_Switch(False), sw_filter_preview=_Switch(False),
        sw_permissive=_Switch(False),
        sl_thr=None, cb_clean=None, cb_det_method=None, tabs=None,
        analysis_ctrl=SimpleNamespace(apply_autonomic_params=lambda p: None),
    )
    app._toggle_edit_mode = lambda: toggles.append(1)
    return app, toggles


def test_edit_mode_on_is_shown_after_rebuild():
    app, toggles = _rebuilt_app(edit_mode=True, free=True)
    SessionController(app).restore_ui_state({"edit_mode": True})
    assert toggles == []                     # the mode itself is unchanged
    assert app.btn_edit_mode.text == "Edit Mode ON"
    assert app.lbl_edit_hint.packed
    assert app.btn_free_placement.text == "Free Placement ON"


def test_edit_mode_off_stays_off_after_rebuild():
    app, _ = _rebuilt_app(edit_mode=False)
    app.lbl_edit_hint.packed = True
    SessionController(app).restore_ui_state({"edit_mode": False})
    assert app.btn_edit_mode.text == "Edit Peaks"
    assert not app.lbl_edit_hint.packed
    assert app.btn_free_placement.text == "Free Placement"


def test_verified_and_filter_preview_follow_the_state():
    app, _ = _rebuilt_app(verified=True, preview=True)
    SessionController(app).restore_ui_state({})
    assert app.sw_verified_training.on
    assert app.sw_filter_preview.on
    # Syncing the widget must not change the state it mirrors.
    assert app.session.verified_for_training is True
    assert app.ui.filter_preview_on is True


def test_permissive_bounds_survive_a_rebuild():
    src, _ = _rebuilt_app()
    src.sw_permissive.select()
    snap = SessionController(src).snapshot_ui_state()
    assert snap["sw_permissive"] is True
    dst, _ = _rebuilt_app()
    SessionController(dst).restore_ui_state(snap)
    assert dst.sw_permissive.on
