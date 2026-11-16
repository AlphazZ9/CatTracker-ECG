# -*- coding: utf-8 -*-
"""Audit #16: with Filtering OFF the main window's FILTER SETTINGS group is
disabled, and Apply & Close in the Parameters dialog used to write into the
disabled widgets -- the cutoffs, notch and cleaning method set there were
silently dropped even though the dialog also turned Filtering on."""
from __future__ import annotations

import tkinter as tk

import pytest

ctk = pytest.importorskip("customtkinter")

from ecg.ui.detection_controller import DetectionController  # noqa: E402
from ecg.ui.params_dialog import ParamsDialogMixin  # noqa: E402


def _walk(w):
    for c in w.winfo_children():
        yield c
        yield from _walk(c)


def _make_app(root, filtering_on: bool):
    """Just the main-window widgets the dialog mirrors, wired like the app."""
    app = root
    ctrl = DetectionController.__new__(DetectionController)
    holder = ctk.CTkFrame(root)
    grp = ctk.CTkFrame(holder)
    app._filter_advanced_group = grp

    def ent(parent, text):
        e = ctk.CTkEntry(parent)
        e.insert(0, text)
        return e

    app.ent_channel, app.ent_subject = ent(holder, "ECG"), ent(holder, "M1")
    app.ent_fs, app.ent_t_start, app.ent_t_end = ent(holder, "2000"), ent(holder, ""), ent(holder, "")
    app.ent_sg_target_fs, app.ent_sg_window_ms = ent(holder, "10000"), ent(holder, "20")
    app.ent_lp, app.ent_hp = ent(grp, "0.5"), ent(grp, "100")
    app.sw_notch = ctk.CTkSwitch(grp, text="Notch 50 Hz")
    app.cb_clean = ctk.CTkComboBox(grp, values=["neurokit", "elgendi2010"])
    app.cb_clean.set("neurokit")
    for name in ("sw_filtering", "sw_invert_signal", "sw_show_raw", "sw_artifact"):
        setattr(app, name, ctk.CTkSwitch(holder, text=name))
    if filtering_on:
        app.sw_filtering.select()

    def on_filtering_toggle():
        ctrl._set_group_availability(grp, bool(app.sw_filtering.get()))

    app._on_filtering_toggle = on_filtering_toggle
    app._on_show_raw_toggle = lambda: None
    app._set_status = lambda *a, **k: None
    on_filtering_toggle()
    return app


class _Root(ParamsDialogMixin, ctk.CTk):
    pass


def _open_dialog(app):
    before = set(app.winfo_children())
    app._open_params_dialog()
    (win,) = [w for w in app.winfo_children() if w not in before]
    widgets = list(_walk(win))
    sw = {w.cget("text"): w for w in widgets if isinstance(w, ctk.CTkSwitch)}
    ents = [w for w in widgets if isinstance(w, ctk.CTkEntry)
            and not isinstance(w.master, ctk.CTkComboBox)]
    combo = next(w for w in widgets if isinstance(w, ctk.CTkComboBox))
    apply_btn = next(w for w in widgets if isinstance(w, ctk.CTkButton)
                     and "Apply" in w.cget("text"))
    return win, sw, ents, combo, apply_btn


def _set(entry, val):
    entry.delete(0, "end")
    entry.insert(0, val)


@pytest.fixture
def root():
    try:
        r = _Root()
        r.withdraw()
    except tk.TclError:
        pytest.skip("no display")
    yield r
    try:
        r.destroy()
    except tk.TclError:
        pass


def test_filter_settings_survive_turning_filtering_on(root):
    app = _make_app(root, filtering_on=False)
    assert app.ent_lp.cget("state") == "disabled"
    win, sw, ents, combo, apply_btn = _open_dialog(app)
    # Entries in build order: channel, subject, fs, t_start, t_end, low, high, ...
    _set(ents[5], "5")
    _set(ents[6], "150")
    sw["Filtering"].select()
    sw["Notch 50 Hz"].select()
    combo.set("elgendi2010")
    apply_btn.invoke()

    assert bool(app.sw_filtering.get())
    assert app.ent_lp.get() == "5"
    assert app.ent_hp.get() == "150"
    assert bool(app.sw_notch.get())
    assert app.cb_clean.get() == "elgendi2010"
    # Availability follows the new Filtering state.
    assert app.ent_lp.cget("state") == "normal"


def test_values_set_in_advance_kept_and_group_regreyed_when_off(root):
    app = _make_app(root, filtering_on=False)
    win, sw, ents, combo, apply_btn = _open_dialog(app)
    _set(ents[5], "3")
    sw["Notch 50 Hz"].select()
    combo.set("elgendi2010")
    apply_btn.invoke()

    assert not bool(app.sw_filtering.get())
    assert app.ent_lp.get() == "3"
    assert bool(app.sw_notch.get())
    assert app.cb_clean.get() == "elgendi2010"
    assert app.ent_lp.cget("state") == "disabled"
    assert app.sw_notch.cget("state") == "disabled"
