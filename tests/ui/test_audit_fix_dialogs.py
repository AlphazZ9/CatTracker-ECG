# -*- coding: utf-8 -*-
"""Regression tests for audit bugs #14 and #15 (ecg/ui/dialogs.py).

Both methods are called unbound on light fake objects, so no Tk display is
needed and nothing is written to the user's home files.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

dialogs = pytest.importorskip("ecg.ui.dialogs")


# ── #14: 'Save as Default' must persist the selected theme ──────────────────

class _FakeTheme:
    def __init__(self):
        self.preset_name = "Light"
        self.font_scale = 1.00
        self.saved = None

    def apply_preset(self, name):
        self.preset_name = name

    def save(self):
        self.saved = (self.preset_name, self.font_scale)


def test_save_default_persists_working_theme(monkeypatch):
    fake = _FakeTheme()
    monkeypatch.setattr(dialogs, "THEME", fake)
    applied = []
    dlg = SimpleNamespace(
        _working=SimpleNamespace(preset_name="Dark", font_scale=1.20),
        _saved=False,
        _apply=lambda: applied.append(True),
    )
    dialogs.ThemeDialog._save_default(dlg)
    assert fake.saved == ("Dark", 1.20)
    assert dlg._saved is True
    assert applied == [True]


# ── #15: calibration config choice survives a reload ────────────────────────

class _FakeCombo:
    def __init__(self, value=""):
        self.value = value
        self.values = []

    def get(self):
        return self.value

    def set(self, v):
        self.value = v

    def configure(self, **kw):
        if "values" in kw:
            self.values = kw["values"]


def test_reload_keeps_chosen_config_when_counts_change(monkeypatch):
    cfg_a, cfg_b = ("A",), ("B",)
    opts = {"first": [(cfg_a, 10), (cfg_b, 6)], "second": [(cfg_a, 4), (cfg_b, 3)]}
    state = {"opts": opts["first"]}
    monkeypatch.setattr(dialogs, "list_event_rows", lambda **kw: [])
    monkeypatch.setattr(dialogs, "registry_frame", lambda rows: None)
    monkeypatch.setattr(dialogs, "config_options", lambda frame: state["opts"])
    monkeypatch.setattr(dialogs, "config_label", lambda cfg: cfg[0])
    monkeypatch.setattr(dialogs, "list_event_labels", lambda: [])

    recomputed = []
    dlg = SimpleNamespace(
        ALL="All labels",
        _cb_label=_FakeCombo("All labels"),
        _cb_config=_FakeCombo(),
        _configs={},
        _want_config=None,
        _lbl_labels=SimpleNamespace(configure=lambda **kw: None),
        _label_counts=lambda project: {},
        _recompute=lambda: recomputed.append(dlg._configs.get(dlg._cb_config.get())),
    )
    dlg._filters = lambda: dialogs.CalibrationDialog._filters(dlg)
    dlg._ent_project = SimpleNamespace(get=lambda: "")

    dialogs.CalibrationDialog._reload(dlg)
    assert dlg._configs[dlg._cb_config.get()] == cfg_a       # default: first
    # User picks B, then a filter change alters every event count.
    dlg._cb_config.set(next(n for n, c in dlg._configs.items() if c == cfg_b))
    state["opts"] = opts["second"]
    dialogs.CalibrationDialog._reload(dlg)
    assert dlg._configs[dlg._cb_config.get()] == cfg_b
    assert recomputed[-1] == cfg_b
