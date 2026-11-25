# -*- coding: utf-8 -*-
"""The analysis window entries stand where the strip states the window: they
must say the applied window after it is reset (new file, Detect Peaks), and
the strip says when typed entries are not applied. The quality panel's ML
DETECTOR section is there whatever the Method, collapsed to one row by default
so the correlation plot keeps a readable height on a 768-px screen."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ecg
from ecg.app.theme import THEME
from ecg.app.context_fields import unapplied_window_field

UI_DIR = Path(ecg.__file__).parent / "app"   # the controllers moved there


class _Entry:
    def __init__(self, text: str = "") -> None:
        self.text = text

    def get(self) -> str:
        return self.text

    def delete(self, *_a) -> None:
        self.text = ""

    def insert(self, _i, val: str) -> None:
        self.text = val + self.text




def test_every_reset_of_the_window_also_resets_the_entries():
    # Detect Peaks, a raw load and a new file set the window back to the full
    # signal: each must also empty the entries (the reported stale strip).
    src = (UI_DIR / "signal_controller.py").read_text(encoding="utf-8")
    for fn in ast.walk(ast.parse(src)):
        if not isinstance(fn, ast.FunctionDef):
            continue
        resets = any(isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute)
                     and n.targets[0].attr == "t_start" and isinstance(n.value, ast.Constant)
                     and n.value.value == 0.0 for n in ast.walk(fn))
        if not resets:
            continue
        attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        assert "_show_window_in_entries" in attrs or "ent_analysis_t0" in attrs, fn.name


def test_typed_window_not_applied_is_said():
    assert unapplied_window_field(("", ""), (0.0, 0.0)) is None
    assert unapplied_window_field(("5.0", " 100 "), (5.0, 100.0)) is None
    f = unapplied_window_field(("5", "100"), (0.0, 0.0))
    assert f is not None and "Apply" in f.text and f.key == "window_unapplied"
    assert unapplied_window_field(("abc", ""), (0.0, 0.0)) is not None
    assert unapplied_window_field(("", ""), None) is None




def _scaled(height: int) -> int:
    """The panel heights here are those of a 1366x768 screen at font_scale
    1.0: the panel's fixed rows (score, cells, buttons) grow with the font,
    and so does the window the user needs at a larger font_scale."""
    return int(round(height * THEME.font_scale))
















