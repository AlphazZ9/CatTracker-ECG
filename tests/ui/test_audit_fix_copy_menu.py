# -*- coding: utf-8 -*-
"""Regression test for audit bug #11 (2026-11-13): the right-click copy menu
was re-bound (add=True) on every _set_textbox(tsv=...), so N refreshes made
one right-click open N modal menus in a row.  Pure fakes, no Tk display.
"""
from __future__ import annotations

from ecg.ui.app_helpers import HelpersMixin


class _Inner:
    def __init__(self):
        self.binds = []

    def bind(self, seq, func, add=None):
        self.binds.append((seq, func, add))


class _Textbox:
    def __init__(self):
        self._textbox = _Inner()
        self.text = ""

    def configure(self, **kw):
        pass

    def delete(self, a, b):
        self.text = ""

    def insert(self, idx, text):
        self.text = text

    def get(self, a, b):
        return self.text


class _App(HelpersMixin):
    def __init__(self):
        self._tsv_store = {}
        self.clip = None
        self.status = None

    def clipboard_clear(self):
        self.clip = ""

    def clipboard_append(self, s):
        self.clip += s

    def _set_status(self, text, color=None):
        self.status = text


def test_copy_menu_bound_once_across_refreshes():
    app, tb = _App(), _Textbox()
    for i in range(3):
        app._set_textbox(tb, f"text {i}", tsv=f"a\tb\n{i}\t{i}")
    right_clicks = [b for b in tb._textbox.binds if b[0] == "<Button-3>"]
    assert len(right_clicks) == 1


def test_copy_tsv_uses_latest_tsv_after_rebinds():
    app, tb = _App(), _Textbox()
    app._set_textbox(tb, "old", tsv="old\tTSV")
    app._set_textbox(tb, "new", tsv="new\tTSV")
    app._copy_tsv(tb)
    assert app.clip == "new\tTSV"


def test_copy_tsv_falls_back_to_plain_when_tsv_cleared():
    app, tb = _App(), _Textbox()
    app._set_textbox(tb, "first", tsv="x\ty")
    app._set_textbox(tb, "plain only")
    app._copy_tsv(tb)
    assert app.clip == "plain only"
    # Re-adding a TSV later still does not stack a second binding.
    app._set_textbox(tb, "again", tsv="z\tw")
    assert len(tb._textbox.binds) == 1
