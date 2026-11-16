# -*- coding: utf-8 -*-
"""Audit fix #12: the shell tick called a method that exists nowhere, so the
Report and Compare refreshes after it never ran. Every step it calls must
exist, and one failing step must not skip the others."""
from __future__ import annotations

from types import SimpleNamespace

from ecg.ui.app import ECGApp
from ecg.ui.context_strip import _SHELL_REFRESHES, ContextStripMixin
from ecg.ui.tab_keys import TAB_COMPARE


def test_every_shell_refresh_exists_on_the_app():
    missing = [n for n in _SHELL_REFRESHES if not callable(getattr(ECGApp, n, None))]
    assert missing == []
    assert "_refresh_report" in _SHELL_REFRESHES
    assert "_refresh_compare_if_open" in _SHELL_REFRESHES


def _fake(open_tab: str, failing: "set[str]"):
    calls: "list[str]" = []

    def step(name):
        def run():
            calls.append(name)
            if name in failing:
                raise RuntimeError(f"{name} failed")
        return run

    app = SimpleNamespace(tabs=SimpleNamespace(get=lambda: open_tab))
    for name in _SHELL_REFRESHES:
        setattr(app, name, step(name))
    app._refresh_compare_tab = step("_refresh_compare_tab")
    app._refresh_compare_if_open = lambda: ContextStripMixin._refresh_compare_if_open(app)
    return app, calls


def test_a_failing_refresh_does_not_skip_the_ones_after_it():
    app, calls = _fake(TAB_COMPARE, failing={"_refresh_tab_states", "_update_topbar_filename"})
    ContextStripMixin._refresh_shell(app)       # never raises
    assert "_refresh_report" in calls
    assert "_refresh_compare_tab" in calls
    assert "_refresh_status_lines" in calls


def test_compare_is_rebuilt_only_while_its_tab_is_open():
    app, calls = _fake("Report", failing=set())
    ContextStripMixin._refresh_shell(app)
    assert "_refresh_report" in calls
    assert "_refresh_compare_tab" not in calls
