# -*- coding: utf-8 -*-
"""The analysis window entries stand where the strip states the window: they
must say the applied window after it is reset (new file, Detect Peaks), and
the strip says when typed entries are not applied. The quality panel's ML
DETECTOR section is there whatever the Method, collapsed to one row by default
so the correlation plot keeps a readable height on a 768-px screen."""
from __future__ import annotations

import ast
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ecg
from ecg.ui.app_helpers import HelpersMixin
from ecg.ui.context_strip import ContextStripMixin, unapplied_window_field
from ecg.ui.tab_detection import QUALITY_PANEL_W, QUALITY_PLOT_MIN_H, TabDetectionMixin
from ecg.ui.theme import THEME

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


def test_entries_are_rewritten_with_the_applied_window():
    app = SimpleNamespace(ent_analysis_t0=_Entry("5"), ent_analysis_t1=_Entry("100"),
                          analysis=SimpleNamespace(t_start=0.0, t_end=0.0))
    ContextStripMixin._show_window_in_entries(app)
    assert (app.ent_analysis_t0.text, app.ent_analysis_t1.text) == ("", "")
    app.analysis.t_start, app.analysis.t_end = 12.5, 60.0
    ContextStripMixin._show_window_in_entries(app)
    assert (app.ent_analysis_t0.text, app.ent_analysis_t1.text) == ("12.5", "60")


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


class _Panel(TabDetectionMixin, HelpersMixin):
    """Just what _build_quality_panel needs."""

    def __init__(self) -> None:
        self._slots = {}
        self.ui = SimpleNamespace(ml_section_open=False)
        self.signal = SimpleNamespace(filepath=None)
        self.detection = SimpleNamespace(rpeaks_ok=None)

    def after(self, *_a, **_k) -> None:
        pass

    def _bind_hover_tip(self, *_a, **_k) -> None:
        pass

    def _set_btn_enabled(self, btn, enabled: bool, kind: str = "primary") -> None:
        btn.configure(state="normal" if enabled else "disabled")

    def __getattr__(self, name):       # command callbacks
        if name.startswith(("_open", "_check", "_on_", "_save", "_add", "_remove", "refresh")):
            return lambda *a, **k: None
        raise AttributeError(name)


def _scaled(height: int) -> int:
    """The panel heights here are those of a 1366x768 screen at font_scale
    1.0: the panel's fixed rows (score, cells, buttons) grow with the font,
    and so does the window the user needs at a larger font_scale."""
    return int(round(height * THEME.font_scale))


def _tk_root():
    try:
        return tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")


def _host(root, height: int) -> tk.Frame:
    host = tk.Frame(root, width=QUALITY_PANEL_W, height=_scaled(height))
    host.pack()
    host.grid_propagate(False)
    host.grid_rowconfigure(0, weight=1)
    return host


def _settle(root) -> None:
    for _ in range(3):
        root.update()


def test_ml_section_keeps_the_correlation_plot_readable():
    root = _tk_root()
    try:
        # The panel's height on a 1366x768 screen (Signal & Detection tab).
        host = _host(root, 440)
        p = _Panel()
        p._build_quality_panel(host)
        _settle(root)
        head, curve = p._quality_time_rows
        # Whatever the Method, the section is there, collapsed by default to
        # its header: the curve stays readable at this height.
        assert p._ml_frame.winfo_ismapped() and p.lbl_ml_title.winfo_ismapped()
        assert "▸" in p.lbl_ml_title.cget("text")
        assert not p.btn_ml_example.winfo_ismapped()
        assert curve.winfo_ismapped()
        collapsed_h = p._slots["quality_time"].frame.winfo_height()
        assert collapsed_h >= QUALITY_PLOT_MIN_H, collapsed_h
        # Open: the training controls take the room; the curve is either
        # still readable or replaced by its title saying why, never a sliver.
        p._toggle_ml_section()
        _settle(root)
        assert "▾" in p.lbl_ml_title.cget("text") and p.ui.ml_section_open is True
        if curve.winfo_ismapped():
            assert p._slots["quality_time"].frame.winfo_height() >= QUALITY_PLOT_MIN_H
        else:
            assert "taller window" in head.cget("text")
        # Either way the curve's title stays on screen, above the ML section.
        assert head.winfo_ismapped()
        assert head.winfo_rooty() + head.winfo_height() <= p._ml_frame.winfo_rooty()
        # Every ML control is on screen, under the plot.
        bottom = host.winfo_rooty() + host.winfo_height()
        for w in (p.lbl_ml_status, p.btn_ml_example, p.btn_train_ml):
            assert w.winfo_ismapped() and w.winfo_rooty() + w.winfo_height() <= bottom
        # The one action's label fits the panel at this font size.
        assert p.btn_ml_example.winfo_reqwidth() <= p.btn_ml_example.winfo_width()
        p._toggle_ml_section()
        _settle(root)
        assert p.ui.ml_section_open is False
        assert curve.winfo_ismapped()
        assert p._slots["quality_time"].frame.winfo_height() == collapsed_h
    finally:
        root.destroy()


def test_ml_section_open_state_is_kept_by_a_rebuild():
    root = _tk_root()
    try:
        host = _host(root, 900)
        p = _Panel()
        p.ui.ml_section_open = True      # what the UI state held before the rebuild
        p._build_quality_panel(host)
        _settle(root)
        assert "▾" in p.lbl_ml_title.cget("text")
        assert p.btn_ml_example.winfo_ismapped() and p.btn_train_ml.winfo_ismapped()
        assert not p._quality_hist_rows[1].winfo_ismapped()
    finally:
        root.destroy()


@pytest.mark.parametrize("height, curve_shown, hist_shown",
                         [(330, False, False), (440, True, False), (900, True, True)])
def test_short_panel_drops_plots_that_cannot_be_read(height, curve_shown, hist_shown):
    """On a 768-px screen both plots were squeezed until their ticks
    overlapped: the histogram steps aside first, then the curve (its title
    says why), and both come back when the window is tall enough. The ML
    section, collapsed, is part of the panel in all three."""
    root = _tk_root()
    try:
        host = _host(root, height)
        p = _Panel()
        p._build_quality_panel(host)
        _settle(root)
        head, curve = p._quality_time_rows
        hist = p._quality_hist_rows[1]
        assert bool(curve.winfo_ismapped()) is curve_shown
        assert bool(hist.winfo_ismapped()) is hist_shown
        assert ("taller window" in head.cget("text")) is not curve_shown
        for box, shown in ((curve, curve_shown), (hist, hist_shown)):
            if shown:
                assert box.winfo_height() >= QUALITY_PLOT_MIN_H
        # The section open never shows the histogram; closing it follows the
        # height again.
        p._set_ml_expanded(True)
        root.update()
        assert not hist.winfo_ismapped()
        p._set_ml_expanded(False)
        _settle(root)
        assert bool(hist.winfo_ismapped()) is hist_shown
        assert bool(curve.winfo_ismapped()) is curve_shown
    finally:
        root.destroy()


def test_example_button_follows_the_training_cache(tmp_path, monkeypatch):
    """Add -> ✓ In training examples — Remove -> Update after a peak edit;
    disabled with a hint without a recording or peaks."""
    from ecg.core import ml_detector
    from ecg.io.session import _file_fingerprint
    from ecg.ui import ml_examples
    monkeypatch.setattr(ml_detector, "TRAINING_DATA_DIR", tmp_path / "ml_training")
    rec = tmp_path / "rec.mat"
    rec.write_bytes(b"x")
    root = _tk_root()
    try:
        host = _host(root, 900)
        p = _Panel()
        p._build_quality_panel(host)
        p._set_ml_expanded(True)
        p._refresh_ml_example(force=True)
        _settle(root)
        assert p.btn_ml_example.cget("state") == "disabled"
        assert p.lbl_ml_hint.winfo_ismapped()
        assert "Open a recording" in p.lbl_ml_hint.cget("text")

        p.signal.filepath = str(rec)
        p._refresh_ml_example()
        _settle(root)
        assert p.btn_ml_example.cget("state") == "disabled"
        assert "Detect the peaks" in p.lbl_ml_hint.cget("text")

        fs = 1000
        rp = np.arange(100, 4 * fs - 100, 120)
        sig = np.zeros(4 * fs)
        sig[rp] = 1.0
        p.detection.rpeaks_ok = rp
        p._refresh_ml_example()
        _settle(root)
        assert p.btn_ml_example.cget("text") == ml_examples.BTN_ADD
        assert p.btn_ml_example.cget("state") == "normal"
        assert p._ml_example_mode == "add" and not p.lbl_ml_hint.winfo_ismapped()

        ml_detector.save_training_sample(_file_fingerprint(str(rec)), sig, fs, rp,
                                         source_path=str(rec))
        p._refresh_ml_example()          # the cache changed: seen on the next tick
        _settle(root)
        assert p.btn_ml_example.cget("text") == ml_examples.BTN_REMOVE
        assert p._ml_example_mode == "remove" and not p.btn_ml_remove.winfo_ismapped()

        p.detection.rpeaks_ok = rp[:-1]  # a peak deleted after it was added
        p._refresh_ml_example()
        _settle(root)
        assert p.btn_ml_example.cget("text") == ml_examples.BTN_UPDATE
        assert p._ml_example_mode == "update"
        assert p.btn_ml_remove.winfo_ismapped()
        # The label says "peaks edited": no hint row taken from the plots.
        assert not p.lbl_ml_hint.winfo_ismapped()

        # A job running (Save Session, Add / Update): both buttons wait, as
        # removing now would be undone by that job; back once it is over.
        p._async_busy = True
        p._refresh_ml_example()
        assert p.btn_ml_example.cget("state") == "disabled"
        assert p.btn_ml_remove.cget("state") == "disabled"
        p._async_busy = False
        p._refresh_ml_example()
        assert p.btn_ml_example.cget("state") == "normal"
        assert p.btn_ml_remove.cget("state") == "normal"
    finally:
        root.destroy()


def test_ml_tooltips_float_rather_than_use_the_status_bar():
    """On a 1366-px screen the status bar holds a few words: the ML
    section's explanations are floating tooltips, shown on hover, gone on
    leave."""
    import time
    from ecg.ui import ml_examples
    root = _tk_root()
    try:
        host = _host(root, 900)
        p = _Panel()
        p._build_quality_panel(host)
        p._set_ml_expanded(True)
        _settle(root)

        def tips():
            return [w for w in p.btn_train_ml.winfo_children() + root.winfo_children()
                    if isinstance(w, tk.Toplevel)]

        target = p.btn_train_ml._canvas
        target.event_generate("<Enter>")
        end = time.time() + 2
        while not tips() and time.time() < end:
            root.update()
            time.sleep(0.02)
        [tip] = tips()
        texts = [c.cget("text") for c in tip.winfo_children()]
        assert texts == [ml_examples.TIP_TRAIN]
        target.event_generate("<Leave>")
        root.update()
        assert tips() == []
    finally:
        root.destroy()
