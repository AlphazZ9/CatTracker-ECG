"""Nothing moves while a job runs: the progress bar sits in a slot of fixed
size in the header (placed, not packed), a busy button keeps its width (its
longer busy label cut to it), a figure is sized with its own dpi (the Tk
canvas renders at the screen's device pixel ratio: sized with the nominal
dpi it overflowed its frame and was resized back and forth), and Compare
segments is not rebuilt each time it is opened without a recording."""
from __future__ import annotations

import time
import tkinter as tk
from types import SimpleNamespace

import pandas as pd
import pytest

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.ui.app_async import fit_progress_text, freeze_button_width, release_button_width
from ecg.ui.plots import CanvasSlot


@pytest.fixture
def root():
    try:
        r = ctk.CTk()
    except tk.TclError:
        pytest.skip("no display")
    r.geometry("900x500+0+0")
    yield r
    for after_id in r.tk.splitlist(r.tk.call("after", "info")):
        r.tk.call("after", "cancel", after_id)
    r.destroy()


def _pump(r, sec=0.6):
    t = time.time()
    while time.time() - t < sec:
        r.update()
        time.sleep(0.01)


def test_progress_text_keeps_the_percentage_and_time_left():
    measure = 6  # px per character without a Tk font
    short = fit_progress_text("42%  Core", "  (~3s left)", width=300)
    assert short == "42%  Core  (~3s left)"
    long = fit_progress_text("42%  " + "x" * 200, "  (~3s left)", width=120)
    assert long.startswith("42%") and long.endswith("…  (~3s left)")
    assert len(long) * measure <= 120 or len(long) < 40


def test_busy_button_keeps_its_width(root):
    row = ctk.CTkFrame(root)
    row.pack(fill="x")
    btn = ctk.CTkButton(row, text="Detect Peaks")
    btn.pack(side="left")
    nxt = ctk.CTkButton(row, text="Analyze all")
    nxt.pack(side="left")
    _pump(root)
    w, x = btn.winfo_width(), nxt.winfo_rootx()
    label = freeze_button_width(btn, "Checking for saved session…")
    btn.configure(text=label)
    _pump(root)
    assert label.endswith("…") and len(label) < len("Checking for saved session…")
    assert btn.winfo_width() == w and nxt.winfo_rootx() == x
    btn.configure(text="Detect Peaks")
    release_button_width(btn)
    _pump(root)
    assert btn.winfo_width() == w and not hasattr(btn, "_busy_own_width")
    # A stub (tests, a destroyed button): the label as given.
    assert freeze_button_width(SimpleNamespace(), "Saving…") == "Saving…"


def test_progress_slot_shows_without_moving_anything(root):
    """The header's arrangement: the slot is always packed, its content placed."""
    row = ctk.CTkFrame(root, height=60)
    row.pack(fill="x")
    row.pack_propagate(False)
    right = ctk.CTkButton(row, text="Open")
    right.pack(side="right")
    slot = ctk.CTkFrame(row, width=300, height=40, fg_color="transparent")
    slot.pack(side="right")
    slot.pack_propagate(False)
    inner = ctk.CTkFrame(slot, fg_color="transparent")
    ctk.CTkProgressBar(inner).pack(fill="x")
    ctk.CTkLabel(inner, text="42%  Core HRV").pack(fill="x")
    below = ctk.CTkLabel(root, text="tabs")
    below.pack()
    _pump(root)
    before = (right.winfo_rootx(), below.winfo_rooty(), slot.winfo_width())
    inner.place(relx=0, rely=0, relwidth=1, relheight=1)
    _pump(root)
    assert (right.winfo_rootx(), below.winfo_rooty(), slot.winfo_width()) == before
    inner.place_forget()
    _pump(root)
    assert (right.winfo_rootx(), below.winfo_rooty(), slot.winfo_width()) == before


def test_a_figure_fills_its_frame_at_the_screen_dpi(root):
    holder = tk.Frame(root, width=700, height=300)
    holder.pack(fill="both", expand=True)
    slot = CanvasSlot(holder, 12, 4, toolbar=False)
    slot.update(lambda fig: fig.add_subplot(111).plot([0, 1], [0, 1]))
    _pump(root, 1.0)
    w_in, h_in = slot.fig.get_size_inches()
    assert abs(w_in * slot.fig.dpi - slot._cv_frame.winfo_width()) <= 3
    assert abs(h_in * slot.fig.dpi - slot._cv_frame.winfo_height()) <= 3
    # Settled: another fit with the same frame draws nothing.
    calls = []
    slot._draw_fn = lambda fig: calls.append(1)
    slot._fit()
    assert calls == []


def test_compare_tab_is_not_rebuilt_without_a_recording(root):
    from ecg.ui.compare_segments import CompareSegmentsMixin
    holder = ctk.CTkFrame(root)
    built = []
    app = SimpleNamespace(_compare_holder=holder, _compare_built_for=None,
                          _compare_signature=lambda: None,
                          _build_group_pointer=lambda h: built.append(
                              ctk.CTkLabel(h, text="group").pack()))
    CompareSegmentsMixin._refresh_compare_tab(app)
    CompareSegmentsMixin._refresh_compare_tab(app)
    assert len(built) == 1


def test_a_waiting_draw_is_made_when_its_scrolling_page_is_shown_again(root):
    """A slot inside a CTkScrollableFrame gets no <Map> when its tab comes
    back: the size check that follows draws what was deferred."""
    tab = ctk.CTkFrame(root)
    tab.pack(fill="both", expand=True)
    page = ctk.CTkScrollableFrame(tab)
    page.pack(fill="both", expand=True)
    holder = tk.Frame(page, width=600, height=250)
    holder.pack(fill="x")
    holder.pack_propagate(False)
    slot = CanvasSlot(holder, 6, 2.5, toolbar=False)
    _pump(root)
    tab.pack_forget()
    _pump(root, 0.2)
    slot.update(lambda fig: fig.add_subplot(111).set_title("drawn"))
    assert slot._pending == "draw"
    tab.pack(fill="both", expand=True)
    _pump(root, 1.0)
    slot._fit()
    assert slot._pending is None and slot.fig.axes[0].get_title() == "drawn"


def test_the_tab_legend_gives_the_heads_the_room(root):
    """With eight tabs the state legend (docked right, packed first) squeezed
    the last heads on a 1280 px window; its texts go when the heads need the
    room, and come back when the window is wide again."""
    from ecg.ui.tab_keys import TAB_ORDER
    from ecg.ui.tabbar import TabBar
    root.geometry("1280x200+0+0")
    bar = TabBar(root)
    bar.pack(fill="both", expand=True)
    for n in TAB_ORDER:
        bar.add(n)
    _pump(root)
    last = bar._heads[TAB_ORDER[-1]]["head"]
    assert bar._legend_compact and last.winfo_width() == last.winfo_reqwidth()
    root.geometry("2000x200+0+0")
    _pump(root)
    assert not bar._legend_compact
    assert all(t.winfo_ismapped() for _g, t in bar._legend)
    from ecg.ui.tabbar import HEAD_PAD
    assert bar._head_pad == HEAD_PAD


def test_placeholders_wait_for_their_tab_while_the_app_builds(root):
    """At start-up the window is not shown yet, so every slot drew its
    placeholder at once (~35 renders, ~1 s): while the app builds they wait,
    and the size check of a shown slot draws them."""
    root.deiconify()
    holder = tk.Frame(root, width=500, height=200)
    holder.pack(fill="both", expand=True)
    CanvasSlot.building = True
    try:
        slot = CanvasSlot(holder, 5, 2, toolbar=False, placeholder="waiting")
    finally:
        CanvasSlot.building = False
    assert slot._pending == "placeholder" and not slot.fig.axes
    _pump(root, 1.0)
    slot._fit()
    assert slot._pending is None and slot.fig.axes
    # A tab's own empty state drawn while the app builds waits too (no
    # recording is loaded then), and is drawn once shown.
    CanvasSlot.building = True
    try:
        slot.update(lambda fig: fig.add_subplot(111).set_title("empty state"))
    finally:
        CanvasSlot.building = False
    assert slot._pending == "draw"
    slot.flush()
    assert slot.fig.axes[0].get_title() == "empty state"
    # After the build, an update draws at once.
    slot.update(lambda fig: fig.add_subplot(111).set_title("data"))
    assert slot.fig.axes[0].get_title() == "data"


def test_no_group_table_is_opened_at_start_up(root):
    """Nothing is read at start-up: the Group tab opens empty."""
    from ecg.ui.tab_group import TabGroupMixin
    app = SimpleNamespace(_bind_hover_tip=lambda *a, **k: None)
    frame = ctk.CTkFrame(root)
    TabGroupMixin._build_group_page(app, frame)
    texts = [str(w.cget("text")) for w in _all(frame) if isinstance(w, ctk.CTkLabel)]
    assert "No group table open" in texts and app._grouptab_state["path"] == ""


def _all(w):
    yield w
    for c in w.winfo_children():
        yield from _all(c)



def test_a_hidden_canvas_renders_when_shown(root):
    """Matplotlib renders a canvas on every <Configure>, hidden ones too:
    a render asked for while the canvas is hidden waits until it is shown."""
    root.deiconify()
    tab = ctk.CTkFrame(root)
    tab.pack(fill="both", expand=True)
    holder = tk.Frame(tab, width=500, height=200)
    holder.pack(fill="both", expand=True)
    slot = CanvasSlot(holder, 5, 2, toolbar=False)
    slot.update(lambda fig: fig.add_subplot(111).plot([0, 1]))
    _pump(root, 0.8)
    tab.pack_forget()
    _pump(root, 0.2)
    renders = []
    import ecg.ui.plots as plots
    orig = plots.FigureCanvasTkAgg.draw
    plots.FigureCanvasTkAgg.draw = lambda self: (renders.append(1), orig(self))[1]
    try:
        slot.canvas.draw()
        assert renders == [] and slot.canvas._stale_hidden
        tab.pack(fill="both", expand=True)
        _pump(root, 0.8)
        slot.flush()
        assert renders and not slot.canvas._stale_hidden
    finally:
        plots.FigureCanvasTkAgg.draw = orig
