# -*- coding: utf-8 -*-
"""
ecg.ui.widgets
---------------
Small, reusable, standalone widget factories shared across multiple UI
modules -- unlike ECGApp's private _btn()/_switch() helpers (app.py), which
are bound methods used only within app.py itself, these are plain functions
meant to be imported from more than one call site.
"""
from __future__ import annotations

import tkinter as tk
from typing import Optional

import customtkinter as ctk  # type: ignore[import-untyped]
import numpy as np

from ecg.ui.theme import (
    CARD, BORDER, TEXT, MUTED, LIGHT, BLUE,
    GREEN, GREEN_MID, AMBER, RED,
    FONT_KPI_LABEL, FONT_KPI_VALUE, FONT_KPI_HERO, FONT_MICRO, FONT_HINT,
    button_style, combo_style,
)


def make_stat_tile(
    parent,
    label: str,
    value: str,
    *,
    unit: str = "",
    sublabel: str = "",
    value_color: "Optional[str]" = None,
    hero: bool = False,
    card: bool = False,
    accent: "Optional[str]" = None,
) -> ctk.CTkFrame:
    """Build one label/value stat tile. Unplaced -- caller grids/packs it.

    Serves both a dense multi-cell strip (card=False: chrome-less, packed
    tightly into a panel that already supplies the background) and a
    standalone stat-cards panel (card=True: self-contained CARD background
    + padding) from the same function, so neither future call site needs
    its own implementation.

    Returns the outer frame with ``.value_label`` (and ``.sublabel_label``
    if *sublabel* was given) exposed as attributes, so callers can update
    the displayed value later: ``tile.value_label.configure(text="524")``.
    """
    if card:
        outer = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=8,
                              border_width=1, border_color=BORDER)
        inner = ctk.CTkFrame(outer, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=10, pady=10)
    else:
        outer = ctk.CTkFrame(parent, fg_color="transparent")
        inner = outer

    if accent:
        ctk.CTkFrame(inner, fg_color=accent, height=3, corner_radius=2).pack(
            fill="x", pady=(0, 6))

    ctk.CTkLabel(inner, text=label, font=FONT_KPI_LABEL, text_color=MUTED,
                 anchor="w").pack(fill="x")

    value_row = ctk.CTkFrame(inner, fg_color="transparent")
    value_row.pack(fill="x")
    value_label = ctk.CTkLabel(
        value_row, text=value, font=FONT_KPI_HERO if hero else FONT_KPI_VALUE,
        text_color=value_color or TEXT, anchor="w")
    value_label.pack(side="left")
    if unit:
        ctk.CTkLabel(value_row, text=f" {unit}", font=FONT_KPI_LABEL,
                     text_color=MUTED, anchor="w").pack(side="left", padx=(2, 0))

    sublabel_label = None
    if sublabel:
        sublabel_label = ctk.CTkLabel(
            inner, text=sublabel, font=FONT_MICRO, text_color=LIGHT, anchor="w")
        sublabel_label.pack(fill="x")

    outer.value_label = value_label  # type: ignore[attr-defined]
    if sublabel_label is not None:
        outer.sublabel_label = sublabel_label  # type: ignore[attr-defined]
    return outer


def make_sparkline(
    parent,
    values: "Optional[np.ndarray]",
    *,
    width: int = 90,
    height: int = 26,
    color: str = BLUE,
    bg: "Optional[str]" = None,
) -> tk.Canvas:
    """Tiny polyline preview of a short numeric series (e.g. RR_ms per beat).

    Plain tkinter.Canvas rather than a matplotlib Figure -- this is meant
    to be instantiated once per row in a list of recent recordings, where
    a full Figure/FigureCanvasTkAgg per row would be far too heavy for
    what's a handful of pixels of line.
    """
    canvas = tk.Canvas(parent, width=width, height=height,
                       bg=bg or CARD, highlightthickness=0)
    if values is None:
        return canvas
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return canvas
    if len(vals) > 200:   # ~90px wide -- no need to plot more points than that
        idx = np.linspace(0, len(vals) - 1, 200).astype(int)
        vals = vals[idx]
    lo, hi = float(vals.min()), float(vals.max())
    span = (hi - lo) or 1.0
    pad = 3
    xs = np.linspace(pad, width - pad, len(vals))
    ys = height - pad - (vals - lo) / span * (height - 2 * pad)
    canvas.create_line(*(c for xy in zip(xs, ys) for c in xy),
                       fill=color, width=1.4, smooth=True)
    return canvas


class PillGroup(ctk.CTkFrame):
    """The mockup's mode pills: one rounded button per value, 8 px apart, the
    selected one blue. Same .set / .get as the CTkSegmentedButton it
    replaces: .set selects without calling *command*, a click selects and
    calls command(value)."""

    def __init__(self, parent, values, command=None, **kw) -> None:
        super().__init__(parent, fg_color="transparent", **kw)
        self._command = command
        self._value: "Optional[str]" = None
        self._buttons: "dict[str, ctk.CTkButton]" = {}
        for i, v in enumerate(values):
            b = ctk.CTkButton(self, text=v, width=0, command=lambda v=v: self._click(v),
                              **button_style("pill_off"))
            b.pack(side="left", padx=(0 if i == 0 else 8, 0))
            self._buttons[v] = b

    def _click(self, value: str) -> None:
        self.set(value)
        if self._command is not None:
            self._command(value)

    def set(self, value: str) -> None:
        if value not in self._buttons:
            raise ValueError(f"no pill named {value!r}")
        self._value = value
        for v, b in self._buttons.items():
            look = button_style("pill_on" if v == value else "pill_off")
            b.configure(fg_color=look["fg_color"], hover_color=look["hover_color"],
                        text_color=look["text_color"], border_color=look["border_color"],
                        font=look["font"])

    def get(self) -> str:
        return self._value or ""


class FamilyCombo(ctk.CTkFrame):
    """A metric chooser in two steps: the family first (Heart rate, HRV time
    domain, frequency, non-linear... ecg.core.metric_families), then the
    metric of that family. Long lists of HR / SD / power metrics are easier
    to find this way than in one flat list.

    Same .set / .get / .configure(values=...) / .cget("values") as the
    CTkComboBox it replaces: .set selects without calling *command*, a pick
    in either combo calls command(metric). Each family remembers the metric
    last chosen in it. Up / Down step through the list of the focused combo,
    Return or space opens it.

    *flat_max*: up to that many metrics, one flat list (in family order)
    instead: two drop-downs to reach one of a handful of metrics is a click
    too many. 0: always two steps. With no value chosen, the first metric of
    the first family is shown (a heart rate before a count of beats)."""

    def __init__(self, parent, command=None, values=(), width: int = 260,
                 family_width: int = 170, flat_max: int = 0, **kw) -> None:
        super().__init__(parent, fg_color="transparent", **kw)
        self._command = command
        self._flat_max = int(flat_max)
        self._flat = False
        self._values: "list[str]" = []
        self._by_family: "dict[str, list[str]]" = {}
        self._last: "dict[str, str]" = {}
        self.cb_family = ctk.CTkComboBox(self, width=family_width, values=[""], state="readonly",
                                         command=self._pick_family, **combo_style())
        self.cb_family.pack(side="left", padx=(0, 4))
        self.cb_metric = ctk.CTkComboBox(self, width=width, values=[""], state="readonly",
                                         command=self._pick_metric, **combo_style())
        self.cb_metric.pack(side="left")
        for cb, step in ((self.cb_family, self._step_family), (self.cb_metric, self._step_metric)):
            entry = getattr(cb, "_entry", None)
            if entry is None:
                continue
            entry.bind("<Up>", lambda _e, s=step: (s(-1), "break")[1])
            entry.bind("<Down>", lambda _e, s=step: (s(1), "break")[1])
            for key in ("<Return>", "<space>"):
                entry.bind(key, lambda _e, c=cb: (c._open_dropdown_menu(), "break")[1])
        if values:
            self.set_values(values)

    # ── the CTkComboBox surface ──

    def configure(self, require_redraw=False, **kwargs) -> None:
        if "values" in kwargs:
            self.set_values(kwargs.pop("values"))
        if kwargs or require_redraw:
            super().configure(require_redraw=require_redraw, **kwargs)

    def cget(self, attribute_name: str):
        if attribute_name == "values":
            return list(self._values)
        return super().cget(attribute_name)

    def set_values(self, values) -> None:
        """The metrics offered (in their order), sorted into families. The
        metric shown stays when it is still offered."""
        from ecg.core.metric_families import by_family
        cur = self.get()
        self._values = [str(v) for v in values or [] if str(v)]
        self._by_family = dict(by_family(self._values))
        self._last = {f: m for f, m in self._last.items()
                      if m in self._by_family.get(f, [])}
        self.cb_family.configure(values=list(self._by_family) or [""])
        flat = 0 < len(self._values) <= self._flat_max
        if flat != self._flat:
            self._flat = flat
            if flat:
                self.cb_family.pack_forget()
            else:
                self.cb_family.pack(side="left", padx=(0, 4), before=self.cb_metric)
        self.set(cur if cur in self._values else self.first())

    def first(self) -> str:
        """The first metric of the first family ("" when none)."""
        return next((ns[0] for ns in self._by_family.values() if ns), "")

    def _offered(self, fam: str) -> "list[str]":
        """The metrics of the metric combo: those of *fam*, or every one in
        family order when the list is flat."""
        if self._flat:
            return [m for ns in self._by_family.values() for m in ns]
        return self._by_family.get(fam, [])

    def set(self, value: str) -> None:
        from ecg.core.metric_families import family_of
        value = str(value or "")
        if value not in self._values:
            self.cb_family.set("")
            self.cb_metric.configure(values=[""])
            self.cb_metric.set("")
            return
        fam = family_of(value)
        self._last[fam] = value
        self.cb_family.set(fam)
        self.cb_metric.configure(values=self._offered(fam) or [value])
        self.cb_metric.set(value)

    def get(self) -> str:
        v = self.cb_metric.get()
        return v if v in self._values else ""

    def family(self) -> str:
        return self.cb_family.get()

    # ── picks ──

    def _pick_family(self, fam: str) -> None:
        names = self._by_family.get(fam)
        if not names:
            return
        m = self._last.get(fam, names[0])
        self.set(m if m in names else names[0])
        if self._command is not None:
            self._command(self.get())

    def _pick_metric(self, value: str) -> None:
        self.set(value)
        if self._command is not None:
            self._command(self.get())

    def _step_family(self, d: int) -> None:
        fams = list(self._by_family)
        if fams:
            i = fams.index(self.family()) if self.family() in fams else -d
            new = fams[max(0, min(len(fams) - 1, i + d))]
            if new != self.family():
                self._pick_family(new)

    def _step_metric(self, d: int) -> None:
        names = self._offered(self.family())
        if names:
            i = names.index(self.get()) if self.get() in names else -d
            new = names[max(0, min(len(names) - 1, i + d))]
            if new != self.get():
                self._pick_metric(new)
