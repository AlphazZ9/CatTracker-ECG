"""Annotations suggested from the other channels of the recording open
(Annotations → Suggest from channels…): the marker / keyboard channels and
the stimulus pulses of its waveform channels, read in the background; the
user ticks the ones to add and can rename them."""
from __future__ import annotations

import logging
import threading
import tkinter as tk
from typing import TYPE_CHECKING, Any, Callable, Optional

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.services.annotation_suggest_service import collect, to_annotations
from ecg.ui.theme import (
    BG, CARD, BORDER, FONT_HINT, FONT_SMALL, FONT_SUBSECTION, MUTED, ORANGE, PANEL, TEXT,
    button_style, input_style,
)

if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")


class AnnotationSuggestDialog(ctk.CTkToplevel):
    """The suggestions, each with a tick and an editable label; Add selected
    calls *on_add(list of annotations)*."""

    def __init__(self, parent: Any, app: "ECGApp",
                 on_add: "Callable[[list[dict]], None]") -> None:
        super().__init__(parent)
        self.title("Suggest annotations from channels")
        self.geometry("720x460")
        self.configure(fg_color=PANEL)
        self.transient(parent.winfo_toplevel())
        self._app, self._on_add = app, on_add
        self._rows: "list[tuple[tk.BooleanVar, ctk.CTkEntry, dict]]" = []
        self._lbl = ctk.CTkLabel(self, text="Reading the marker, keyboard and stimulus "
                                            "channels…", font=FONT_SMALL, text_color=MUTED,
                                 anchor="w", justify="left", wraplength=680)
        self._lbl.pack(fill="x", padx=12, pady=(10, 4))
        self._list = ctk.CTkScrollableFrame(self, fg_color=CARD)
        self._list.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=10, pady=(0, 10))
        self._btn_add = ctk.CTkButton(bar, text="Add selected", width=0, state="disabled",
                                      command=self._add, **button_style("primary"))
        self._btn_add.pack(side="right")
        ctk.CTkButton(bar, text="Close", width=0, command=self.destroy,
                      **button_style("secondary")).pack(side="right", padx=(0, 6))
        for text, val in (("Tick all", True), ("Untick all", False)):
            ctk.CTkButton(bar, text=text, width=0, command=lambda v=val: self._tick(v),
                          **button_style("secondary")).pack(side="left", padx=(0, 6))
        ctk.CTkLabel(bar, text="A lone marker lasts 1 s; a train spans its markers or pulses.",
                     font=FONT_HINT, text_color=MUTED).pack(side="left", padx=(6, 0))
        self._result: "Optional[tuple[list[dict], list[str]] | str]" = None
        sig = app.signal
        t = getattr(app, "_time", None)
        t_max = float(t[-1]) if t is not None and len(t) else None
        args = (sig.filepath, sig.channel, sig.crop, t_max, list(app.analysis.annotations))

        def _work() -> None:
            try:
                self._result = collect(*args)
            except Exception as exc:          # shown, never raised in a thread
                log.warning("annotation suggestions: %s", exc)
                self._result = f"Could not read the channels: {exc}"
        threading.Thread(target=_work, daemon=True).start()
        self.after(100, self._poll)

    def _poll(self) -> None:
        if not self.winfo_exists():
            return
        if self._result is None:
            self.after(100, self._poll)
            return
        self.show(self._result)

    def show(self, result: "tuple[list[dict], list[str]] | str") -> None:
        for w in self._list.winfo_children():
            w.destroy()
        self._rows = []
        if isinstance(result, str):
            self._lbl.configure(text=result, text_color=ORANGE)
            return
        found, read = result
        if not found:
            self._lbl.configure(
                text=("No marker, keyboard or stimulus-pulse event that is not annotated "
                      "already." if read else
                      "This file has no marker or keyboard channel and no channel of "
                      "stimulus pulses (Spike2 .smr or its MATLAB export)."),
                text_color=MUTED)
            return
        self._lbl.configure(text=f"{len(found)} suggestion{'s' if len(found) != 1 else ''} "
                                 f"from {', '.join(read)}: untick the ones to leave out, "
                                 "edit a label if needed.", text_color=TEXT)
        hdr = ctk.CTkFrame(self._list, fg_color=BORDER, corner_radius=4)
        hdr.pack(fill="x", padx=2, pady=(0, 2))
        for txt, w in (("", 30), ("Start (s)", 90), ("End (s)", 90), ("Label", 380)):
            ctk.CTkLabel(hdr, text=txt, width=w, font=FONT_SUBSECTION, text_color=TEXT,
                         anchor="w").pack(side="left", padx=3)
        for i, s in enumerate(found):
            row = ctk.CTkFrame(self._list, fg_color=CARD if i % 2 == 0 else BG, corner_radius=0)
            row.pack(fill="x", padx=2, pady=1)
            var = tk.BooleanVar(value=True)
            ctk.CTkCheckBox(row, text="", variable=var, width=30, checkbox_width=16,
                            checkbox_height=16).pack(side="left", padx=(4, 0))
            for txt in (f"{s['t_start']:.3f}", f"{s['t_end']:.3f}"):
                ctk.CTkLabel(row, text=txt, width=90, font=FONT_SMALL, text_color=TEXT,
                             anchor="w").pack(side="left", padx=3)
            ent = ctk.CTkEntry(row, width=380, **input_style())
            ent.insert(0, s["label"])
            ent.pack(side="left", padx=3, pady=2)
            self._rows.append((var, ent, s))
        self._btn_add.configure(state="normal")

    def _tick(self, on: bool) -> None:
        for var, _e, _s in self._rows:
            var.set(on)

    def _add(self) -> None:
        out = to_annotations([(s, ent.get()) for var, ent, s in self._rows if var.get()])
        if out:
            self._on_add(out)
        self.destroy()
