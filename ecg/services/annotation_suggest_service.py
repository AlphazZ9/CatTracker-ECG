# -*- coding: utf-8 -*-
"""
ecg.services.annotation_suggest_service
---------------------------------------
Annotations suggested from the other channels of the recording open: its
marker / keyboard channels and the stimulus pulses of its waveform
channels (ecg.core.annotation_suggest, ecg.io.event_channels).

Moved unchanged from the Tk annotation_suggest_dialog (collect, and the
colours _add gave the ticked suggestions), so both interfaces suggest and
add the same annotations. Reads the file only: safe in a background thread.
"""
from __future__ import annotations

from typing import Optional

from ecg.core import annotation_suggest as sug
from ecg.io import event_channels as evc
from ecg.io.export import ANNOTATION_COLORS


def collect(filepath: Optional[str], channel: Optional[str], crop: "Optional[dict]",
            t_max: Optional[float], annotations: "list[dict]") -> "tuple[list[dict], list[str]]":
    """(suggestions not annotated yet, the channels read) for the file. Runs
    off the main thread: reads the file only."""
    crop = crop or {}
    t0 = float(crop.get("t_start") or 0.0)
    t1 = crop.get("t_end")
    # A crop to the file end is "no end crop" (0).
    dur = crop.get("file_duration_s")
    t1 = 0.0 if t1 is None or (dur is not None and float(t1) >= float(dur)) else float(t1)
    events = evc.read_event_channels(filepath, t0, t1)
    waves = evc.read_waveforms(filepath, skip=channel, t_start=t0, t_end=t1)
    found = (sug.from_events({k: v for k, v in events.items() if v}, t_max=t_max)
             + sug.from_waveforms(waves, t_max=t_max))
    found.sort(key=lambda d: d["t_start"])
    read = [k for k, v in events.items() if v] + list(waves)
    return sug.drop_existing(found, annotations), read


def to_annotations(picked: "list[tuple[dict, str]]") -> "list[dict]":
    """The annotations of the ticked suggestions, each (suggestion, label
    typed); one colour per source channel, in order."""
    colors = [c for c, _n in ANNOTATION_COLORS]
    by_source: "dict[str, str]" = {}
    out = []
    for s, label in picked:
        color = by_source.setdefault(s["source"], colors[len(by_source) % len(colors)])
        out.append({"t_start": float(s["t_start"]), "t_end": float(s["t_end"]),
                    "label": label.strip() or s["label"], "color": color})
    return out
