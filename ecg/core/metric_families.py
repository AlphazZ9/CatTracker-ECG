# -*- coding: utf-8 -*-
"""
ecg.core.metric_families
------------------------
Which family a metric belongs to (Heart rate, HRV time domain, frequency
domain, non-linear...), so that long metric lists can be shown in
sub-categories: the Group tab's metric chooser and results table, the
Compare segments table and the Family column of some Excel sheets.

A metric table row already carries its family (MetricRow.group); the group
tables and the comparisons only keep a column name such as "SDNN (ms)",
"Poincaré SD1 (ms) Pre" or "HRV_SDANN1 (ms)". family_of() works the family
out of such a name: the NeuroKit columns (HRV_...) by the frame they come
from, every other name by a few patterns, in an order where the narrower
family is tried first ("SD01 (whole-recording SD1)" is Autonomic, not
Poincaré). A name nothing matches is "Other".

No UI import.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional, Sequence

HEART_RATE = "Heart rate"
TIME_DOMAIN = "HRV time domain"
FREQUENCY = "HRV frequency domain"
NONLINEAR = "HRV non-linear"
ASYMMETRY = "RR asymmetry"
AUTONOMIC = "Autonomic"
INTERVALS = "ECG intervals"
EVENTS = "Abnormal events"
QUALITY = "Quality"
DATA = "Data analysed"
OTHER = "Other"

# The order families are listed in: the order of the Rhythm & HRV tab (time,
# frequency, non-linear) after the heart rate, then the other modules; the
# amount of data (beats, duration, blocks) last: it is not an outcome.
FAMILIES = (HEART_RATE, TIME_DOMAIN, FREQUENCY, NONLINEAR, ASYMMETRY, AUTONOMIC,
            INTERVALS, EVENTS, QUALITY, DATA, OTHER)

# MetricRow.group -> family. The RR exclusion rows sit in "Heart rate" in the
# metric table (they come from the same report), but are about data quality.
_GROUP_FAMILY = {
    "Heart rate": HEART_RATE,
    "HRV time domain": TIME_DOMAIN,
    "HRV frequency domain": FREQUENCY,
    "HRV non-linear": NONLINEAR,
    "RR asymmetry": ASYMMETRY,
    "Autonomic": AUTONOMIC,
    "Intervals": INTERVALS,
    "Abnormal events": EVENTS,
    "Premature beats": EVENTS,
    "Signal quality": QUALITY,
    "Artifact correction": QUALITY,
}
# The counts and the duration analysed sit in "Heart rate" there too, but
# say how much data there is, not what the heart rate was.
_DATA_KEYS = ("hr.n_beats", "hr.n_valid", "rr.n_intervals", "core.duration_s")
_QUALITY_KEYS = ("rr.n_excluded", "rr.n_too_short", "rr.n_too_long", "rr.n_spanning_removed",
                 "rr.excluded_pct", "rr.excluded_time_s", "rr.rr_min_ms", "rr.rr_max_ms")

# The NeuroKit time- and frequency-domain columns, and its asymmetry indices
# (Guzik, Slope, Area, Porta and the contributions C1d, SD1a...: the same
# numbers as the app's own Porta / Guzik rows, so one family for both);
# every other HRV_ column comes from the non-linear frame (HRV_CSI there is
# Toichi's, not Autonomic).
_NK_ASYM = re.compile(r"^HRV_(GI|SI|AI|PI|C[12]?[da]|SD[12][da]|SDNN[da])$")
_NK_TIME = re.compile(
    r"^HRV_(MeanNN|SDNN|SDANN\d*|SDNNI\d*|RMSSD|SDSD|CVNN|CVSD|MedianNN|MadNN|MCVNN|IQRNN|"
    r"SDRMSSD|Prc\d+NN|pNN\d+|MinNN|MaxNN|HTI|TINN)$")
_NK_FREQ = re.compile(r"^HRV_(ULF|VLF|LF|HF|VHF|TP|LFHF|LFn|HFn|LnHF|\w*peak)$")

_I = re.IGNORECASE
# (family, pattern), tried in this order.
_RULES = (
    (QUALITY, re.compile(r"quality|correlation|exclud|artifact|artefact|removed|coverage|"
                         r"poor-shape|below 0\.9|\bbound\b|duplicate|non-physio|ectopic", _I)),
    (AUTONOMIC, re.compile(r"\br?c[sp]i\b|\bsd0[12]\b|\bd0\b|^d$|\bccd\b|autonomic", _I)),
    (ASYMMETRY, re.compile(r"porta|guzik|asymmetr|decelerations|accelerations|^pairs\b|"
                           r"no change", _I)),
    (EVENTS, re.compile(r"premature|couplet|\bruns?\b|pause|brady|tachy|av[_ ]delay|"
                        r"irregular|arrhythm", _I)),
    (INTERVALS, re.compile(r"\b(PR|QRS|QTc?)\b|[PT][ -]wave|beats measured", _I)),
    (NONLINEAR, re.compile(r"\bsd[12]\b|sd1/sd2|poincar|entrop|sampen|apen|dfa|α|alpha|"
                           r"fractal|non-?linear", _I)),
    (FREQUENCY, re.compile(r"\b(u?v?lf|hf|vhf|tp)\b|lf/hf|n\.u\.|power|spectr", _I)),
    (TIME_DOMAIN, re.compile(r"sdnn|rmssd|\bpnn\d*|sdsd|meannn|sdann|cvnn|tinn|\bhti\b", _I)),
    # Beats, Duration, Blocks used, Valid RR, Analysed duration: how much
    # data a value rests on (after Quality, Events and Intervals, so "Beats
    # below 0.90", "Pause duration" or "QRS duration" stay there).
    (DATA, re.compile(r"^beats\b|duration|^blocks? used|^valid rr\b|^rr intervals\b", _I)),
    (HEART_RATE, re.compile(r"\bhr\b|heart rate|\brr\b|bpm|\bnn\b", _I)),
)


def base_name(name: Any) -> str:
    """The metric part of a chooser entry or a display name: without the
    pair suffix the Group tab adds (" · Post vs Pre") and outer spaces."""
    return str(name or "").split(" · ")[0].strip()


def family_of(name: Any) -> str:
    """The family of a metric column name ("SDNN (ms)", "Poincaré SD1 (ms)",
    "HRV_SDANN1 (ms)", "rCSI", "LF (n.u.) (%)"...); OTHER when none fits."""
    text = base_name(name)
    if not text:
        return OTHER
    head = text.split(" (")[0].strip()
    if head.startswith("HRV_"):
        if _NK_ASYM.match(head):
            return ASYMMETRY
        if _NK_TIME.match(head):
            return TIME_DOMAIN
        if _NK_FREQ.match(head):
            return FREQUENCY
        return NONLINEAR
    for family, rx in _RULES:
        if rx.search(text):
            return family
    return OTHER


def family_of_row(row: Any) -> str:
    """The family of a core.metric_table.MetricRow: its group, with the RR
    exclusion rows under Quality and the counts / duration under Data
    analysed; a row of no known group goes by its label."""
    if getattr(row, "key", "") in _QUALITY_KEYS:
        return QUALITY
    if getattr(row, "key", "") in _DATA_KEYS:
        return DATA
    fam = _GROUP_FAMILY.get(str(getattr(row, "group", "") or ""))
    return fam if fam is not None else family_of(getattr(row, "label", ""))


def family_rank(family: str) -> int:
    return FAMILIES.index(family) if family in FAMILIES else len(FAMILIES)


def by_family(names: "Iterable[Any]", family: "Optional[Any]" = None
              ) -> "list[tuple[str, list[Any]]]":
    """(family, its names) in FAMILIES order, each list in the order given;
    families without a name are left out. *family* maps a name to its family
    (default family_of)."""
    fam = family or family_of
    groups: "dict[str, list[Any]]" = {}
    for n in names:
        groups.setdefault(fam(n), []).append(n)
    return sorted(groups.items(), key=lambda kv: family_rank(kv[0]))


def families_in(names: "Sequence[Any]") -> "list[str]":
    """The families of *names*, in FAMILIES order."""
    return [f for f, _ in by_family(names)]


def is_descriptive(name: Any) -> bool:
    """True for a Data analysed metric (Beats, Duration, Blocks used, Valid
    RR...): how much data a value rests on, not an outcome. Such rows are
    shown per group but never tested, and are left out of the Holm family."""
    return family_of(name) == DATA
