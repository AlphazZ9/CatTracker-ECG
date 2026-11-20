# -*- coding: utf-8 -*-
"""
ecg.io.subject_name
-------------------
The animal and its sex read from a recording's file name:
"Stim BNST - 1146 F.smr" is subject 1146, female; "Baseline_STIM_BNST_206.mat"
is subject 206, sex unknown.

Rules (``subject_from_filename``), on the file name without its folder and
extension:

* sex: a standalone letter M or F (Male / Female) -- a token of its own,
  separated by a space, _ or - ("1146 F", "M_206"), or glued to a number
  ("1146F", "F1146"). The letters of a word never count (the F of "FIELD",
  the M of "Mouse"). Lower case m / f never count glued to a number
  ("10m" is ten minutes); a lower case m does not count after a number at
  all ("10 m", "30_m" are minutes), a lower case f does ("1146 f"). Two
  different sexes in one name, none next to a number, give no sex.
* subject: the number directly before, else directly after, that M / F --
  but not a number glued to a word on its other side ("Ch6_F", "Exp4_F":
  a channel or an experiment, not an animal), nor a single digit separated
  from the letter; otherwise the LAST number of two digits or more that is
  not part of a date or a time (2026-11-19, 20261119, 19112026, 173903,
  17-39-03), preferring one not followed by a unit ("100Hz", "10 min",
  "30 s").

Nothing here may import from ecg.ui.
"""
from __future__ import annotations

import os
import re
from typing import Optional

# Tokens: runs of letters (any alphabet) or of digits.
_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+")
# What may sit between a sex letter and its number.
_GAP_RE = re.compile(r"^[\s_\-()\[\]]*$")
# Dates and times written with separators.
_SEP = r"[-_.]"
_DATE_YMD_RE = re.compile(rf"(?<!\d)(\d{{4}}){_SEP}(\d{{1,2}}){_SEP}(\d{{1,2}})(?!\d)")
_DATE_DMY_RE = re.compile(rf"(?<!\d)(\d{{1,2}}){_SEP}(\d{{1,2}}){_SEP}(\d{{4}})(?!\d)")
_TIME_RE = re.compile(r"(?<!\d)(\d{1,2})[-_.:hH](\d{2})(?:[-_.:mM](\d{2}))?(?!\d)")


def _year_ok(y: int) -> bool:
    return 1990 <= y <= 2099


def _ymd_ok(y: int, m: int, d: int) -> bool:
    return _year_ok(y) and 1 <= m <= 12 and 1 <= d <= 31


def _hms_ok(h: int, m: int, s: int = 0) -> bool:
    return 0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59


def _compact_date(run: str) -> bool:
    """A digit run that reads as a date or a time stamp: YYYYMMDD / DDMMYYYY
    (and longer runs starting with one: 20261119173903), YYMMDD / DDMMYY or
    HHMMSS."""
    n = len(run)
    if n >= 8:
        a = run[:8]
        if _ymd_ok(int(a[:4]), int(a[4:6]), int(a[6:8])):
            return True
        if n == 8 and _ymd_ok(int(a[4:8]), int(a[2:4]), int(a[:2])):
            return True
        return False
    if n == 6:
        x, y, z = int(run[:2]), int(run[2:4]), int(run[4:6])
        return (_hms_ok(x, y, z) or (1 <= y <= 12 and 1 <= z <= 31)
                or (1 <= x <= 31 and 1 <= y <= 12))
    return False


def _date_spans(stem: str) -> "list[tuple[int, int]]":
    """Character spans of the dates and times written with separators."""
    spans = []
    for m in _DATE_YMD_RE.finditer(stem):
        if _ymd_ok(int(m[1]), int(m[2]), int(m[3])):
            spans.append(m.span())
    for m in _DATE_DMY_RE.finditer(stem):
        if _ymd_ok(int(m[3]), int(m[2]), int(m[1])):
            spans.append(m.span())
    # Times are looked for outside the dates ("2026-11-19_17-39-03": not 11-19_17).
    blank = list(stem)
    for a, b in spans:
        blank[a:b] = " " * (b - a)
    for m in _TIME_RE.finditer("".join(blank)):
        h, mi, s = int(m[1]), int(m[2]), int(m[3]) if m[3] else 0
        if not _hms_ok(h, mi, s):
            continue
        # HH-MM alone is too common a pair of numbers: a time only with its
        # seconds, or right after a date.
        after_date = any(0 <= m.start() - e <= 1 for _s, e in spans)
        if m[3] or after_date:
            spans.append(m.span())
    return spans


# Units a number may carry ("100Hz", "10 min", "30 s"): such a number is a
# setting or a duration, not an animal.
_UNITS = frozenset({
    "hz", "khz", "mhz", "a", "ma", "ua", "µa", "na", "v", "mv", "uv", "µv",
    "w", "mw", "s", "sec", "secs", "ms", "us", "µs", "min", "mins", "mn", "m",
    "h", "hr", "hrs", "d", "days", "g", "mg", "ug", "µg", "kg", "ml", "ul",
    "µl", "nl", "mm", "cm", "um", "µm", "nm", "pct", "bpm", "x",
})


def _sex_of(tok: str, num_before: bool, glued: bool) -> Optional[str]:
    """*num_before*: a number comes right before *tok* (any separator);
    *glued*: a number touches it. Lower case f counts when set apart from
    the number ("1146 f"), lower case m only with no number before it
    ("10 m" is ten minutes)."""
    if tok in ("M", "F"):
        return tok
    if tok == "f" and not glued:
        return "F"
    if tok == "m" and not glued and not num_before:
        return "M"
    return None


def subject_from_filename(path: "Optional[str]") -> "tuple[Optional[str], Optional[str]]":
    """(subject, sex) read from the name of *path* ("Stim BNST - 1146 F.smr"
    gives ("1146", "F")); None for what the name does not tell. *sex* is
    "M", "F" or None. See the module docstring for the rules."""
    if not path:
        return None, None
    stem = os.path.splitext(os.path.basename(str(path).replace("\\", "/")))[0]
    if not stem:
        return None, None
    toks = [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(stem)]
    spans = _date_spans(stem)

    def is_num(i: int) -> bool:
        return 0 <= i < len(toks) and toks[i][0].isdigit()

    def is_date(i: int) -> bool:
        tok, s, e = toks[i]
        return _compact_date(tok) or any(a <= s and e <= b for a, b in spans)

    def gap(i: int, j: int) -> str:
        return stem[toks[i][2]:toks[j][1]]

    def is_word(i: int) -> bool:
        return 0 <= i < len(toks) and not toks[i][0].isdigit()

    def glued(i: int, j: int) -> bool:
        lo, hi = min(i, j), max(i, j)
        return gap(lo, hi) == ""

    sexes: "list[tuple[int, str]]" = []
    for i, (tok, _s, _e) in enumerate(toks):
        if tok.isdigit():
            continue
        touch = ((i > 0 and is_num(i - 1) and glued(i - 1, i))
                 or (is_num(i + 1) and glued(i, i + 1)))
        sex = _sex_of(tok, i > 0 and is_num(i - 1), touch)
        if sex is not None:
            sexes.append((i, sex))
    sex_at = {i for i, _x in sexes}

    # The number next to a sex letter: the one before it first. Not one
    # glued to a word on its other side ("Ch6_F", "Exp4_F"), unless that
    # word is a sex letter too ("M206F"); not a single digit set apart from
    # the letter ("206 F 3").
    for i, sex in sexes:
        for j in (i - 1, i + 1):
            if not is_num(j) or is_date(j):
                continue
            lo, hi = min(i, j), max(i, j)
            if not _GAP_RE.match(gap(lo, hi)):
                continue
            far = j - 1 if j < i else j + 1
            if is_word(far) and glued(far, j) and far not in sex_at:
                continue
            if len(toks[j][0]) < 2 and not glued(i, j):
                continue
            return toks[j][0], sex

    def has_unit(k: int) -> bool:
        # "100Hz", "10 min", "30 s": the next word is a unit, glued or after
        # a space (not after _ or -, which separate the parts of a name),
        # except a lower case m, minutes after any separator (as for the sex).
        n = k + 1
        if not is_word(n) or n in sex_at:
            return False
        word = toks[n][0]
        if word == "m":
            return bool(_GAP_RE.match(gap(k, n)))
        return word.casefold() in _UNITS and gap(k, n).strip(" ") == ""

    distinct = {s for _i, s in sexes}
    sex = distinct.pop() if len(distinct) == 1 else None
    nums = [k for k, (t, _s, _e) in enumerate(toks)
            if t.isdigit() and len(t) >= 2 and not is_date(k)]
    # Numbers without a unit first; one with a unit only when it is all
    # the name has ("206m.mat").
    plain = [k for k in nums if not has_unit(k)] or nums
    return (toks[plain[-1]][0] if plain else None), sex


_SEX_WORDS = {"m": "M", "male": "M", "mâle": "M", "♂": "M",
              "f": "F", "female": "F", "femelle": "F", "fem": "F", "♀": "F"}


def normalise_sex(text: object) -> str:
    """'M' or 'F' for the usual ways of writing a sex ('m', 'Male', 'femelle',
    '♀'...), '' for an empty value, anything else as typed (stripped)."""
    s = "" if text is None else str(text).strip()
    if s.lower() == "nan":
        return ""
    return _SEX_WORDS.get(s.casefold(), s)


# The Recording sheet's untouched Subject ID and the like ("Subject 2").
DEFAULT_SUBJECT = "subject_01"
_DEFAULT_SUBJECT_RE = re.compile(r"^subject[\s_-]*\d*$", re.IGNORECASE)


def is_default_subject(subject: object) -> bool:
    """True for the Recording sheet's untouched Subject ID ("subject_01") and the
    like ("Subject 2", "subject") -- a placeholder rather than an animal."""
    return bool(_DEFAULT_SUBJECT_RE.match(str(subject or "").strip()))
