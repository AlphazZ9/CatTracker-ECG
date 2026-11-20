# -*- coding: utf-8 -*-
"""
ecg.io.prism_template
---------------------
GraphPad Prism: one Column table per metric (one row per animal, one column
per segment: ready for a paired t / Wilcoxon / RM ANOVA and a before-after
graph), written to a new .pzfx or poured into a Prism *template*.

A template is a .pzfx the user saved once from Prism with their graphs and
analyses. Prism keeps graphs, analyses and layouts in one binary blob
(``<Template dt:dt="bin.base64">``) that refers to the data tables by ID; it
cannot be written from outside, but Prism recomputes and redraws everything
from the tables when the file opens. ``fill_template`` therefore replaces
only the rows and columns of the data tables whose titles match the tables
made here (a table of this protocol without data this time is emptied, never
left with older values) and copies every other byte of the template as it is.

Table titles (the same protocol gives the same titles in every file):

* all segments: the metric with its unit, as in the comparison table
  ("SDNN (ms)"), one column per segment named by its label;
* pairs (Δ % "Pairs" of a recording, "All pairs" of the group card): one
  table per pair and metric, "<metric> · <B label> vs <A label>" (with the
  tags, "(B vs A)", when the labels are not the tags: repeated labels),
  the reference segment's column first.

Rows (group tables) are titled by the subject, with the animal's sex in
brackets when the group table gives one ("1146 (F)", "206" when unknown); the
project notes list the animals by sex (``sex_notes``). The tests run on every
animal: the sex only describes the rows.

Nothing here may import from ecg.ui; only the standard library parses XML.
"""
from __future__ import annotations

import codecs
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union
from xml.sax.saxutils import escape, quoteattr

import numpy as np
import pandas as pd

from ecg.core.segment_plan import PAIRS, comparators_reference
from ecg.io.group_stats import (
    ID_COLUMNS, META_PREFIXES, _tag_columns, all_pairs_stats, metric_names, pair_title,
    segment_labels, segment_tags, sex_text, subject_text,
)

# The last template filled, remembered per user (read at call time, so a
# test can point it elsewhere).
PREFS_PATH: Path = Path.home() / ".ecg_prism_template.json"

# The project-info sheet this module writes (replaced on a refill, never another).
NOTES_TITLE = "ECG Analysis export"
PAIR_SEP = " · "
NOT_XML = ("{name} is not a Prism XML file. A .prism file (Prism 10 and later) is binary: "
           "save the template as .pzfx (File > Save As > Prism XML .pzfx) and choose that file.")
BAD_XML = ("{name} is not a valid Prism XML file (line {line}): open it in Prism and save it "
           "again as .pzfx (File > Save As > Prism XML .pzfx).")

LAYOUT_ALL = "all"          # one table per metric, every segment a column
LAYOUT_PAIRS = "pairs"      # one table per pair x metric


class PrismTemplateError(ValueError):
    """A template that cannot be filled (not Prism XML, the template itself
    chosen as the output...)."""


# A table: (title, frame) or (title, frame, x column name). The frame's
# index gives the row titles unless it is a plain 0..n-1 RangeIndex.
TableSpec = Union["tuple[str, pd.DataFrame]", "tuple[str, pd.DataFrame, Optional[str]]"]


# ── Builders ─────────────────────────────────────────────────────────────

def _clean(text: Any) -> str:
    return " ".join(str(text or "").split())


def _norm(text: Any) -> str:
    return _clean(text).casefold()


def _unique(names: "Sequence[str]", tags: "Sequence[str]") -> "list[str]":
    """*names* (labels), each one shared by several made '<name> (<tag>)'."""
    count: "dict[str, int]" = {}
    for n in names:
        count[_norm(n)] = count.get(_norm(n), 0) + 1
    out = [f"{n} ({t})" if count[_norm(n)] > 1 and _norm(n) != _norm(t) else n
           for n, t in zip(names, tags)]
    # Still the same (a label equal to another segment's tag): the tags.
    if len({_norm(n) for n in out}) < len(out):
        out = list(tags)
    return out


def _unique_titles(specs: "list[tuple]") -> "list[tuple]":
    seen: "dict[str, int]" = {}
    out = []
    for spec in specs:
        title = spec[0]
        k = seen.get(_norm(title), 0)
        seen[_norm(title)] = k + 1
        out.append(spec if not k else (f"{title} ({k + 1})",) + tuple(spec[1:]))
    return out


def _sex_title(name: str, sex: str) -> str:
    return f"{name} ({sex})" if sex else name


def _row_titles(df: "pd.DataFrame") -> "list[str]":
    """The subject of each row (the file without its extension when it has
    none) with its sex when known ("1146 (F)"); a subject on several rows
    gets its file, a repeat a number."""
    subj = [subject_text(v) for v in (df["subject"] if "subject" in df.columns
                                      else [""] * len(df))]
    sexes = [sex_text(v) for v in (df["sex"] if "sex" in df.columns else [""] * len(df))]
    files = [subject_text(v) for v in (df["file"] if "file" in df.columns else [""] * len(df))]
    stems = [os.path.splitext(f)[0] for f in files]
    base = [_sex_title(s or st, sx) if (s or st) else f"row {i + 1}"
            for i, (s, st, sx) in enumerate(zip(subj, stems, sexes))]
    count: "dict[str, int]" = {}
    for b in base:
        count[b] = count.get(b, 0) + 1
    titles = [f"{b} ({st})" if count[b] > 1 and st and st != b else b
              for b, st in zip(base, stems)]
    seen: "dict[str, int]" = {}
    out = []
    for t in titles:
        k = seen.get(t, 0)
        seen[t] = k + 1
        out.append(t if not k else f"{t} #{k + 1}")
    return out


def sex_notes(df: "Optional[pd.DataFrame]") -> "list[tuple[str, str]]":
    """(name, value) notes on the sex of the animals of a group table *df*
    ("F: 1146, 1150; M: 206; not given: 300"), none when it has no sex."""
    if df is None or "sex" not in df.columns or not len(df):
        return []
    groups: "dict[str, list[str]]" = {}
    for i, (_k, r) in enumerate(df.iterrows()):
        who = (subject_text(r.get("subject")) or
               os.path.splitext(subject_text(r.get("file")))[0] or f"row {i + 1}")
        sx = sex_text(r.get("sex")) or "not given"
        if who not in groups.setdefault(sx, []):
            groups[sx].append(who)
    if set(groups) == {"not given"}:
        return []
    order = sorted(groups, key=lambda k: (k == "not given", k))
    return [("Sex", "; ".join(f"{k}: {', '.join(groups[k])}" for k in order)),
            ("Row titles", "subject (sex), e.g. 1146 (F)")]


def _owner(col: str, tags: "Sequence[str]") -> "Optional[str]":
    return max((t for t in tags if col.endswith(f" {t}")), key=len, default=None)


def group_metric_names(df: "pd.DataFrame", tags: "Optional[Sequence[str]]" = None) -> "list[str]":
    """Every metric (label with unit) of a group table, in its column order
    (the comparison's order): the per-segment columns that are not times,
    quality, labels or indices."""
    all_tags = [c[len("start (s) "):] for c in map(str, df.columns) if c.startswith("start (s) ")]
    tags = list(tags) if tags is not None else all_tags
    names = []
    for c in map(str, df.columns):
        t = _owner(c, all_tags)
        if t is None or t not in tags:
            continue
        name = c[: -len(t) - 1]
        if name in ID_COLUMNS or name.startswith(META_PREFIXES) or name in names:
            continue
        names.append(name)
    return names


def _values(df: "pd.DataFrame", col: str) -> "np.ndarray":
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)


def group_tables(df: "pd.DataFrame", tags: "Optional[Sequence[str]]" = None
                 ) -> "list[tuple[str, pd.DataFrame]]":
    """All segments: per metric of the group table *df* (the rows to write),
    a table with one row per animal and one column per segment of *tags*
    (default: those holding a value), named by the segment's label."""
    tags = list(tags) if tags is not None else segment_tags(df)
    labels = [segment_labels(df, t)[0] if f"start (s) {t}" in df.columns else "" for t in tags]
    cols = _unique([lbl or t for lbl, t in zip(labels, tags)], tags)
    rows = _row_titles(df)
    all_tags = [c[len("start (s) "):] for c in map(str, df.columns) if c.startswith("start (s) ")]
    cols_of = {t: _tag_columns(df, t, all_tags) for t in tags}
    out = []
    for name in group_metric_names(df, tags):
        data = {col: _values(df, cols_of[t].get(name, f"{name} {t}")) for col, t in zip(cols, tags)}
        out.append((name, pd.DataFrame(data, index=pd.Index(rows, name="subject"))))
    return _unique_titles(out)


def pair_tables(res: Any) -> "list[tuple[str, pd.DataFrame]]":
    """All pairs (group_stats.all_pairs_stats): per pair and per metric of
    the table, the rows the pair used (the rows left out are not written),
    its two segments as columns, the reference first."""
    out = []
    for pr in res.pairs:
        used = pr.used
        # A pair without rows still gives its tables (no row): a template
        # table of that pair is emptied, never left with older data.
        if used is None:
            continue
        cols = _unique([pr.name_a, pr.name_b], [pr.a, pr.b])
        rows = _row_titles(used)
        for name in metric_names(used, pr.a, pr.b):
            data = {cols[0]: _values(used, f"{name} {pr.a}"),
                    cols[1]: _values(used, f"{name} {pr.b}")}
            out.append((f"{name}{PAIR_SEP}{pr.title}",
                        pd.DataFrame(data, index=pd.Index(rows, name="subject"))))
    return _unique_titles(out)


def left_out_notes(checks: "Sequence[Any]", keep: "Sequence[bool]", pair: str = ""
                   ) -> "list[tuple[str, str]]":
    """(name, value) notes on the rows to check of a group analysis
    (group_stats.RowCheck and whether each row was kept): which rows were
    not written and why, and the rows to check that were kept."""
    sfx = f" — {pair}" if pair else ""
    out = []
    for c, k in zip(checks, keep):
        if c.ok:
            continue
        who = c.subject or c.file or "—"
        missing = any(p.startswith("no segment") for p in c.problems)
        what = ("Kept, to check" if k else
                "Not written (segment missing)" if missing else "Left out (row to check)")
        out.append((f"{what}{sfx}", f"{who}: " + "; ".join(c.problems)))
    return out


def _result_subject(res: Any) -> str:
    return (_clean(getattr(res, "subject", "")) or
            os.path.splitext(os.path.basename(getattr(res, "filepath", "") or ""))[0] or
            "Recording")


def is_pairs(res: Any) -> bool:
    """The comparison is in Pairs mode (2 vs 1, 4 vs 3, 6 vs 5)."""
    try:
        return comparators_reference(res.comps) == PAIRS and len(res.segs) > 2
    except Exception:
        return False


def result_tables(res: Any, layout: str = "auto") -> "list[tuple[str, pd.DataFrame]]":
    """The per-metric tables of one comparison (segment_export.ComparisonResult),
    one row (the recording, named by its subject or file). *layout*: "all"
    (a column per segment), "pairs" (a table per pair x metric) or "auto"
    (pairs when the comparison is in Pairs mode)."""
    if layout == "auto":
        layout = LAYOUT_PAIRS if is_pairs(res) else LAYOUT_ALL
    tags = [_clean(t) for t in res.tags]
    labels = [_clean(s.get("label", "")) for s in res.segs]
    names = [lbl or t for lbl, t in zip(labels, tags)]
    index = pd.Index([_result_subject(res)], name="subject")

    def val(seg: dict, key: str) -> float:
        try:
            v = float(seg.get(key))
        except (TypeError, ValueError):
            return math.nan
        return v if math.isfinite(v) else math.nan

    metrics = [(key, f"{label} ({unit})" if unit else label) for key, label, unit in res.metrics]
    out = []
    if layout == LAYOUT_PAIRS:
        for j, k in res.pairs:
            cols = _unique([names[j], names[k]], [tags[j], tags[k]])
            ptitle = pair_title(tags[j], tags[k], labels[j], labels[k])
            for key, title in metrics:
                out.append((f"{title}{PAIR_SEP}{ptitle}", pd.DataFrame(
                    {cols[0]: [val(res.segs[j], key)], cols[1]: [val(res.segs[k], key)]},
                    index=index)))
    else:
        cols = _unique(names, tags)
        for key, title in metrics:
            out.append((title, pd.DataFrame({c: [val(s, key)] for c, s in zip(cols, res.segs)},
                                            index=index)))
    return _unique_titles(out)


def recording_tables(res: Any, layout: str = "auto") -> "list[tuple]":
    """Everything the Prism export of one comparison holds: the comparison
    table, the per-metric tables (result_tables), then the HR block means,
    the RR series and the Poincaré pairs (segment_export.prism_tables)."""
    from ecg.io.segment_export import prism_tables
    old = [(name, df, (x or None)) for name, df, x, _rows in prism_tables(res)]
    return _unique_titles(old[:1] + list(result_tables(res, layout)) + old[1:])


def metric_tables(source: Any, layout: str = "auto", **kw: Any) -> "list[tuple]":
    """The per-metric tables of *source*: a ComparisonResult (result_tables),
    an all-pairs result (pair_tables) or a group table DataFrame
    (group_tables; layout "pairs" runs all_pairs_stats on it, **kw passed)."""
    if isinstance(source, pd.DataFrame):
        if layout == LAYOUT_PAIRS:
            return pair_tables(all_pairs_stats(source, **kw))
        return group_tables(source, kw.get("tags"))
    if hasattr(source, "pairs") and hasattr(source, "holm_across"):
        return pair_tables(source)
    return result_tables(source, layout)


# ── XML text ─────────────────────────────────────────────────────────────

def _num_text(v: Any) -> "Optional[str]":
    """A cell as Prism reads it: a plain decimal number (no exponent), the
    text of a non-number, None when empty / NaN."""
    if v is None:
        return None
    try:
        if pd.isna(v):                       # NaN, pd.NA, NaT
            return None
    except (TypeError, ValueError):          # not a scalar
        pass
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        s = str(v).strip()
        return s or None
    if not math.isfinite(f):
        return None
    if f.is_integer() and abs(f) < 1e15:
        return str(int(f))
    # The shortest text that reads back as the same number, no exponent; a
    # float32 at its own precision (0.1, not 0.1000000015).
    if isinstance(v, np.floating) and np.dtype(type(v)).itemsize < 8:
        return np.format_float_positional(v, unique=True, trim="-")
    return np.format_float_positional(f, unique=True, trim="-")


def _d(v: Any, text: bool = False) -> str:
    s = (None if v is None or (isinstance(v, float) and not math.isfinite(v)) else
         _clean(v) or None) if text else _num_text(v)
    return "<d/>" if s is None else f"<d>{escape(s)}</d>"


def _subcolumn(values: "Iterable[Any]", nl: str, text: bool = False) -> str:
    return nl.join(["<Subcolumn>", *(_d(v, text) for v in values), "</Subcolumn>"])


def _empty_subcolumn(n: int, nl: str) -> str:
    return _subcolumn([None] * n, nl)


def _attrs(**kw: Any) -> str:
    return "".join(f" {k}={quoteattr(str(v))}" for k, v in kw.items())


@dataclass
class _Table:
    title: str
    frame: "pd.DataFrame"
    x: "Optional[str]" = None

    @property
    def row_titles(self) -> "Optional[list[str]]":
        idx = self.frame.index
        if isinstance(idx, pd.RangeIndex) and idx.start == 0 and idx.step == 1:
            return None
        return [str(v) for v in idx]

    @property
    def y_columns(self) -> "list[str]":
        return [str(c) for c in self.frame.columns if c != self.x]


def _tables(specs: "Iterable[Any]") -> "list[_Table]":
    out = []
    for spec in specs:
        if isinstance(spec, _Table):
            out.append(spec)
            continue
        title, frame = spec[0], spec[1].copy()
        frame.columns = [str(c) for c in frame.columns]
        x = spec[2] if len(spec) > 2 and spec[2] not in (0, "", None) else None
        x = str(x) if x is not None else None
        out.append(_Table(str(title), frame, x if x in frame.columns else None))
    return out


def _width(texts: "Iterable[str]", lo: int, hi: int = 240) -> int:
    longest = max((len(str(t)) for t in texts), default=0)
    return int(min(hi, max(lo, 7 * longest + 16)))


def _new_table_xml(t: _Table, table_id: str, nl: str) -> str:
    """A table of the export, written as Prism writes one."""
    xy = t.x is not None
    head = (_attrs(ID=table_id, XFormat="numbers", YFormat="replicates", Replicates=1,
                   TableType="XY", EVFormat="AsteriskAfterNumber") if xy else
            _attrs(ID=table_id, XFormat="none", YFormat="replicates", Replicates=1,
                   TableType="OneWay", EVFormat="AsteriskAfterNumber"))
    parts = [f"<Table{head}>", f"<Title>{escape(t.title)}</Title>"]
    rows = t.row_titles
    if rows is not None:
        parts += [f"<RowTitlesColumn{_attrs(Width=_width(rows, 39))}>",
                  _subcolumn(rows, nl, text=True), "</RowTitlesColumn>"]
    if xy:
        parts += [f"<XColumn{_attrs(Width=_width([t.x], 89), Decimals=4, Subcolumns=1)}>",
                  f"<Title>{escape(str(t.x))}</Title>",
                  _subcolumn(t.frame[t.x].tolist(), nl), "</XColumn>"]
    for c in t.y_columns:
        parts += [f"<YColumn{_attrs(Width=_width([c], 89), Decimals=4, Subcolumns=1)}>",
                  f"<Title>{escape(c)}</Title>",
                  _subcolumn(t.frame[c].tolist(), nl), "</YColumn>"]
    parts.append("</Table>")
    return nl.join(parts)


def _info_xml(rows: "Sequence[tuple[Any, Any]]", info_id: str, nl: str) -> str:
    parts = [f"<Info{_attrs(ID=info_id)}>", f"<Title>{escape(NOTES_TITLE)}</Title>", "<Notes/>"]
    for name, value in rows:
        v = "" if value is None or (isinstance(value, float) and not math.isfinite(value)) \
            else str(value)
        parts.append(f"<Constant><Name>{escape(str(name))}</Name>"
                     + (f"<Value>{escape(v)}</Value>" if v else "<Value/>") + "</Constant>")
    parts.append("</Info>")
    return nl.join(parts)


# ── Reading a template ───────────────────────────────────────────────────

_DATA_TAGS = ("Table1024", "HugeTable", "Table")
_DATA_RE = re.compile(r"<(Table1024|HugeTable|Table)(?=[\s/>])")


def _decode(raw: bytes, name: str) -> "tuple[str, str]":
    if raw.startswith(codecs.BOM_UTF8):
        enc = "utf-8-sig"
    elif raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        enc = "utf-16"
    else:
        m = re.match(rb"\s*<\?xml[^>]*?encoding\s*=\s*[\"']([A-Za-z0-9._-]+)", raw)
        enc = m.group(1).decode("ascii") if m else "utf-8"
    try:
        text = raw.decode(enc)
    except (UnicodeDecodeError, LookupError):
        raise PrismTemplateError(NOT_XML.format(name=name)) from None
    head = text.lstrip()[:4096]
    if not head.startswith("<") or "<GraphPadPrismFile" not in head:
        raise PrismTemplateError(NOT_XML.format(name=name))
    return text, enc


def _end_of(text: str, tag: str, start: int) -> int:
    """The end of the element *tag* whose start tag begins at *start*."""
    close = text.find(">", start)
    if close < 0:
        raise PrismTemplateError(f"<{tag}> is not closed")
    if text[close - 1] == "/":
        return close + 1
    end = text.find(f"</{tag}>", close)
    if end < 0:
        raise PrismTemplateError(f"<{tag}> is not closed")
    return end + len(tag) + 3


def _comment_spans(text: str) -> "list[tuple[int, int]]":
    return [(m.start(), m.end()) for m in re.finditer(r"<!--.*?-->|<!\[CDATA\[.*?\]\]>",
                                                      text, re.S)]


def _skip_spans(text: str) -> "list[tuple[int, int]]":
    """Comments, CDATA and the binary Template blob: never searched."""
    spans = _comment_spans(text)
    for m in re.finditer(r"<Template(?=[\s/>])", text):
        if not any(a <= m.start() < b for a, b in spans):
            spans.append((m.start(), _end_of(text, "Template", m.start())))
    return spans


def _elements(text: str, pattern: "re.Pattern[str]", skip: "list[tuple[int, int]]",
              inside: "Optional[list[tuple[int, int]]]" = None) -> "list[tuple[str, int, int]]":
    """(tag, start, end) of each element matching *pattern* outside *skip*
    (and outside the *inside* spans, e.g. the data tables)."""
    out = []
    for m in pattern.finditer(text):
        p = m.start()
        if any(a <= p < b for a, b in skip) or (inside and any(a <= p < b for a, b in inside)):
            continue
        out.append((m.group(1), p, _end_of(text, m.group(1), p)))
    return out


def _title_of(xml: str) -> str:
    try:
        el = ET.fromstring(xml)
    except ET.ParseError:
        return ""
    t = el.find("Title")
    return _clean("".join(t.itertext())) if t is not None else ""


def _child_spans(sub: str, tag: str) -> "list[tuple[int, int]]":
    out, pos = [], 0
    pat = re.compile(rf"<{tag}(?=[\s/>])")
    while True:
        m = pat.search(sub, pos)
        if not m:
            return out
        end = _end_of(sub, tag, m.start())
        out.append((m.start(), end))
        pos = end


def _open_tag(el: str, tag: str) -> str:
    """The start tag of the element text *el*, made a non-empty one."""
    head = el[: el.find(">") + 1]
    return head[:-2].rstrip() + ">" if head.endswith("/>") else head


def _column_parts(el: str, tag: str) -> "tuple[str, Optional[str], str, int]":
    """(start tag, raw <Title> element or None, its text, subcolumns) of a
    YColumn / XColumn element text."""
    head = _open_tag(el, tag)
    inner = "" if el[: el.find(">") + 1].endswith("/>") else el[el.find(">") + 1: el.rfind("<")]
    m = re.match(r"\s*(<Title(?:\s[^>]*)?(?:/>|>.*?</Title>))", inner, re.S)
    raw = m.group(1) if m else None
    title = _title_of(f"<x>{raw}</x>") if raw else ""
    n_sub = len(re.findall(r"<Subcolumn(?=[\s/>])", inner))
    am = re.search(r'\sSubcolumns\s*=\s*"(\d+)"', head)
    n = max(1, n_sub, int(am.group(1)) if am else 1)
    return head, raw, title, n


def _title_xml(raw: "Optional[str]", kept: str, new: str) -> "Optional[str]":
    """The template's own <Title> when it has text, else one of *new*."""
    if raw is not None and _clean(kept):
        return raw
    return f"<Title>{escape(new)}</Title>" if _clean(new) else raw


def _column_xml(head: str, title_xml: "Optional[str]", values: "Optional[list]",
                n_sub: int, n_rows: int, tag: str, nl: str) -> str:
    parts = [head] + ([title_xml] if title_xml else [])
    first = _subcolumn(values, nl) if values is not None else _empty_subcolumn(n_rows, nl)
    parts.append(first)
    parts += [_empty_subcolumn(n_rows, nl) for _ in range(n_sub - 1)]
    parts.append(f"</{tag}>")
    return nl.join(parts)


@dataclass
class FillReport:
    """What fill_template did, for the status line and the project notes."""
    matched: "list[tuple[str, str]]" = field(default_factory=list)    # (title, table ID)
    appended: "list[tuple[str, str]]" = field(default_factory=list)   # (title, new ID)
    untouched: "list[tuple[str, str]]" = field(default_factory=list)  # (table ID, title)
    cleared: "list[tuple[str, str]]" = field(default_factory=list)    # (table ID, title)
    notes: "list[str]" = field(default_factory=list)
    out_path: str = ""

    @property
    def warn(self) -> bool:
        """Template tables emptied or left as they were: worth a look."""
        return bool(self.cleared or self.untouched)

    @staticmethod
    def _names(items: "list[tuple[str, str]]", k: int = 3) -> str:
        names = [t or i for i, t in items]
        return ", ".join(names[:k]) + (", …" if len(names) > k else "")

    def summary(self) -> str:
        parts = [f"{len(self.matched)} table{'s' if len(self.matched) != 1 else ''} filled"]
        if self.appended:
            parts.append(f"{len(self.appended)} added (not in the template: no graph)")
        if self.cleared:
            parts.append(f"{len(self.cleared)} template table"
                         f"{'s' if len(self.cleared) != 1 else ''} emptied, no data this time ("
                         + self._names(self.cleared) + ")")
        if self.untouched:
            parts.append(f"{len(self.untouched)} template table"
                         f"{'s' if len(self.untouched) != 1 else ''} not filled, left as "
                         f"{'they were' if len(self.untouched) != 1 else 'it was'} ("
                         + self._names(self.untouched) + ")")
        return "  ·  ".join(parts)

    def rows(self) -> "list[tuple[str, str]]":
        """(name, value) rows for the project notes."""
        out = [("Template tables filled", ", ".join(t for t, _i in self.matched) or "none")]
        if self.appended:
            out.append(("Tables added (not in the template)",
                        ", ".join(t for t, _i in self.appended)))
        if self.cleared:
            out.append(("Template tables emptied (no data this time)",
                        ", ".join(t or i for i, t in self.cleared)))
        if self.untouched:
            out.append(("Template tables not filled (left as they were)",
                        ", ".join(t or i for i, t in self.untouched)))
        for n in self.notes:
            out.append(("Template note", n))
        return out


def _apply(sub: str, edits: "list[tuple[int, int, str]]") -> str:
    """*edits* (start, end, text) applied from the end; at one place a
    replacement first, then the insertions before it in the order given."""
    order = [(a, b, k, s) for k, (a, b, s) in enumerate(edits)]
    for a, b, _k, s in sorted(order, key=lambda e: (e[0], e[1], e[2]), reverse=True):
        sub = sub[:a] + s + sub[b:]
    return sub


def _fill_table(sub: str, t: _Table, nl: str, report: FillReport, ttitle: str) -> str:
    """The template table element text *sub* with the rows and columns of *t*."""
    if sub[: sub.find(">") + 1].endswith("/>"):
        tag = re.match(r"<(\w+)", sub).group(1)
        sub = _open_tag(sub, tag) + nl + f"<Title>{escape(t.title)}</Title>" + nl + f"</{tag}>"
    tag = re.match(r"<(\w+)", sub).group(1)
    rows = t.row_titles
    n_rows = len(t.frame)
    edits: "list[tuple[int, int, str]]" = []
    rt = _child_spans(sub, "RowTitlesColumn")
    xc = _child_spans(sub, "XColumn")
    xa = _child_spans(sub, "XAdvancedColumn")
    ys = _child_spans(sub, "YColumn")
    fn = _child_spans(sub, "FloatingNote")
    body_end = sub.rfind(f"</{tag}>")
    # Row titles.
    if rt:
        a, b = rt[0]
        edits.append((a, b, nl.join([_open_tag(sub[a:b], "RowTitlesColumn"),
                                     _subcolumn(rows if rows is not None else [], nl, text=True),
                                     "</RowTitlesColumn>"])))
    elif rows is not None:
        firsts = [s for s, _e in (xa + xc + ys + fn)]
        at = min(firsts) if firsts else body_end
        edits.append((at, at, nl.join([f"<RowTitlesColumn{_attrs(Width=_width(rows, 39))}>",
                                       _subcolumn(rows, nl, text=True),
                                       "</RowTitlesColumn>"]) + nl))
    # X column, and the copy Prism 8+ keeps in XAdvancedColumn: both get the
    # new X values (else a graph could pair old X with new Y).
    xs = [(span, "XColumn") for span in xc[:1]] + [(span, "XAdvancedColumn") for span in xa[:1]]
    if t.x is not None and xs:
        for (a, b), xtag in xs:
            head, raw, title, n_sub = _column_parts(sub[a:b], xtag)
            edits.append((a, b, _column_xml(head, _title_xml(raw, title, str(t.x)),
                                            t.frame[t.x].tolist(), n_sub, n_rows, xtag, nl)))
    elif t.x is not None:
        report.notes.append(f"{ttitle}: the template table has no X column, X values not written")
    elif xs:
        # A Column table made XY in the template: its old X values would no
        # longer match the rows, so they are emptied.
        for (a, b), xtag in xs:
            head, raw, _title, n_sub = _column_parts(sub[a:b], xtag)
            edits.append((a, b, _column_xml(head, raw, None, n_sub, n_rows, xtag, nl)))
        report.notes.append(f"{ttitle}: the template table is XY, its X column was emptied "
                            "(no X values)")
    # Y columns: by title (case and spacing do not count), else the free
    # template columns in order.
    produced = t.y_columns
    tmpl = [_column_parts(sub[a:b], "YColumn") for a, b in ys]
    ttitles = [_norm(p[2]) for p in tmpl]
    assign: "dict[int, int]" = {}            # produced column -> template column
    for i, c in enumerate(produced):
        j = next((j for j, tt in enumerate(ttitles) if tt and tt == _norm(c)
                  and j not in assign.values()), None)
        if j is not None:
            assign[i] = j
    free = [j for j in range(len(tmpl)) if j not in assign.values()]
    for i, c in enumerate(produced):
        if i in assign or not free:
            continue
        j = free.pop(0)
        assign[i] = j
        if ttitles[j]:
            report.notes.append(f"{ttitle}: '{c}' written in column '{tmpl[j][2]}' "
                                "(by position: no column of that title)")
    by_tmpl = {j: i for i, j in assign.items()}
    for j, ((a, b), (head, raw, title, n_sub)) in enumerate(zip(ys, tmpl)):
        i = by_tmpl.get(j)
        values = t.frame[produced[i]].tolist() if i is not None else None
        if i is None:
            report.notes.append(f"{ttitle}: column '{title or j + 1}' has no data, left empty")
        new_title = produced[i] if i is not None else ""
        edits.append((a, b, _column_xml(head, _title_xml(raw, title, new_title),
                                        values, n_sub, n_rows, "YColumn", nl)))
    extra = [i for i in range(len(produced)) if i not in assign]
    if extra:
        if ys:
            last = sub[ys[-1][0]: ys[-1][1]]
            head0 = _open_tag(last, "YColumn")
            head0 = re.sub(r'\sSubcolumns\s*=\s*"\d+"', ' Subcolumns="1"', head0)
            at, lead = ys[-1][1], True
        else:
            # After the row titles and X columns (the schema's order), else
            # before the floating notes or at the end.
            head0 = f"<YColumn{_attrs(Width=89, Decimals=4, Subcolumns=1)}>"
            ends = [e for _s, e in (rt + xc + xa)]
            if ends:
                at, lead = max(ends), True
            else:
                at, lead = min([s for s, _e in fn] or [body_end]), False
        new = []
        for i in extra:
            new.append(_column_xml(head0, _title_xml(None, "", produced[i]),
                                   t.frame[produced[i]].tolist(), 1, n_rows, "YColumn", nl))
            report.notes.append(f"{ttitle}: column '{produced[i]}' added")
        text = nl.join(new)
        edits.append((at, at, nl + text if lead else text + nl))
    return _apply(sub, edits)


def _clear_table(sub: str, nl: str) -> str:
    """The template table element text *sub* with no row: its columns,
    titles, attributes and ID kept, every cell gone."""
    if sub[: sub.find(">") + 1].endswith("/>"):
        return sub
    edits: "list[tuple[int, int, str]]" = []
    for a, b in _child_spans(sub, "RowTitlesColumn")[:1]:
        edits.append((a, b, nl.join([_open_tag(sub[a:b], "RowTitlesColumn"),
                                     _empty_subcolumn(0, nl), "</RowTitlesColumn>"])))
    for ctag in ("XColumn", "XAdvancedColumn", "YColumn"):
        for a, b in _child_spans(sub, ctag):
            head, raw, _title, n_sub = _column_parts(sub[a:b], ctag)
            edits.append((a, b, _column_xml(head, raw, None, n_sub, 0, ctag, nl)))
    return _apply(sub, edits)


# Titles of a recording's extra tables (segment_export.prism_tables).
_EXTRA_TITLES = ("comparison", "hr block means (bpm)")
_EXTRA_XY_PREFIXES = ("rr ", "poincaré ")


def _metric_key(title: str) -> str:
    """The metric of a table title ('SDNN (ms)' of 'SDNN (ms) · Post vs Pre')."""
    return _norm(_clean(title).split(PAIR_SEP.strip(), 1)[0])


def _is_ours(title: str, head: str, keys: "set[str]") -> bool:
    """A template table this app makes: a metric of the tables written (all
    segments or a pair), or one of a recording's extra tables."""
    n = _norm(title)
    if not n:
        return False
    if _metric_key(title) in keys or n in _EXTRA_TITLES:
        return True
    xy = re.search(r'\bTableType\s*=\s*"XY"', head) is not None
    return xy and n.startswith(_EXTRA_XY_PREFIXES)


def _ids(text: str, skip: "list[tuple[int, int]]") -> "set[str]":
    return {m.group(1) for m in re.finditer(r'\bID\s*=\s*"([^"]*)"', text)
            if not any(a <= m.start() < b for a, b in skip)}


def _new_id(prefix: str, used: "set[str]") -> str:
    nums = [int(m.group(1)) for u in used for m in [re.fullmatch(rf"{prefix}(\d+)", u)] if m]
    k = max(nums) + 1 if nums else 0
    while f"{prefix}{k}" in used:
        k += 1
    used.add(f"{prefix}{k}")
    return f"{prefix}{k}"


def _fill_text(text: str, tables: "list[_Table]", notes: "Optional[Sequence[tuple[Any, Any]]]",
               report: FillReport, with_report: bool = True) -> str:
    # The template's line ends (the blob's own may differ: not looked at).
    nl = "\r\n" if "\r\n" in text[:_first_insert(text)] else "\n"
    skip = _skip_spans(text)
    data = _elements(text, _DATA_RE, skip)
    spans = [(a, b) for _t, a, b in data]
    edits: "list[tuple[int, int, int, str]]" = []          # (start, end, order, text)
    seq = [0]

    def edit(a: int, b: int, s: str) -> None:
        seq[0] += 1
        edits.append((a, b, seq[0], s))

    # Match the tables: exact title, then case / spacing insensitive.
    ttitles = [_title_of(text[a:b]) for _t, a, b in data]
    tids = [(re.search(r'\bID\s*=\s*"([^"]*)"', text[a: text.find(">", a)]) or [None, ""])[1]
            for _t, a, b in data]
    taken: "dict[int, int]" = {}                        # produced -> template index
    for i, t in enumerate(tables):
        j = next((j for j, tt in enumerate(ttitles) if tt == _clean(t.title)
                  and j not in taken.values()), None)
        if j is not None:
            taken[i] = j
    for i, t in enumerate(tables):
        if i in taken:
            continue
        j = next((j for j, tt in enumerate(ttitles) if _norm(tt) == _norm(t.title)
                  and j not in taken.values()), None)
        if j is not None:
            taken[i] = j
    for i, j in taken.items():
        _tag, a, b = data[j]
        edit(a, b, _fill_table(text[a:b], tables[i], nl, report, tables[i].title))
        report.matched.append((tables[i].title, tids[j]))
    filled = set(taken.values())
    # A template table of this protocol with no data this time (a pair
    # without animals, a recording's table filled from a group): emptied,
    # never left with the previous experiment's values. Other tables (the
    # user's own) are left as they were.
    keys = {_metric_key(t.title) for t in tables}
    for j in range(len(data)):
        if j in filled:
            continue
        _tag, a, b = data[j]
        if _is_ours(ttitles[j], text[a: text.find(">", a) + 1], keys):
            edit(a, b, _clear_table(text[a:b], nl))
            report.cleared.append((tids[j], ttitles[j]))
        else:
            report.untouched.append((tids[j], ttitles[j]))
    used = _ids(text, skip)
    # Tables not in the template: appended after the last one.
    new = [(t, _new_id("Table", used)) for i, t in enumerate(tables) if i not in taken]
    if new:
        if data:
            at = data[-1][2]
            block = "".join(nl + _new_table_xml(t, tid, nl) for t, tid in new)
        else:
            at = _first_insert(text)
            block = "".join(_new_table_xml(t, tid, nl) + nl for t, tid in new)
        refs = [f"<Ref{_attrs(ID=tid)}/>" for _t, tid in new]
        seqs = _elements(text, re.compile(r"<(TableSequence)(?=[\s/>])"), skip, spans)
        if seqs:
            _s, a, b = seqs[0]
            el = text[a:b]
            if not re.search(r"<Ref(?=[\s/>])", el):
                refs[0] = refs[0][:-2] + ' Selected="1"/>'
            if el.rstrip().endswith("/>"):
                edit(a, b, nl.join(["<TableSequence>", *refs, "</TableSequence>"]))
            else:
                p = a + el.rfind("</TableSequence>")
                edit(p, p, nl.join(refs) + nl)
        else:
            # No sequence yet: one for every table, the template's first.
            refs = [f"<Ref{_attrs(ID=tid)}/>" for tid in tids if tid] + refs
            refs[0] = refs[0][:-2] + ' Selected="1"/>'
            seq_xml = nl.join(["<TableSequence>", *refs, "</TableSequence>"]) + nl
            p = data[0][1] if data else at
            edit(p, p, seq_xml)
        edit(at, at, block)
        for t, tid in new:
            report.appended.append((t.title, tid))
    # The project notes: this module's own Info sheet, replaced or added.
    if notes is not None:
        rows = list(notes) + (report.rows() if with_report else [])
        infos = _elements(text, re.compile(r"<(Info)(?=[\s/>])"), skip, spans)
        mine = [(a, b) for _t, a, b in infos if _title_of(text[a:b]) == NOTES_TITLE]
        if mine:
            a, b = mine[0]
            iid = re.search(r'\bID\s*=\s*"([^"]*)"', text[a:b]).group(1)
            edit(a, b, _info_xml(rows, iid, nl))
        else:
            iid = _new_id("Info", used)
            info = _info_xml(rows, iid, nl)
            ref = f"<Ref{_attrs(ID=iid)}/>"
            iseq = _elements(text, re.compile(r"<(InfoSequence)(?=[\s/>])"), skip, spans)
            tseq = _elements(text, re.compile(r"<(TableSequence)(?=[\s/>])"), skip, spans)
            first_data = tseq[0][1] if tseq else data[0][1] if data else _first_insert(text)
            if infos:
                p = infos[-1][2]
                edit(p, p, nl + info)
            if iseq:
                _s, a, b = iseq[0]
                el = text[a:b]
                if not re.search(r"<Ref(?=[\s/>])", el):
                    ref = ref[:-2] + ' Selected="1"/>'
                if el.rstrip().endswith("/>"):
                    edit(a, b, nl.join(["<InfoSequence>", ref, "</InfoSequence>"]))
                else:
                    p = a + el.rfind("</InfoSequence>")
                    edit(p, p, ref + nl)
                if not infos:
                    p = iseq[0][2]
                    edit(p, p, nl + info)
            else:
                ref = ref[:-2] + ' Selected="1"/>'
                block = nl.join(["<InfoSequence>", ref, "</InfoSequence>"]) + nl
                if not infos:
                    block += info + nl
                edit(first_data, first_data, block)
    # From the end; at one place a replacement first, then the insertions
    # before it (applied last to first, so that they stay in their order).
    for a, b, _o, s in sorted(edits, key=lambda e: (e[0], e[1], e[2]), reverse=True):
        text = text[:a] + s + text[b:]
    return text


def _first_insert(text: str) -> int:
    """Where the root elements after the data tables start (scripts, the
    template blob) or the root ends: a new table goes before."""
    comments = _comment_spans(text)
    for m in re.finditer(r"<(?:Script|TemplateDescription|Template)(?=[\s/>])|"
                         r"</GraphPadPrismFile>", text):
        if not any(a <= m.start() < b for a, b in comments):
            return m.start()
    return len(text)


def _same_file(a: str, b: str) -> bool:
    try:
        return os.path.exists(a) and os.path.exists(b) and os.path.samefile(a, b)
    except OSError:
        return False


def fill_template(template_path: str, tables: "Iterable[Any]", out_path: str,
                  notes: "Optional[Sequence[tuple[Any, Any]]]" = None) -> FillReport:
    """Write *out_path*: the Prism template *template_path* with the rows and
    columns of each of its data tables whose title is one of *tables*
    ((title, frame[, x column]); exact title, else case and spacing
    insensitive) replaced by that table's, every other byte of the template
    kept (the graphs and analyses blob, the info sheets, the table IDs).
    Columns are matched by title, else by position; template columns without
    data are emptied (kept: graphs refer to them), extra ones appended. A
    table not in the template is appended (no graph). A template table
    without data that this app makes (a metric of *tables*, a pair of one, a
    recording's comparison / block / RR / Poincaré table) is emptied, never
    left with older values; any other is left as it was. *notes* (name,
    value) rows go to this module's own info sheet (NOTES_TITLE), with the
    report. A template that is not well-formed XML is refused, and nothing
    is written unless the result parses."""
    name = os.path.basename(template_path)
    if os.path.abspath(os.path.normcase(template_path)) == \
            os.path.abspath(os.path.normcase(out_path)) or _same_file(template_path, out_path):
        raise PrismTemplateError("Choose another file name: the template itself is never "
                                 "overwritten.")
    with open(template_path, "rb") as fh:
        raw = fh.read()
    if raw[:2] == b"PK" or template_path.lower().endswith(".prism"):
        raise PrismTemplateError(NOT_XML.format(name=name))
    text, enc = _decode(raw, name)
    try:
        ET.fromstring(raw)          # the blob is base64 text: it parses too
    except ET.ParseError as exc:
        raise PrismTemplateError(BAD_XML.format(name=name, line=exc.position[0])) from None
    report = FillReport(out_path=str(out_path))
    try:
        out = _fill_text(text, _tables(tables), notes, report)
    except PrismTemplateError as exc:        # an element not closed
        raise PrismTemplateError(BAD_XML.format(name=name, line="?") + f" ({exc})") from None
    data = out.encode(enc, errors="xmlcharrefreplace")
    try:
        ET.fromstring(data)
    except ET.ParseError as exc:
        raise PrismTemplateError(
            f"{name}: the filled file would not be valid XML (line {exc.position[0]}); "
            "nothing was written. Please report this template.") from None
    with open(out_path, "wb") as fh:
        fh.write(data)
    return report


def write_metric_pzfx(tables: "Iterable[Any]", path: str,
                      notes: "Optional[Sequence[tuple[Any, Any]]]" = None) -> int:
    """A new .pzfx with *tables* ((title, frame[, x column]): a Column table
    each, or XY with an x column; the frame's index as row titles) and
    *notes* in the project info sheet. Returns the number of tables."""
    nl = "\n"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    skeleton = nl.join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<GraphPadPrismFile PrismXMLVersion="5.00">',
        "<Created>",
        f'<OriginalVersion CreatedByProgram="GraphPad Prism" CreatedByVersion="6.0f.254" '
        f'Login="" DateTime="{now}"/>',
        "</Created>",
        "<InfoSequence>", "</InfoSequence>",
        "<TableSequence>", "</TableSequence>",
        "</GraphPadPrismFile>", ""])
    specs = _tables(tables)
    report = FillReport(out_path=str(path))
    # A fresh file: the report's lines (every table "added") say nothing.
    text = _fill_text(skeleton, specs, list(notes) if notes is not None else [], report,
                      with_report=False)
    with open(path, "wb") as fh:
        fh.write(text.encode("utf-8"))
    return len(specs)


# ── Reading back (tests, the live check) ─────────────────────────────────

def read_tables(path: str) -> "dict[str, dict]":
    """{title: {"id", "rows", "columns": {title: values}, "x"}} of each data
    table of a .pzfx (values as text, None for an empty cell)."""
    root = ET.parse(path).getroot()

    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    def sub_values(col: "ET.Element") -> "list[list[Optional[str]]]":
        out = []
        for sc in col:
            if local(sc.tag) == "Subcolumn":
                out.append([("".join(d.itertext()).strip() or None) for d in sc
                            if local(d.tag) == "d"])
        return out

    tables: "dict[str, dict]" = {}
    for el in root:
        if local(el.tag) not in _DATA_TAGS:
            continue
        info: "dict[str, Any]" = {"id": el.get("ID"), "rows": None, "columns": {}, "x": None}
        title = ""
        for ch in el:
            tag = local(ch.tag)
            if tag == "Title":
                title = _clean("".join(ch.itertext()))
            elif tag == "RowTitlesColumn":
                subs = sub_values(ch)
                info["rows"] = subs[0] if subs else []
            elif tag in ("XColumn", "YColumn"):
                t = next((c for c in ch if local(c.tag) == "Title"), None)
                ct = _clean("".join(t.itertext())) if t is not None else ""
                subs = sub_values(ch)
                if tag == "XColumn":
                    info["x"] = (ct, subs[0] if subs else [])
                else:
                    info["columns"][ct or f"#{len(info['columns']) + 1}"] = subs[0] if subs else []
        tables[title] = info
    return tables


# ── The last template, remembered ────────────────────────────────────────

def load_last_template(prefs: "Optional[Path]" = None) -> str:
    """The template filled last ('' when none, or it is gone)."""
    p = Path(prefs) if prefs is not None else PREFS_PATH
    try:
        with open(p, encoding="utf-8") as fh:
            path = str(json.load(fh).get("template", ""))
    except (OSError, ValueError, AttributeError):
        return ""
    return path if path and os.path.exists(path) else ""


def save_last_template(template: str, prefs: "Optional[Path]" = None) -> None:
    p = Path(prefs) if prefs is not None else PREFS_PATH
    try:
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"template": str(template)}, fh)
    except OSError:
        pass
