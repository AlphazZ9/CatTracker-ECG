"""GraphPad Prism tables and templates: one Column table per metric (rows =
animals, columns = segments named by their labels; per pair in Pairs / All
pairs mode), a new .pzfx, and a Prism template filled with new data while
every byte outside its data tables (the graphs-and-analyses blob above all)
is kept."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from ecg.core.segment_plan import PAIRS, reference_comparators
from ecg.io import group_stats as gs
from ecg.io import prism_template as pt
from ecg.io.group_table import group_row
from ecg.io.segment_export import ComparisonResult

METRICS = [("hr_mean", "Mean HR", "bpm"), ("sdnn", "SDNN", "ms"), ("lf_hf", "LF/HF", "")]
LABELS = ["Pre 1", "Post 1", "Pre 5", "Post 5"]


def _segs(base: float, labels=LABELS) -> "list[dict]":
    return [{"label": lbl, "t_lo": 60.0 * k, "t_hi": 60.0 * k + 60, "duration_s": 60.0,
             "hr_mean": base + 10 * k, "sdnn": 3.0 + 0.5 * k, "lf_hf": 0.25 * (k + 1)}
            for k, lbl in enumerate(labels)]


def _group(n: int = 3, labels=LABELS) -> pd.DataFrame:
    rows = [group_row(f"M{i + 1}", f"D:/data/M{i + 1}.mat", _segs(500 + i, labels), METRICS,
                      flags=[[] for _ in labels]) for i in range(n)]
    return pd.DataFrame(rows)


def _result(comparators=None, labels=LABELS, subject="M7") -> ComparisonResult:
    return ComparisonResult(segs=_segs(480, labels), metrics=METRICS, subject=subject,
                            filepath="D:/data/rec.mat", comparators=comparators)


# ── Builders ─────────────────────────────────────────────────────────────

def test_group_tables_one_per_metric_rows_are_animals_columns_are_labels():
    df = _group()
    tables = pt.group_tables(df)
    assert [t for t, _d in tables] == ["Mean HR (bpm)", "SDNN (ms)", "LF/HF"]
    hr = dict(tables)["Mean HR (bpm)"]
    assert list(hr.index) == ["M1", "M2", "M3"]
    assert list(hr.columns) == LABELS
    assert hr.loc["M2", "Post 5"] == pytest.approx(501 + 30)
    # No meta column (times, quality, labels, indices) is a metric.
    assert pt.group_metric_names(df) == ["Mean HR (bpm)", "SDNN (ms)", "LF/HF"]


def test_group_tables_repeated_labels_keep_one_column_per_segment():
    df = _group(labels=["Pre", "Post", "Pre", "Post"])          # tags A, B, C, D
    hr = dict(pt.group_tables(df))["Mean HR (bpm)"]
    assert list(hr.columns) == ["Pre (A)", "Post (B)", "Pre (C)", "Post (D)"]


def test_pair_tables_from_all_pairs_titles_and_rows_left_out():
    df = _group(4)
    df.loc[2, "QC Post 5"] = "few beats"                         # M3: a row to check
    res = gs.all_pairs_stats(df, exclude=True)
    tables = dict(pt.pair_tables(res))
    assert list(tables)[:3] == ["Mean HR (bpm) · Post 1 vs Pre 1", "SDNN (ms) · Post 1 vs Pre 1",
                                "LF/HF · Post 1 vs Pre 1"]
    t1 = tables["Mean HR (bpm) · Post 1 vs Pre 1"]
    assert list(t1.columns) == ["Pre 1", "Post 1"] and list(t1.index) == ["M1", "M2", "M3", "M4"]
    t2 = tables["SDNN (ms) · Post 5 vs Pre 5"]
    assert list(t2.columns) == ["Pre 5", "Post 5"] and list(t2.index) == ["M1", "M2", "M4"]
    notes = pt.left_out_notes(res.pairs[1].checks, res.pairs[1].keep, res.pairs[1].title)
    assert notes == [("Left out (row to check) — Post 5 vs Pre 5", "M3: Post 5: few beats")]
    # The same through the dispatcher on the table.
    assert list(dict(pt.metric_tables(df, layout="pairs"))) == list(tables)


def test_result_tables_one_row_named_by_the_subject():
    res = _result()
    tables = pt.result_tables(res)
    assert [t for t, _d in tables] == ["Mean HR (bpm)", "SDNN (ms)", "LF/HF"]
    hr = dict(tables)["Mean HR (bpm)"]
    assert list(hr.index) == ["M7"] and list(hr.columns) == LABELS
    res.subject = ""
    assert list(dict(pt.result_tables(res))["SDNN (ms)"].index) == ["rec"]


def test_result_tables_pairs_mode_match_the_group_titles():
    res = _result(reference_comparators(4, PAIRS))
    assert pt.is_pairs(res)
    tables = dict(pt.result_tables(res))
    assert list(tables) == [f"{m} · {p}" for p in ("Post 1 vs Pre 1", "Post 5 vs Pre 5")
                            for m in ("Mean HR (bpm)", "SDNN (ms)", "LF/HF")]
    assert list(tables["SDNN (ms) · Post 5 vs Pre 5"].columns) == ["Pre 5", "Post 5"]
    # The group table of the same protocol gives the same titles.
    group = dict(pt.pair_tables(gs.all_pairs_stats(_group())))
    assert list(group) == list(tables)
    # Repeated labels: the tags say which pair.
    res2 = _result(reference_comparators(4, PAIRS), labels=["Pre", "Post", "Pre", "Post"])
    assert list(dict(pt.result_tables(res2)))[0] == "Mean HR (bpm) · Post vs Pre (B vs A)"
    g2 = gs.all_pairs_stats(_group(labels=["Pre", "Post", "Pre", "Post"]))
    assert list(dict(pt.pair_tables(g2))) == list(dict(pt.result_tables(res2)))


def test_recording_tables_keep_the_existing_tables():
    titles = [t[0] for t in pt.recording_tables(_result())]
    assert titles[0] == "Comparison" and titles[1:4] == ["Mean HR (bpm)", "SDNN (ms)", "LF/HF"]


# ── A new file ───────────────────────────────────────────────────────────

def test_new_file_parses_and_holds_the_tables(tmp_path):
    path = tmp_path / "new.pzfx"
    tables = pt.group_tables(_group())
    n = pt.write_metric_pzfx(tables, str(path), notes=[("Group table", "g.xlsx")])
    assert n == 3
    root = ET.parse(path).getroot()
    assert root.tag == "GraphPadPrismFile"
    got = pt.read_tables(str(path))
    assert list(got) == ["Mean HR (bpm)", "SDNN (ms)", "LF/HF"]
    assert got["SDNN (ms)"]["rows"] == ["M1", "M2", "M3"]
    assert list(got["SDNN (ms)"]["columns"]) == LABELS
    assert got["Mean HR (bpm)"]["columns"]["Pre 5"] == ["520", "521", "522"]
    refs = [r.get("ID") for r in root.find("TableSequence")]
    assert refs == [got[t]["id"] for t in got]
    assert "g.xlsx" in path.read_text(encoding="utf-8")
    pzfx = pytest.importorskip("pzfx")
    back = pzfx.read_pzfx(str(path), table="SDNN (ms)")
    assert list(back.columns)[-4:] == LABELS


def test_numbers_are_plain_decimals():
    assert pt._num_text(1.5e-05) == "0.000015"
    assert pt._num_text(12345678.0) == "12345678"
    assert pt._num_text(float("nan")) is None
    assert pt._d(float("nan")) == "<d/>"
    # The shortest text that reads back as the same number, no exponent.
    assert pt._num_text(2.0 / 3.0) == "0.6666666666666666"
    assert pt._num_text(123456789.123456789) == "123456789.12345679"
    assert float(pt._num_text(123456789.123456789)) == 123456789.123456789
    assert pt._num_text(np.float32(0.1)) == "0.1"                # no float32 noise
    assert pt._num_text(pd.NA) is None and pt._d(pd.NA) == "<d/>"
    assert pt._num_text(pd.NaT) is None


# ── Filling a template ───────────────────────────────────────────────────

BLOB = ("eNrsfQtgVMW58Jyzm+wjmychgCgcHtUAAQIiBsSyiTx/ESNQAUUhkI0Ekw0kiyAqrBIeCgio\r\n"
        "UKGAVKE/VoWoaMVeLaIgtOq9rWhVtJUrVOsDqA+8tZbcmXNm9sxOzjk5k92EbO58MDk7Z858\r\n"
        "gd/fB3rq84loV+rwN99446dGnNRrrQK/I+wHjZ9Y1BhI/v8FAAD//wMA5M1uUA==")


def _col(title, vals, width=87):
    d = "\r\n".join(f"<d>{v}</d>" for v in vals)
    return (f'<YColumn Width="{width}" Decimals="2" Subcolumns="1">\r\n<Title>{title}</Title>\r\n'
            f"<Subcolumn>\r\n{d}\r\n</Subcolumn>\r\n</YColumn>")


def _rows(vals):
    d = "\r\n".join(f"<d>{v}</d>" for v in vals)
    return f'<RowTitlesColumn Width="81">\r\n<Subcolumn>\r\n{d}\r\n</Subcolumn>\r\n</RowTitlesColumn>'


def _template(tmp_path, sdnn_title="SDNN (ms)", cols=("Pre 1", "Post 1")) -> "tuple[str, bytes]":
    """A template as Prism 11 saves one: CRLF, a namespace, two Column tables
    (one the export makes, one it does not), the comment and the blob."""
    t0 = "\r\n".join([
        '<Table ID="Table0" XFormat="none" TableType="OneWay" EVFormat="AsteriskAfterNumber">',
        f"<Title>{sdnn_title}</Title>", _rows(["A1", "A2", "A3", "A4", "A5"]),
        *[_col(c, [10 + k, 11, 12, 13, 14], width=87 + k) for k, c in enumerate(cols)],
        "</Table>"])
    t1 = "\r\n".join([
        '<Table ID="Table1" XFormat="none" TableType="OneWay" EVFormat="AsteriskAfterNumber">',
        "<Title>My own table</Title>", _rows(["x"]), _col("Q", [1]), "</Table>"])
    text = "\r\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<GraphPadPrismFile xmlns="http://graphpad.com/prism/Prism.htm" PrismXMLVersion="5.00">',
        "<Created>",
        '<OriginalVersion CreatedByProgram="GraphPad Prism" CreatedByVersion="6.0f.254" '
        'DateTime="2026-11-20T08:23:10+00:00"/>',
        "</Created>",
        "<InfoSequence>", '<Ref ID="Info0" Selected="1"/>', "</InfoSequence>",
        '<Info ID="Info0">', "<Title>Project Info 1</Title>", "<Notes/>",
        "<Constant><Name>Experiment Date</Name><Value>2026-11-20</Value></Constant>",
        "</Info>", "",
        "<TableSequence>", '<Ref ID="Table0" Selected="1"/>', '<Ref ID="Table1"/>',
        "</TableSequence>", t0, t1, "",
        "<!--Analyses, graphs and layouts as compressed binary. Don't edit this part of the "
        "file.-->", "", "",
        f'<Template xmlns:dt="urn:schemas-microsoft-com:datatypes" dt:dt="bin.base64">{BLOB}'
        "</Template></GraphPadPrismFile>", ""])
    path = tmp_path / "template.pzfx"
    raw = text.encode("utf-8")
    path.write_bytes(raw)
    return str(path), raw


def _blob(raw: bytes) -> bytes:
    a = raw.index(b"<!--Analyses")
    return raw[a:]


def _tables3():
    """SDNN for 3 animals and 3 segments (one more than the template), and
    a metric the template does not have."""
    idx = pd.Index(["M1", "M2", "M3"], name="subject")
    return [("SDNN (ms)", pd.DataFrame({"Pre 1": [3.5, 3.25, np.nan], "Post 1": [4.0, 4.5, 5.0],
                                        "Pre 5": [1.0, 2.0, 3.0]}, index=idx)),
            ("Mean HR (bpm)", pd.DataFrame({"Pre 1": [500.0, 501.0, 502.0],
                                            "Post 1": [510.0, 511.0, 512.0]}, index=idx))]


def test_fill_keeps_the_blob_and_replaces_the_values(tmp_path):
    tpl, raw = _template(tmp_path)
    out = tmp_path / "filled.pzfx"
    rep = pt.fill_template(tpl, _tables3(), str(out), notes=[("Subject", "M1-M3")])
    data = out.read_bytes()
    # Everything from the comment on, the blob above all, byte for byte; the
    # head of the file too (encoding, namespace, the user's own info sheet).
    assert _blob(data) == _blob(raw)
    head = raw[: raw.index(b"</InfoSequence>")]
    info0 = raw[raw.index(b'<Info ID="Info0">'): raw.index(b"</Info>") + len(b"</Info>")]
    assert data.startswith(head) and info0 in data
    assert data.count(b"\n") == data.count(b"\r\n")              # CRLF kept
    assert open(tpl, "rb").read() == raw                          # the template untouched
    ET.fromstring(data)                                           # well-formed
    got = pt.read_tables(str(out))
    sd = got["SDNN (ms)"]
    assert sd["id"] == "Table0"                                   # IDs never change
    assert sd["rows"] == ["M1", "M2", "M3"]
    assert sd["columns"]["Pre 1"] == ["3.5", "3.25", None]
    assert sd["columns"]["Post 1"] == ["4", "4.5", "5"]
    assert sd["columns"]["Pre 5"] == ["1", "2", "3"]              # the extra column appended
    # Its attributes kept; the appended one copies the last column's.
    t0 = data.decode()[data.decode().index('<Table ID="Table0"'):]
    t0 = t0[: t0.index("</Table>")]
    widths = re.findall(r'<YColumn Width="(\d+)"', t0)
    assert widths == ["87", "88", "88"]
    # The table not in the template: appended with a new ID, in the sequence.
    hr = got["Mean HR (bpm)"]
    assert hr["id"] == "Table2" and hr["columns"]["Post 1"] == ["510", "511", "512"]
    root = ET.fromstring(data)
    ns = "{http://graphpad.com/prism/Prism.htm}"
    assert [r.get("ID") for r in root.find(f"{ns}TableSequence")] == ["Table0", "Table1", "Table2"]
    # The template's own table: left as it was, and reported.
    assert got["My own table"]["columns"]["Q"] == ["1"]
    assert rep.untouched == [("Table1", "My own table")]
    assert rep.matched == [("SDNN (ms)", "Table0")]
    assert rep.appended == [("Mean HR (bpm)", "Table2")]
    assert any("'Pre 5' added" in n for n in rep.notes)
    assert "1 table filled" in rep.summary() and "1 added" in rep.summary()
    # The notes: a sheet of their own, the user's kept.
    infos = [i for i in root if i.tag == f"{ns}Info"]
    assert [i.find(f"{ns}Title").text for i in infos] == ["Project Info 1", pt.NOTES_TITLE]
    assert [r.get("ID") for r in root.find(f"{ns}InfoSequence")] == ["Info0", "Info1"]
    assert "M1-M3" in data.decode() and "My own table" in data.decode()


def test_fill_matches_titles_case_and_spacing_insensitive_and_columns_by_position(tmp_path):
    tpl, raw = _template(tmp_path, sdnn_title="sdnn  (MS)", cols=("Baseline", "Stim", "Old"))
    out = tmp_path / "filled.pzfx"
    tables = [("SDNN (ms)", pd.DataFrame({"Pre 1": [1.0, 2.0], "Post 1": [3.0, 4.0]},
                                         index=pd.Index(["M1", "M2"])))]
    rep = pt.fill_template(tpl, tables, str(out))
    got = pt.read_tables(str(out))
    sd = got["sdnn (MS)"]                                         # the template's title kept
    assert sd["columns"] == {"Baseline": ["1", "2"], "Stim": ["3", "4"], "Old": [None, None]}
    assert any("by position" in n for n in rep.notes)
    assert any("'Old' has no data" in n for n in rep.notes)
    assert _blob(out.read_bytes()) == _blob(raw)
    # No notes given: no info sheet added.
    assert pt.NOTES_TITLE not in out.read_text(encoding="utf-8")


def test_refilling_a_filled_file_replaces_its_own_notes(tmp_path):
    tpl, raw = _template(tmp_path)
    first = tmp_path / "first.pzfx"
    pt.fill_template(tpl, _tables3(), str(first), notes=[("Run", "one")])
    second = tmp_path / "second.pzfx"
    rep = pt.fill_template(str(first), _tables3(), str(second), notes=[("Run", "two")])
    text = second.read_text(encoding="utf-8")
    assert text.count(f"<Title>{pt.NOTES_TITLE}</Title>") == 1
    assert "two" in text and "<Value>one</Value>" not in text
    assert rep.appended == [] and len(rep.matched) == 2          # Mean HR is a table now
    assert _blob(second.read_bytes()) == _blob(raw)


def test_template_xy_table_gets_its_x_values(tmp_path):
    path = tmp_path / "xy.pzfx"
    df = pd.DataFrame({"Time in segment (s)": [0.0, 0.1, 0.2], "RR Pre (ms)": [120.0, 121, 119]})
    pt.write_metric_pzfx([("RR Pre", df, "Time in segment (s)")], str(path))
    text = path.read_text(encoding="utf-8").replace("</GraphPadPrismFile>",
                                                    f'<Template dt:dt="bin.base64" '
                                                    f'xmlns:dt="urn:x">{BLOB}</Template>'
                                                    "</GraphPadPrismFile>")
    path.write_bytes(text.encode("utf-8"))
    df2 = pd.DataFrame({"Time in segment (s)": [0.0, 0.2], "RR Pre (ms)": [130.0, 131]})
    out = tmp_path / "xy_filled.pzfx"
    pt.fill_template(str(path), [("RR Pre", df2, "Time in segment (s)")], str(out))
    got = pt.read_tables(str(out))["RR Pre"]
    assert got["x"] == ("Time in segment (s)", ["0", "0.2"])
    assert got["columns"]["RR Pre (ms)"] == ["130", "131"] and got["rows"] is None
    data = out.read_bytes()
    assert BLOB.encode() in data
    # LF file, CRLF inside its blob: the new lines follow the file's.
    assert b"\r\n" not in data[: data.index(b"<Template")]


def test_a_prism_binary_file_is_refused_with_how_to_save_it(tmp_path):
    binary = tmp_path / "graphs.prism"
    binary.write_bytes(b"PK\x03\x04\x14\x00\x00\x00binary")
    with pytest.raises(pt.PrismTemplateError, match=r"Save As > Prism XML \.pzfx"):
        pt.fill_template(str(binary), _tables3(), str(tmp_path / "o.pzfx"))
    junk = tmp_path / "junk.pzfx"
    junk.write_bytes(b"\x00\x01\x02 not xml")
    with pytest.raises(pt.PrismTemplateError, match="not a Prism XML file"):
        pt.fill_template(str(junk), _tables3(), str(tmp_path / "o.pzfx"))
    assert not (tmp_path / "o.pzfx").exists()


def test_the_template_itself_is_never_overwritten(tmp_path):
    tpl, raw = _template(tmp_path)
    with pytest.raises(pt.PrismTemplateError, match="never"):
        pt.fill_template(tpl, _tables3(), tpl)
    assert open(tpl, "rb").read() == raw


def test_last_template_is_remembered(tmp_path):
    prefs = tmp_path / "prefs.json"
    assert pt.load_last_template(prefs) == ""
    tpl, _raw = _template(tmp_path)
    pt.save_last_template(tpl, prefs)
    assert pt.load_last_template(prefs) == tpl
    (tmp_path / "template.pzfx").unlink()
    assert pt.load_last_template(prefs) == ""                     # gone: not offered


def test_template_without_row_titles_or_table_sequence(tmp_path):
    """A table with no row titles gets them before its columns; a file with
    no TableSequence gets one for the table it adds."""
    text = "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<GraphPadPrismFile PrismXMLVersion="5.00">',
        '<Table ID="Table3" XFormat="none" TableType="OneWay">',
        "<Title>SDNN (ms)</Title>", _col("Pre 1", [1, 2]).replace("\r\n", "\n"), "</Table>",
        f'<Template dt:dt="bin.base64" xmlns:dt="urn:x">{BLOB}</Template>',
        "</GraphPadPrismFile>", ""])
    tpl = tmp_path / "bare.pzfx"
    tpl.write_bytes(text.encode("utf-8"))
    out = tmp_path / "bare_filled.pzfx"
    rep = pt.fill_template(str(tpl), _tables3(), str(out))
    data = out.read_bytes()
    ET.fromstring(data)
    got = pt.read_tables(str(out))
    assert got["SDNN (ms)"]["rows"] == ["M1", "M2", "M3"]
    assert got["SDNN (ms)"]["columns"]["Pre 1"] == ["3.5", "3.25", None]
    assert got["Mean HR (bpm)"]["id"] == "Table4" and rep.appended == [("Mean HR (bpm)", "Table4")]
    root = ET.fromstring(data)
    assert [r.get("ID") for r in root.find("TableSequence")] == ["Table3", "Table4"]
    assert BLOB.encode() in data


# ── Review fixes: XAdvancedColumn, free columns, stale data, bad XML ─────

def _xy_template(tmp_path, with_y=True) -> str:
    """An XY table as Prism 8+ saves one: a FloatingNote after the title,
    the X values twice (XColumn and XAdvancedColumn Version="1")."""
    xs = "\n".join(f"<d>{v}</d>" for v in (1, 2, 3, 4))
    y = ('<YColumn Width="169" Decimals="0" Subcolumns="1">\n<Title>RR Pre (ms)</Title>\n'
         "<Subcolumn>\n<d>9</d>\n<d>9</d>\n<d>9</d>\n<d>9</d>\n</Subcolumn>\n</YColumn>")
    text = "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<GraphPadPrismFile PrismXMLVersion="5.00">',
        "<TableSequence>", '<Ref ID="Table0" Selected="1"/>', "</TableSequence>",
        '<Table ID="Table0" XFormat="numbers" YFormat="replicates" Replicates="1" '
        'TableType="XY" EVFormat="AsteriskAfterNumber">',
        "<Title>RR Pre</Title>",
        '<FloatingNote ID="Sticky1" Auto="0" Color="Yellow">note</FloatingNote>',
        '<XColumn Width="158" Subcolumns="1" Decimals="1">', "<Title>Time in segment (s)</Title>",
        f"<Subcolumn>\n{xs}\n</Subcolumn>", "</XColumn>",
        '<XAdvancedColumn Version="1" Width="158" Decimals="1" Subcolumns="1">',
        "<Title>Time in segment (s)</Title>", f"<Subcolumn>\n{xs}\n</Subcolumn>",
        "</XAdvancedColumn>", *([y] if with_y else []), "</Table>",
        f'<Template dt:dt="bin.base64" xmlns:dt="urn:x">{BLOB}</Template>',
        "</GraphPadPrismFile>", ""])
    path = tmp_path / ("xy_tpl.pzfx" if with_y else "xy_tpl_noy.pzfx")
    path.write_bytes(text.encode("utf-8"))
    return str(path)


def test_xadvanced_column_gets_the_new_x_values(tmp_path):
    tpl = _xy_template(tmp_path)
    df = pd.DataFrame({"Time in segment (s)": [0.0, 0.5, 1.0], "RR Pre (ms)": [120.0, 121, 119]})
    out = tmp_path / "xy_out.pzfx"
    pt.fill_template(tpl, [("RR Pre", df, "Time in segment (s)")], str(out))
    root = ET.fromstring(out.read_bytes())
    table = root.find("Table")
    for tag in ("XColumn", "XAdvancedColumn"):
        col = table.find(tag)
        assert [d.text for d in col.find("Subcolumn")] == ["0", "0.5", "1"], tag
        assert col.find("Title").text == "Time in segment (s)"
    assert table.find("XAdvancedColumn").get("Version") == "1"
    assert [d.text for d in table.find("YColumn").find("Subcolumn")] == ["120", "121", "119"]


def test_new_y_column_goes_after_the_x_columns(tmp_path):
    tpl = _xy_template(tmp_path, with_y=False)
    df = pd.DataFrame({"Time in segment (s)": [0.0, 0.5], "RR Pre (ms)": [120.0, 121]})
    out = tmp_path / "xy_out2.pzfx"
    pt.fill_template(tpl, [("RR Pre", df, "Time in segment (s)")], str(out))
    order = [ch.tag for ch in ET.fromstring(out.read_bytes()).find("Table")]
    assert order == ["Title", "FloatingNote", "XColumn", "XAdvancedColumn", "YColumn"]


def test_unmatched_columns_take_the_free_template_columns_first(tmp_path):
    tpl, _raw = _template(tmp_path, cols=("Other", "Pre 1"))
    out = tmp_path / "free.pzfx"
    tables = [("SDNN (ms)", pd.DataFrame({"Pre 1": [1.0], "Post 1": [2.0]},
                                         index=pd.Index(["M1"])))]
    rep = pt.fill_template(tpl, tables, str(out))
    cols = pt.read_tables(str(out))["SDNN (ms)"]["columns"]
    assert cols == {"Other": ["2"], "Pre 1": ["1"]}               # nothing appended
    assert any("'Post 1' written in column 'Other'" in n for n in rep.notes)


def test_a_pair_without_animals_empties_its_template_tables(tmp_path):
    """A template made with both pairs, filled when the second pair keeps
    no row: its tables are emptied (never the previous animals), reported,
    and the user's own table is left alone."""
    df = _group(4)
    res = gs.all_pairs_stats(df, exclude=True)
    tables = pt.pair_tables(res)
    new = tmp_path / "pairs.pzfx"
    pt.write_metric_pzfx(tables, str(new))
    text = new.read_text(encoding="utf-8").replace(
        "</GraphPadPrismFile>",
        '<Table ID="Table99" XFormat="none" TableType="OneWay"><Title>My own table</Title>'
        "<YColumn><Title>Q</Title><Subcolumn><d>7</d></Subcolumn></YColumn></Table>"
        f'<Template dt:dt="bin.base64" xmlns:dt="urn:x">{BLOB}</Template></GraphPadPrismFile>')
    tpl = tmp_path / "pairs_tpl.pzfx"
    tpl.write_bytes(text.encode("utf-8"))
    # Every animal of pair 2 left out: Post 5 to check for all of them.
    df2 = _group(3)
    for i in range(3):
        df2.loc[i, "QC Post 5"] = "few beats"
    res2 = gs.all_pairs_stats(df2, exclude=True)
    assert len(res2.pairs[1].used) == 0
    tables2 = pt.pair_tables(res2)
    p2 = "SDNN (ms) · Post 5 vs Pre 5"
    assert p2 in dict(tables2) and len(dict(tables2)[p2]) == 0    # the pair's tables, no row
    out = tmp_path / "pairs_filled.pzfx"
    rep = pt.fill_template(str(tpl), tables2, str(out))
    got = pt.read_tables(str(out))
    assert got[p2]["rows"] == [] and all(v == [] for v in got[p2]["columns"].values())
    assert list(got[p2]["columns"]) == ["Pre 5", "Post 5"]        # columns kept
    assert got["SDNN (ms) · Post 1 vs Pre 1"]["rows"] == ["M1", "M2", "M3"]
    assert got["My own table"]["columns"]["Q"] == ["7"]
    assert rep.untouched == [("Table99", "My own table")] and rep.warn


def test_a_recording_table_filled_from_a_group_is_emptied(tmp_path):
    """A template made from one recording's export (Comparison, RR, Poincaré
    tables) filled from the group: those tables get no stale animal."""
    res = _result()
    res.segs[0]["t_rr"] = [0.0, 0.12, 0.24]
    res.segs[0]["rr_ms"] = [120.0, 121.0, 119.0]
    new = tmp_path / "rec.pzfx"
    pt.write_metric_pzfx(pt.recording_tables(res), str(new))
    text = new.read_text(encoding="utf-8").replace(
        "</GraphPadPrismFile>",
        f'<Template dt:dt="bin.base64" xmlns:dt="urn:x">{BLOB}</Template></GraphPadPrismFile>')
    tpl = tmp_path / "rec_tpl.pzfx"
    tpl.write_bytes(text.encode("utf-8"))
    out = tmp_path / "rec_group.pzfx"
    rep = pt.fill_template(str(tpl), pt.group_tables(_group()), str(out))
    got = pt.read_tables(str(out))
    cleared = {t for _i, t in rep.cleared}
    assert {"Comparison", "RR Pre 1"} <= cleared and not rep.untouched
    assert got["Comparison"]["rows"] == [] and got["RR Pre 1"]["x"][1] == []
    assert got["SDNN (ms)"]["rows"] == ["M1", "M2", "M3"]
    assert "emptied" in rep.summary()


def test_a_malformed_template_is_refused_and_nothing_written(tmp_path):
    tpl, raw = _template(tmp_path)
    bad = tmp_path / "bad.pzfx"
    bad.write_bytes(raw.replace(b"</Title>", b"</Titel>", 1))
    out = tmp_path / "bad_out.pzfx"
    with pytest.raises(pt.PrismTemplateError, match=r"not a valid Prism XML file \(line \d+\)"):
        pt.fill_template(str(bad), _tables3(), str(out))
    cut = tmp_path / "cut.pzfx"
    cut.write_bytes(raw[: raw.index(BLOB.encode()) + 20])         # truncated in the blob
    with pytest.raises(pt.PrismTemplateError, match="save it again as .pzfx"):
        pt.fill_template(str(cut), _tables3(), str(out))
    assert not out.exists()


def test_untitled_template_table_is_named_by_its_id():
    rep = pt.FillReport(untouched=[("Table5", "")])
    assert "(Table5)" in rep.summary() and rep.rows()[-1][1] == "Table5"
