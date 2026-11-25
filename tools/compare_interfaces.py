# -*- coding: utf-8 -*-
"""
tools/compare_interfaces.py -- phase 4 of the migration: does the Qt
interface export exactly what the Tk interface exports?

    python tools/compare_interfaces.py recording.mat [more.mat ...] [--keep DIR]
    python tools/compare_interfaces.py --dirs TK_DIR QT_DIR

For each recording, tools/interface_outputs.py runs the same scenario in
each interface (each in its own process; the Tk one needs a display) and
the outputs are compared:

* the Excel workbook, cell by cell: the same sheets, the same cells, the
  same values (numbers to the last bit; a cell holding a date-time stamp
  in both is only required to be a stamp in both);
* the RR CSV, byte for byte;
* the report text and the Report tab's provenance rows, line by line;
* the session state Save Session writes, key by key (its save / compute
  time stamps excepted).

Exit status 0 when everything is equal, 1 otherwise (the differences are
listed). Run it on your own recordings before the Tk interface is removed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STAMP = re.compile(r"\d{4}-\d{2}-\d{2}[ T]+\d{2}:\d{2}")
TIME_KEYS = {"saved_at", "computed_at", "peaks_edited_at"}


def _same_value(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def _is_stamp(v: Any) -> bool:
    import datetime as dt
    return isinstance(v, (dt.datetime, dt.date)) or (isinstance(v, str) and bool(STAMP.search(v)))


def compare_workbooks(a: Path, b: Path) -> "tuple[list[str], int, int]":
    """(differences, cells compared, stamp cells) of two .xlsx files."""
    import openpyxl
    wa = openpyxl.load_workbook(a, data_only=True)
    wb = openpyxl.load_workbook(b, data_only=True)
    diffs: "list[str]" = []
    n = stamps = 0
    if wa.sheetnames != wb.sheetnames:
        diffs.append(f"workbook sheets: {wa.sheetnames} != {wb.sheetnames}")
    for name in wa.sheetnames:
        if name not in wb.sheetnames:
            continue
        sa, sb = wa[name], wb[name]
        rows = max(sa.max_row, sb.max_row)
        cols = max(sa.max_column, sb.max_column)
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                va, vb = sa.cell(r, c).value, sb.cell(r, c).value
                n += 1
                if _same_value(va, vb):
                    continue
                if _is_stamp(va) and _is_stamp(vb):
                    stamps += 1
                    continue
                diffs.append(f"workbook {name}!{sa.cell(r, c).coordinate}: {va!r} != {vb!r}")
    return diffs, n, stamps


def _drop_times(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _drop_times(v) for k, v in obj.items() if k not in TIME_KEYS}
    if isinstance(obj, list):
        return [_drop_times(v) for v in obj]
    return obj


def _json_diffs(a: Any, b: Any, path: str = "") -> "list[str]":
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"session {path}/{k}: only in {'Qt' if k in b else 'Tk'}")
            else:
                out += _json_diffs(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"session {path}: {len(a)} items != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _json_diffs(x, y, f"{path}[{i}]")
            if len(out) > 20:
                break
        return out
    return [] if _same_value(a, b) else [f"session {path}: {a!r} != {b!r}"]


def compare_dirs(tk: Path, qt: Path) -> "tuple[list[str], dict]":
    diffs: "list[str]" = []
    stats: dict = {}
    d, n, stamps = compare_workbooks(tk / "workbook.xlsx", qt / "workbook.xlsx")
    diffs += d
    stats["workbook cells"], stats["workbook stamp cells"] = n, stamps
    if (tk / "rr.csv").read_bytes() != (qt / "rr.csv").read_bytes():
        diffs.append("rr.csv differs")
    stats["rr.csv lines"] = len((tk / "rr.csv").read_text().splitlines())
    for name in ("report.txt", "provenance.json"):
        la = (tk / name).read_text(encoding="utf-8").splitlines()
        lb = (qt / name).read_text(encoding="utf-8").splitlines()
        stats[f"{name} lines"] = len(la)
        for i, (x, y) in enumerate(zip(la, lb)):
            if x != y:
                diffs.append(f"{name} line {i + 1}: {x!r} != {y!r}")
        if len(la) != len(lb):
            diffs.append(f"{name}: {len(la)} lines != {len(lb)}")
    sa = _drop_times(json.loads((tk / "session.json").read_text(encoding="utf-8")))
    sb = _drop_times(json.loads((qt / "session.json").read_text(encoding="utf-8")))
    diffs += _json_diffs(sa, sb)
    stats["session keys"] = len(sa)
    return diffs, stats


def run_both(recording: Path, out: Path) -> "tuple[Path, Path]":
    """Both interfaces' outputs for *recording* in out/tk and out/qt, each
    process with its own HOME (no user setting, session or model is read)."""
    py = sys.executable
    script = str(ROOT / "tools" / "interface_outputs.py")
    procs = []
    for ui in ("tk", "qt"):
        home = out / f"home_{ui}"
        home.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, HOME=str(home), USERPROFILE=str(home))
        if ui == "qt":
            env.setdefault("QT_QPA_PLATFORM", "offscreen")
        cmd = [py, script, "--ui", ui, "--out", str(out / ui), str(recording)]
        if ui == "tk" and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            cmd = ["xvfb-run", "-a"] + cmd
        procs.append((ui, subprocess.Popen(cmd, env=env, cwd=str(ROOT),
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True)))
    for ui, p in procs:
        log, _ = p.communicate()
        (out / f"{ui}.log").write_text(log, encoding="utf-8")
        if p.returncode != 0:
            raise RuntimeError(f"{ui} run failed (see {out / f'{ui}.log'}):\n{log[-2000:]}")
    return out / "tk", out / "qt"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recordings", nargs="*", type=Path)
    ap.add_argument("--dirs", nargs=2, type=Path, metavar=("TK_DIR", "QT_DIR"))
    ap.add_argument("--keep", type=Path, help="folder for the outputs (default: temporary)")
    a = ap.parse_args(argv)
    pairs = []
    if a.dirs:
        pairs.append(("(folders)", a.dirs[0], a.dirs[1]))
    tmp = None
    if a.recordings:
        base = a.keep or Path(tempfile.mkdtemp(prefix="ecg_tk_vs_qt_"))
        tmp = None if a.keep else base
        for rec in a.recordings:
            tk, qt = run_both(rec.resolve(), base / rec.stem)
            pairs.append((rec.name, tk, qt))
    if not pairs:
        ap.error("give recordings or --dirs")
    bad = 0
    for name, tk, qt in pairs:
        diffs, stats = compare_dirs(tk, qt)
        status = "EQUAL" if not diffs else f"{len(diffs)} DIFFERENCES"
        print(f"{name}: {status}  ({', '.join(f'{k}: {v}' for k, v in stats.items())})")
        for d in diffs[:50]:
            print("   ", d)
        bad += bool(diffs)
    if tmp is not None:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
