"""Phase 4: the Qt interface exports exactly what the Tk interface exports.

The same scenario runs in each interface (tools/interface_outputs.py, one
process each): open, Detect Peaks (Auto), a noisy region and an annotation,
Analyze, intervals, epochs, sliding, autonomic, events. Then the Excel
workbook (cell by cell), the RR CSV, the report text, the provenance rows
and the session state are compared (tools/compare_interfaces.py). Then each
interface's session is restored in both, and those exports compared too.

Slow (a few minutes): needs Tk with a display (xvfb on Linux) and Qt.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"
sys.path.insert(0, str(ROOT / "tools"))

pytest.importorskip("tkinter")
pytest.importorskip("PySide6.QtWidgets")
from ecg.core import analysis  # noqa: E402

needs_display = pytest.mark.skipif(
    sys.platform.startswith("linux") and not os.environ.get("DISPLAY")
    and shutil.which("xvfb-run") is None, reason="no display for Tk")
pytestmark = [needs_display,
              pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")]


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    from compare_interfaces import run_both
    out = tmp_path_factory.mktemp("tk_vs_qt")
    return run_both(FIXTURE, out), out


def test_exports_are_equal(outputs):
    from compare_interfaces import compare_dirs
    (tk, qt), _out = outputs
    diffs, stats = compare_dirs(tk, qt)
    assert diffs == [], "\n".join(diffs[:30])
    assert stats["workbook cells"] > 100_000 and stats["rr.csv lines"] > 1000


@pytest.mark.parametrize("saved_by", ["tk", "qt"])
def test_sessions_restore_the_same_in_both(outputs, saved_by):
    import subprocess
    from compare_interfaces import _drop_times, _json_diffs, compare_dirs
    (tk, qt), out = outputs
    session = (tk if saved_by == "tk" else qt) / "session.json"
    dirs = {}
    for ui in ("tk", "qt"):
        home = out / f"restore_{saved_by}_home_{ui}"
        home.mkdir()
        env = dict(os.environ, HOME=str(home), USERPROFILE=str(home))
        cmd = [sys.executable, str(ROOT / "tools" / "interface_outputs.py"), "--ui", ui,
               "--restore", str(session), "--out", str(out / f"restore_{saved_by}_{ui}"),
               str(FIXTURE)]
        if ui == "tk" and sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            cmd = ["xvfb-run", "-a"] + cmd
        if ui == "qt":
            env.setdefault("QT_QPA_PLATFORM", "offscreen")
        p = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True, text=True)
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
        dirs[ui] = out / f"restore_{saved_by}_{ui}"
    diffs, _stats = compare_dirs(dirs["tk"], dirs["qt"])
    assert diffs == [], "\n".join(diffs[:30])
    saved = _drop_times(json.loads(session.read_text(encoding="utf-8")))
    for ui in ("tk", "qt"):
        back = _drop_times(json.loads((dirs[ui] / "session.json").read_text(encoding="utf-8")))
        assert _json_diffs(saved, back) == [], ui
