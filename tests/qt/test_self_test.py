"""ecg_app.py --self-test: the check every frozen Qt build runs
(.github/workflows/build-apps.yml): the window, its tabs, a figure, and on
a recording Detect Peaks, Analyze and Measure intervals."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ecg.core import analysis

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"


@pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")
def test_self_test_on_the_fixture(tmp_path):
    out = tmp_path / "result.txt"
    env = dict(os.environ, HOME=str(tmp_path), USERPROFILE=str(tmp_path),
               QT_QPA_PLATFORM=os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    p = subprocess.run([sys.executable, str(ROOT / "ecg_app.py"), "--self-test", str(out),
                        str(FIXTURE)], env=env, cwd=str(ROOT), capture_output=True, text=True,
                       timeout=600)
    text = out.read_text(encoding="utf-8")
    assert p.returncode == 0, text + p.stderr[-2000:]
    assert text.startswith("ok ") and "tabs 8" in text and "intervals" in text
