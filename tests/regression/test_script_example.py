"""examples/analyse_recording.py analyses a recording without the interface
and gets the app's numbers: its results equal the reference outputs of the
app's own pipeline (tests/regression/golden.py, start-up settings)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ecg.core import analysis
from tests.regression import golden

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"


def _example():
    spec = importlib.util.spec_from_file_location("analyse_recording",
                                                  ROOT / "examples" / "analyse_recording.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_gives_the_apps_results(tmp_path):
    ex = _example()
    assert ex.main([str(FIXTURE), "--out", str(tmp_path)]) == 0
    ref, _ = golden.load(ROOT / "tests" / "regression" / "reference", "synthetic_mouse_ecg")
    out = ex.analyse(FIXTURE)
    assert np.array_equal(out["peaks"], ref["detect/auto_raw/rpeaks_ok"])
    # The app's results dict also receives the frequency, non-linear and
    # interval frames computed after Analyze (as the reference holds them):
    # compare the core part here, frequency / non-linear below.
    later = ("/hrv_freq/", "/hrv_nonlin/", "/intervals/")
    flat = {k: v for k, v in golden.flatten(out["results"], "analysis/core/results").items()
            if not any(x in k for x in later)}
    assert golden.compare({k: ref[k] for k in flat}, flat, rtol=0, atol=0) == []
    assert np.allclose(out["epochs"]["RMSSD"], ref["epochs/default/RMSSD"], rtol=0, atol=0)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["hr_mean_bpm"] == ref["analysis/core/results/hr/mean"]
    assert summary["lf_hf"] == float(ref["analysis/frequency/HRV_LFHF"][0])
    assert summary["dfa_alpha1"] == float(ref["analysis/nonlinear/HRV_DFA_alpha1"][0])
    assert (tmp_path / "rr.csv").exists() and (tmp_path / "epochs.csv").exists()


def test_script_does_not_import_the_interface():
    src = (ROOT / "examples" / "analyse_recording.py").read_text(encoding="utf-8")
    assert "ecg.ui" not in src and "customtkinter" not in src and "tkinter" not in src
