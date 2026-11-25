"""examples/analyse_recording.py analyses a recording without the interface
and gets the app's numbers: its results equal the reference outputs of the
app's own pipeline (tests/regression/golden.py, FilterParams defaults)."""
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
    """Same tolerance and same library-version rule as the reference
    (tests/regression/test_golden_master.py)."""
    ex = _example()
    assert ex.main([str(FIXTURE), "--out", str(tmp_path)]) == 0
    ref, manifest = golden.load(ROOT / "tests" / "regression" / "reference", "synthetic_mouse_ecg")
    out = ex.analyse(FIXTURE)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    new = {"detect/auto_raw/rpeaks_ok": out["peaks"],
           "epochs/default/RMSSD": out["epochs"]["RMSSD"].to_numpy(),
           "analysis/core/results/hr/mean": summary["hr_mean_bpm"],
           "analysis/frequency/HRV_LFHF": np.array([summary["lf_hf"]]),
           "analysis/nonlinear/HRV_DFA_alpha1": np.array([summary["dfa_alpha1"]])}
    # The app's results dict also receives the frequency, non-linear and
    # interval frames computed after Analyze (as the reference holds them):
    # the core part is compared here, frequency / non-linear above.
    later = ("/hrv_freq/", "/hrv_nonlin/", "/intervals/")
    new.update({k: v for k, v in golden.flatten(out["results"], "analysis/core/results").items()
                if not any(x in k for x in later)})
    tol = manifest.get("tolerance", {})
    diffs = golden.compare({k: ref[k] for k in new}, new,
                           tol.get("rtol", golden.RTOL), tol.get("atol", golden.ATOL))
    if diffs:
        changed = golden.package_changes(manifest)
        report = "\n".join(str(d) for d in diffs[:20])
        if changed:
            versions = ", ".join(f"{p} {a} -> {b}" for p, (a, b) in changed.items())
            pytest.xfail(f"library versions differ from the reference ({versions}):\n{report}")
        pytest.fail(f"{len(diffs)} result(s) of the script differ from the app's:\n{report}")
    assert (tmp_path / "rr.csv").exists() and (tmp_path / "epochs.csv").exists()


def test_script_does_not_import_the_interface():
    src = (ROOT / "examples" / "analyse_recording.py").read_text(encoding="utf-8")
    assert "ecg.ui" not in src and "customtkinter" not in src and "tkinter" not in src
