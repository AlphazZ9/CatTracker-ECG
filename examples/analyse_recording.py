"""
examples/analyse_recording.py -- analyse one recording without the
interface, with the app's own code and default settings.

    python examples/analyse_recording.py D:/ECG/248_Baseline.mat --out D:/ECG/248_results

Writes, in the output folder:
    summary.json      HR, time-domain / frequency / non-linear HRV, counts
    rr.csv            every RR interval: end time (s), RR (ms), valid or not
    epochs.csv        HRV per 60 s epoch

The steps are those of the app's buttons, with the settings the app starts
with (no filtering, Auto detector, threshold 0.5, RR 66.7-333.3 ms, no
artifact correction): Detect Peaks (ecg.core.pipeline, then the threshold of
ecg.core.peak_editing), Analyze (ecg.services.analysis_service), then
frequency and non-linear HRV and Epochs. Nothing here imports the
interface; change the settings below to script your own protocol.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecg.core.analysis import analyse_hrv_freq, analyse_hrv_nonlinear  # noqa: E402
from ecg.core.models import FilterParams, MouseECG  # noqa: E402
from ecg.core.peak_editing import apply_peak_edits  # noqa: E402
from ecg.core.pipeline import prepare_signal  # noqa: E402
from ecg.core.windowed_hrv import epoch_grid, epoch_hrv  # noqa: E402
from ecg.io.loaders import load_mat_signal  # noqa: E402
from ecg.services.analysis_service import run_core_analysis  # noqa: E402

THRESHOLD = 0.5                                   # fraction of the median R prominence
RR_BOUNDS_MS = (MouseECG.RR_MIN_MS, MouseECG.RR_MAX_MS)
EPOCH_S = MouseECG.EPOCH_DEFAULT_S                # 60 s


def _first(df: pd.DataFrame, col: str) -> "float | None":
    if df is None or df.empty or col not in df.columns:
        return None
    v = float(df[col].iloc[0])
    return v if np.isfinite(v) else None


def analyse(path: "str | Path", channel: str = "ECG") -> dict:
    """Every result of the steps above for the recording at *path*."""
    sig, ch, _, fs_file = load_mat_signal(str(path), channel)
    fs = int(fs_file) if fs_file is not None else MouseECG.FS_DEFAULT

    # Detect Peaks: preparation and candidates, then the threshold.
    params = FilterParams().to_dict()
    params["sg_window_ms"] = 20.0
    prep = prepare_signal(sig, fs, params)
    peaks = apply_peak_edits(prep["signal_flt"], prep["all_cands"], prep["all_proms"],
                             THRESHOLD, fs, set(), set()).accepted

    # Analyze.
    core = run_core_analysis(peaks, prep["signal_flt"], fs, {
        "artifact_correction": False, "min_rr_ms": MouseECG.MIN_RR_MS,
        "hrv_rr_bounds": RR_BOUNDS_MS,
        "break_tol_samples": max(1, int(round(5.0 / 1000.0 * fs))),
        "noise_regions": [], "analysis_t_start": 0.0, "analysis_t_end": 0.0})
    results, rp = core["results"], core["rpeaks_analysed"]
    freq = analyse_hrv_freq(rp, fs, rr_min_ms=RR_BOUNDS_MS[0], rr_max_ms=RR_BOUNDS_MS[1])
    nonlin = analyse_hrv_nonlinear(rp, fs, max_beats=MouseECG.ENTROPY_MAX_BEATS,
                                   rr_min_ms=RR_BOUNDS_MS[0], rr_max_ms=RR_BOUNDS_MS[1])

    # Epochs.
    _, starts, _ = epoch_grid(rp, fs, EPOCH_S, 0.0)
    epochs = pd.DataFrame(epoch_hrv(rp, fs, starts, EPOCH_S, *RR_BOUNDS_MS, None, None,
                                    False, False, set())["rows"])

    hr = results["hr"]
    td = results["hrv_time"]
    summary = {
        "file": Path(path).name, "channel": ch, "fs_hz": fs,
        "duration_s": len(sig) / fs, "filters": prep["filter_text"],
        "n_peaks": int(len(peaks)), "n_rr_valid": int(hr.get("n_valid", 0)),
        "hr_mean_bpm": hr.get("mean"),
        "sdnn_ms": _first(td, "HRV_SDNN"), "rmssd_ms": _first(td, "HRV_RMSSD"),
        "pnn6_pct": _first(td, "HRV_pNN6"),
        "lf_ms2": _first(freq, "HRV_LF"), "hf_ms2": _first(freq, "HRV_HF"),
        "lf_hf": _first(freq, "HRV_LFHF"),
        "sd1_ms": _first(nonlin, "HRV_SD1"), "sd2_ms": _first(nonlin, "HRV_SD2"),
        "dfa_alpha1": _first(nonlin, "HRV_DFA_alpha1"),
    }
    rr = pd.DataFrame({"t_s": results["rr_t_s"], "rr_ms": results["rr_ms"],
                       "valid": results["rr_valid"]})
    return {"summary": summary, "rr": rr, "epochs": epochs, "peaks": peaks, "results": results}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recording", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--channel", default="ECG")
    args = ap.parse_args(argv)
    out = analyse(args.recording, args.channel)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(out["summary"], indent=1), encoding="utf-8")
    out["rr"].to_csv(args.out / "rr.csv", index=False)
    out["epochs"].to_csv(args.out / "epochs.csv", index=False)
    s = out["summary"]
    print(f"{s['file']}: {s['n_peaks']} peaks, HR {s['hr_mean_bpm']:.1f} bpm, "
          f"RMSSD {s['rmssd_ms']:.2f} ms -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
