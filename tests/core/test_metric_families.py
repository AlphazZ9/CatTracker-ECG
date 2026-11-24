"""
Metric families: every metric name the app writes to a group table, a
comparison or a Group-tab menu gets its family (Heart rate, HRV time domain,
frequency, non-linear, ECG intervals, Quality...), the same one its metric
table row has (MetricRow.group), and lists come out in family order.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ecg.core import analysis
from ecg.core import metric_families as mf
from ecg.core.analysis import analyse_core, analyse_hrv_freq, analyse_hrv_nonlinear
from ecg.core.metric_table import build_metric_table
from ecg.io import group_data as gd

from tests.helpers import make_synth_ecg

# The Whole sheet columns of the group table made from Baseline_STIM_BNST_206
# (mouse, Analyze on the whole recording): the names as the app writes them.
WHOLE_206 = (
    "HR mean (60000 / mean NN) (bpm)", "HR min (2nd percentile) (bpm)",
    "HR max (98th percentile) (bpm)", "HR SD (instantaneous) (bpm)", "Beats analysed (beats)",
    "Valid RR intervals (intervals)", "HR mean of rr_df (bpm)", "RR mean of rr_df (ms)",
    "HR range low (2 % quantile of rr_df) (bpm)",
    "HR range high (98 % quantile of rr_df) (bpm)", "RR mean (ms)", "RR intervals (intervals)",
    "RR excluded from HRV (intervals)", "RR excluded: too short (intervals)",
    "RR excluded: too long (intervals)", "RR excluded: spanning removed beats (intervals)",
    "RR excluded (%)", "RR excluded time (s)", "RR lower bound (ms)", "RR upper bound (ms)",
    "Analysed duration (s)", "Signal quality score (%)", "Mean beat correlation",
    "Beats below 0.90 (beats)", "Beats below 0.90 (%)", "Poor-shape-match time (%)",
    "MeanNN (ms)", "SDNN (ms)", "HRV_SDANN1 (ms)", "HRV_SDNNI1 (ms)", "HRV_SDANN2 (ms)",
    "HRV_SDNNI2 (ms)", "HRV_SDANN5 (ms)", "HRV_SDNNI5 (ms)", "RMSSD (ms)", "HRV_SDSD (ms)",
    "HRV_CVNN", "HRV_CVSD", "HRV_MedianNN (ms)", "HRV_MadNN (ms)", "HRV_MCVNN",
    "HRV_IQRNN (ms)", "HRV_SDRMSSD", "HRV_Prc20NN (ms)", "HRV_Prc80NN (ms)",
    "pNN50 (human) (%)", "pNN20 (human) (%)", "HRV_MinNN (ms)", "HRV_MaxNN (ms)", "HRV_HTI",
    "HRV_TINN (ms)", "pNN6 (%)", "VLF power (ms²)", "LF power (ms²)", "HF power (ms²)",
    "HRV_TP (ms²)", "LF/HF (ratio)", "LF (n.u.) (fraction of LF+HF)",
    "HF (n.u.) (fraction of LF+HF)", "HRV_LnHF (ln ms²)", "HRV_LFpeak (Hz)", "HRV_HFpeak (Hz)",
    "Porta index (PI) (%)", "Guzik index (GI) (%)", "Decelerations (ΔRR > 0) (pairs)",
    "Accelerations (ΔRR < 0) (pairs)", "Pairs (pairs)", "No change (ΔRR = 0) (pairs)",
    "SD1 (ms)", "SD2 (ms)", "HRV_SD1SD2", "HRV_S", "HRV_CSI", "HRV_CVI", "HRV_CSI_Modified",
    "HRV_PIP", "HRV_IALS", "HRV_PSS", "HRV_PAS", "HRV_GI", "HRV_SI", "HRV_AI", "HRV_PI",
    "HRV_C1d", "HRV_C1a", "HRV_SD1d", "HRV_SD1a", "HRV_C2d", "HRV_C2a", "HRV_SD2d", "HRV_SD2a",
    "HRV_Cd", "HRV_Ca", "HRV_SDNNd", "HRV_SDNNa", "DFA α1", "HRV_MFDFA_alpha1_Width",
    "HRV_MFDFA_alpha1_Peak", "HRV_MFDFA_alpha1_Mean", "HRV_MFDFA_alpha1_Max",
    "HRV_MFDFA_alpha1_Delta", "HRV_MFDFA_alpha1_Asymmetry", "HRV_MFDFA_alpha1_Fluctuation",
    "HRV_MFDFA_alpha1_Increment", "DFA α2", "HRV_MFDFA_alpha2_Width", "HRV_MFDFA_alpha2_Peak",
    "HRV_MFDFA_alpha2_Mean", "HRV_MFDFA_alpha2_Max", "HRV_MFDFA_alpha2_Delta",
    "HRV_MFDFA_alpha2_Asymmetry", "HRV_MFDFA_alpha2_Fluctuation", "HRV_MFDFA_alpha2_Increment",
    "ApEn", "SampEn", "HRV_ShanEn", "HRV_FuzzyEn", "HRV_MSEn", "HRV_CMSEn", "HRV_RCMSEn",
    "HRV_CD", "HRV_HFD", "HRV_KFD", "HRV_LZC", "HRV_Symbolic_EqualProb4_0V",
    "HRV_Symbolic_EqualProb4_1V", "HRV_Symbolic_EqualProb4_2LV", "HRV_Symbolic_EqualProb4_2UV",
)
# Its Group sheet (Compare segments → Add to group table), without the tag.
SEGMENT_206 = (
    "Beats", "Duration (s)", "Blocks used (n)", "HR, mean of block means (bpm)",
    "HR, SD of block means (bpm)", "Mean HR (60000/mean NN) (bpm)", "Instantaneous HR SD (bpm)",
    "Min inst. HR (2nd %ile) (bpm)", "Max inst. HR (98th %ile) (bpm)", "Mean RR (ms)",
    "RR SD (ms)", "RR CV (%)", "SDNN (ms)", "RMSSD (ms)", "pNN6 (%)", "LF power (ms²)",
    "HF power (ms²)", "LF (n.u.) (%)", "HF (n.u.) (%)", "LF/HF", "Poincaré SD1 (ms)",
    "Poincaré SD2 (ms)", "SD1/SD2",
)
# Event-locked series keys (ecg.ui.event_locked_view) and the tests' rCSI / rCPI.
EVENT_KEYS = ("hr", "d", "sd1", "sd2", "csi", "cpi", "lf", "hf", "rCSI", "rCPI")


def test_segment_metrics_fall_in_their_families():
    got = dict(mf.by_family(SEGMENT_206))
    assert list(got) == [mf.HEART_RATE, mf.TIME_DOMAIN, mf.FREQUENCY, mf.NONLINEAR, mf.DATA]
    assert got[mf.HEART_RATE] == list(SEGMENT_206[3:12])
    # How much data a value rests on, listed last (not an outcome).
    assert got[mf.DATA] == ["Beats", "Duration (s)", "Blocks used (n)"]
    assert got[mf.TIME_DOMAIN] == ["SDNN (ms)", "RMSSD (ms)", "pNN6 (%)"]
    assert got[mf.FREQUENCY] == ["LF power (ms²)", "HF power (ms²)", "LF (n.u.) (%)",
                                 "HF (n.u.) (%)", "LF/HF"]
    assert got[mf.NONLINEAR] == ["Poincaré SD1 (ms)", "Poincaré SD2 (ms)", "SD1/SD2"]


def test_every_whole_recording_metric_of_206_has_a_family():
    fams = {n: mf.family_of(n) for n in WHOLE_206}
    assert not [n for n, f in fams.items() if f == mf.OTHER]
    for name, fam in {
            "HR mean (60000 / mean NN) (bpm)": mf.HEART_RATE,
            "RR mean (ms)": mf.HEART_RATE, "Analysed duration (s)": mf.DATA,
            "Beats analysed (beats)": mf.DATA, "Valid RR intervals (intervals)": mf.DATA,
            "RR intervals (intervals)": mf.DATA, "Beats below 0.90 (beats)": mf.QUALITY,
            "RR excluded (%)": mf.QUALITY, "RR lower bound (ms)": mf.QUALITY,
            "Mean beat correlation": mf.QUALITY, "Beats below 0.90 (%)": mf.QUALITY,
            "MeanNN (ms)": mf.TIME_DOMAIN, "HRV_SDANN1 (ms)": mf.TIME_DOMAIN,
            "pNN50 (human) (%)": mf.TIME_DOMAIN, "HRV_TINN (ms)": mf.TIME_DOMAIN,
            "HRV_TP (ms²)": mf.FREQUENCY, "LF (n.u.) (fraction of LF+HF)": mf.FREQUENCY,
            "HRV_HFpeak (Hz)": mf.FREQUENCY,
            "Porta index (PI) (%)": mf.ASYMMETRY, "Pairs (pairs)": mf.ASYMMETRY,
            "SD1 (ms)": mf.NONLINEAR, "DFA α1": mf.NONLINEAR, "SampEn": mf.NONLINEAR,
            # NeuroKit's CSI is a non-linear column, not Autonomic; its
            # asymmetry indices sit with the app's own Porta / Guzik rows.
            "HRV_CSI": mf.NONLINEAR, "HRV_PIP": mf.NONLINEAR, "HRV_SD1SD2": mf.NONLINEAR,
            "HRV_MFDFA_alpha1_Width": mf.NONLINEAR}.items():
        assert fams[name] == fam, name


def test_each_asymmetry_index_has_one_family():
    """Guzik / Porta come both as the app's rows and as NeuroKit columns (the
    same numbers): one family, RR asymmetry, for both."""
    nk = ("HRV_GI", "HRV_SI", "HRV_AI", "HRV_PI", "HRV_C1d", "HRV_C1a", "HRV_SD1d", "HRV_SD1a",
          "HRV_C2d", "HRV_C2a", "HRV_SD2d", "HRV_SD2a", "HRV_Cd", "HRV_Ca", "HRV_SDNNd",
          "HRV_SDNNa")
    assert {mf.family_of(n) for n in nk} == {mf.ASYMMETRY}
    assert mf.family_of("Porta index (PI) (%)") == mf.family_of("HRV_PI") == mf.ASYMMETRY
    assert mf.family_of("Guzik index (GI) (%)") == mf.family_of("HRV_GI (%)") == mf.ASYMMETRY
    # Not every column starting like one: SD1, SD2, SD1/SD2, CSI stay non-linear.
    assert {mf.family_of(n) for n in ("HRV_SD1", "HRV_SD2", "HRV_SD1SD2", "HRV_CSI",
                                      "HRV_CVI", "HRV_PAS", "HRV_PSS")} == {mf.NONLINEAR}


def test_event_keys_time_course_and_chooser_names():
    assert [mf.family_of(k) for k in EVENT_KEYS] == [
        mf.HEART_RATE, mf.AUTONOMIC, mf.NONLINEAR, mf.NONLINEAR, mf.AUTONOMIC, mf.AUTONOMIC,
        mf.FREQUENCY, mf.FREQUENCY, mf.AUTONOMIC, mf.AUTONOMIC]
    assert [mf.family_of(k) for k in gd.TC_METRICS] == [
        mf.HEART_RATE, mf.HEART_RATE, mf.TIME_DOMAIN, mf.TIME_DOMAIN]
    # The Group tab's figure menu: a pair suffix and the group Poincaré.
    assert mf.family_of("SDNN (ms) · Post vs Pre") == mf.TIME_DOMAIN
    assert mf.family_of("Poincaré (group) · B vs A") == mf.NONLINEAR
    assert mf.family_of("") == mf.OTHER and mf.family_of("Weight (g)") == mf.OTHER


@pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")
def test_a_name_gets_the_family_of_its_metric_table_row():
    """Every row of a full metric table (HR, HRV, intervals, quality,
    artifacts, premature beats, Autonomic): the family worked out from the
    column name the group table keeps is the row's own."""
    sig, fs, rp = make_synth_ecg(n_beats=120, seed=3)
    r = analyse_core(sig, rp, fs)
    r["hrv_freq"] = analyse_hrv_freq(rp, fs)
    r["hrv_nonlin"] = analyse_hrv_nonlinear(rp, fs)
    if r["hrv_nonlin"] is None or not len(r["hrv_nonlin"]):
        # A regular synthetic rhythm can leave NeuroKit's non-linear frame
        # empty: its columns, as the real recording has them.
        r["hrv_nonlin"] = pd.DataFrame([{c: 1.0 for c in (
            "HRV_SD1", "HRV_SD2", "HRV_SD1SD2", "HRV_CSI", "HRV_SDNNd", "HRV_SampEn",
            "HRV_ApEn", "HRV_DFA_alpha1", "HRV_MFDFA_alpha1_Width")}])
    n = 60
    r["intervals"] = pd.DataFrame({c: np.full(n, v) for c, v in
                                   (("PR_ms", 40.0), ("QRS_ms", 10.0), ("QT_ms", 50.0),
                                    ("QTc_ms", 52.0))})
    t = build_metric_table(
        r, sig_quality=90,
        artifact_report={"n_in": 10, "n_out": 8, "n_duplicate": 1, "n_nonphysio": 1,
                         "n_ectopic": 0},
        events=[SimpleNamespace(kind="bradycardia"), SimpleNamespace(kind="premature_beat")],
        premature={"n_premature": 1, "n_isolated": 1, "n_couplets": 0, "n_runs": 0,
                   "n_beats_in_runs": 0, "premature_per_h": 1.0, "isolated_per_h": 1.0,
                   "couplets_per_h": 0.0, "runs_per_h": 0.0, "span_s": 10.0},
        autonomic=SimpleNamespace(sd01=1.0, sd02=2.0, d0=3.0, params={"n_pairs_total": 5}))
    rows = [row for row in t.rows if row.group != "Status"]
    groups = {row.group for row in rows}
    assert {"Heart rate", "HRV time domain", "HRV frequency domain", "HRV non-linear",
            "Intervals",
            "Signal quality", "Artifact correction", "Abnormal events", "Premature beats",
            "Autonomic", "RR asymmetry"} <= groups
    wrong = []
    for row in rows:
        name = f"{row.label} ({row.unit})" if row.unit else row.label
        if mf.family_of(name) != mf.family_of_row(row):
            wrong.append((row.key, name, mf.family_of(name), mf.family_of_row(row)))
    assert not wrong
    # The Whole sheet's columns are those names.
    whole = gd.whole_frame(rows)
    assert all(mf.family_of(c) != mf.OTHER for c in whole.columns)


def test_by_family_keeps_the_order_inside_and_lists_families_in_order():
    names = ["SD1 (ms)", "Mean HR (bpm)", "LF/HF", "RMSSD (ms)", "SDNN (ms)", "QRS (ms)",
             "Weight (g)", "Signal quality score (%)"]
    assert mf.by_family(names) == [
        (mf.HEART_RATE, ["Mean HR (bpm)"]), (mf.TIME_DOMAIN, ["RMSSD (ms)", "SDNN (ms)"]),
        (mf.FREQUENCY, ["LF/HF"]), (mf.NONLINEAR, ["SD1 (ms)"]), (mf.INTERVALS, ["QRS (ms)"]),
        (mf.QUALITY, ["Signal quality score (%)"]), (mf.OTHER, ["Weight (g)"])]
    assert mf.families_in(names)[0] == mf.HEART_RATE
    # Any item, with its own key.
    triples = [("sd1", "Poincaré SD1", "ms"), ("hr_mean", "Mean HR", "bpm")]
    assert mf.by_family(triples, lambda m: mf.family_of(m[1])) == [
        (mf.HEART_RATE, [triples[1]]), (mf.NONLINEAR, [triples[0]])]
