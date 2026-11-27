"""The charts on screen (PyQtGraph) and the figures of the exports
(Matplotlib) are drawn by two pieces of code: they must show the same data.
Layout, colours and scales may differ; the numbers may not (audit G7).

One Analyze on the synthetic recording, then each pair compared point by
point: heart rate / RR over time, Poincaré, multiscale entropy, power
spectrum, RR distribution, quality panel, RR asymmetry, beat distributions.
"""
from pathlib import Path

import numpy as np
import pytest

from ecg.qt.main_window import MainWindow

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"


@pytest.fixture(scope="module")
def analysed(qapp):
    from pytestqt.qtbot import QtBot
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    bot = QtBot(qapp)
    w.open_path(str(FIXTURE))
    w.detection.detect()
    bot.waitUntil(lambda: w.recording.detection.rpeaks_ok is not None, timeout=60000)
    w.analyze()
    bot.waitUntil(lambda: w.statusBar().currentMessage() == "Analysis done", timeout=120000)
    w.tabs.setCurrentWidget(w.rhythm)
    bot.wait(300)
    yield w
    w.jobs.cancel_all()
    w.jobs.wait(5000)
    w.hide()                 # not close(): it would ask whether to discard the analysis
    w.deleteLater()


def _finite_pairs(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _same(a, b, what):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, f"{what}: {a.shape} on screen, {b.shape} in the export"
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9, err_msg=what)


def test_heart_rate_and_rr_over_time(analysed):
    ch = analysed.rhythm.charts
    fig = analysed.figures.slot("rr").figure
    ax_rr, ax_hr = fig.axes[:2]
    t, rr, hr, valid = ch._series()
    # The export's main lines: one point per beat, invalid ones as gaps.
    ex_t, ex_rr = ax_rr.get_lines()[0].get_data()
    _, ex_hr = ax_hr.get_lines()[0].get_data()
    _same(t, ex_t, "beat times")
    for name, screen, export in (("RR", rr, ex_rr), ("HR", hr, ex_hr)):
        s_t, s_v = _finite_pairs(t[valid], screen[valid])
        e_t, e_v = _finite_pairs(ex_t, export)
        assert set(np.round(s_t, 9)) <= set(np.round(e_t, 9)), name
        idx = np.searchsorted(e_t, s_t)
        _same(s_v, np.asarray(e_v)[idx], f"{name} of the valid beats")
    # What the screen draws as its line is that same HR.
    _, line = _finite_pairs(*ch.t_line.getOriginalDataset())   # before downsampling
    _same(line, hr[valid & np.isfinite(hr)], "HR line on screen")


def test_poincare_and_multiscale_entropy(analysed):
    ch = analysed.rhythm.charts
    fig = analysed.figures.slot("poincare").figure
    ax_p, ax_mse = fig.axes[:2]
    pts = np.asarray(ax_p.collections[0].get_offsets(), float)
    screen = np.column_stack([ch.p_points.data["x"], ch.p_points.data["y"]])
    # The export spreads the pairs that the sampling grid stacks (jitter of
    # at most half a step, said on the figure); the pairs are the same ones,
    # in the same order.
    from ecg.app.poincare_figure import jitter_for_display
    _jx, _jy, step = jitter_for_display(screen[:, 0], screen[:, 1])
    assert screen.shape == pts.shape
    assert np.max(np.abs(screen - pts)) <= step / 2 + 1e-9
    sx, sy = ch.m_line.getOriginalDataset()
    ex, ey = ax_mse.get_lines()[0].get_data()
    _same(sx, ex, "MSE scales")
    _same(sy, ey, "MSE values")


def test_power_spectrum(analysed):
    ch = analysed.rhythm.charts
    ax = analysed.figures.slot("psd").figure.axes[0]
    ef, ep = (np.asarray(v, float) for v in ax.get_lines()[0].get_data())
    # The data before the log axis (getData returns it as log10).
    sf, sp = (np.asarray(v, float) for v in ch.s_line.getOriginalDataset())
    # The screen leaves out f = 0 and non-positive power (log axis).
    keep = (ef > 0) & np.isfinite(ep) & (ep > 0)
    _same(sf, ef[keep], "PSD frequencies")
    _same(sp, ep[keep], "PSD power")


def test_rr_distribution_counts_the_same_beats(analysed):
    ch = analysed.rhythm.charts
    ax = analysed.figures.slot("rr_hist").figure.axes[0]
    n_export = sum(int(round(p.get_height())) for p in ax.patches)
    heights = ch.h_bars.opts["height"]
    assert int(np.sum(heights)) == n_export
    _, rr, _, valid = ch._series()
    assert n_export == int(valid.sum())


def test_quality_asymmetry_and_beat_distributions(analysed):
    host = analysed.figures
    q = analysed.detection.quality_charts._series
    ax_t = host.slot("quality_time").figure.axes[0]
    pts = np.asarray(ax_t.collections[0].get_offsets())
    _same(np.column_stack([q["t"], q["v"]]), pts, "beat correlation over time")
    ax_h = host.slot("quality_hist").figure.axes[0]
    assert sum(p.get_height() for p in ax_h.patches) == len(q["all"])

    d = analysed.rhythm.asym_charts._data
    ax_bar, ax_dh = host.slot("asymmetry").figure.axes[:2]
    bars = [p.get_height() for p in ax_bar.patches]
    shares = [d["n_dec"] / d["n_off"] * 100, d["n_acc"] / d["n_off"] * 100]
    _same(bars, shares, "deceleration / acceleration shares")
    assert sum(p.get_height() for p in ax_dh.patches) == int(np.sum(d["drr"] != 0))

    b = analysed.morphology.dist_charts._data
    ax_amp, ax_corr = host.slot("beat_dist").figure.axes[:2]
    assert sum(p.get_height() for p in ax_amp.patches) == int(np.isfinite(b["peak_amps"]).sum())
    assert sum(p.get_height() for p in ax_corr.patches) == int(np.isfinite(b["beat_corr"]).sum())
