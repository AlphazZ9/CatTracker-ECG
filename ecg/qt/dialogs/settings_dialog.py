# -*- coding: utf-8 -*-
"""
ecg.qt.dialogs.settings_dialog
------------------------------
Settings of a recording: channel, sampling rate (when the file has none),
time crop, filters, RR bounds, polarity, artifact correction, automatic
epochs and the SG + Derivative options.

The fields are published on the window's FigureHost under the Tk
interface's widget names (ent_lp, sw_notch, ent_minrr...), so the
parameters are read by the same function as there:
FilterParams.from_widgets (ecg.core.models). Defaults are the Tk ones.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtWidgets

from ecg.core.models import MouseECG
from ecg.qt.figure_host import Field

if TYPE_CHECKING:
    from ecg.qt.main_window import MainWindow

CLEAN_METHODS = ("neurokit", "pantompkins1985", "elgendi2010", "hamilton2002", "biosppy")


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.setWindowTitle("Recording settings")
        host = window.figures
        form = QtWidgets.QFormLayout(self)

        def num(value: float, lo: float, hi: float, decimals: int = 1,
                step: float = 1.0) -> QtWidgets.QDoubleSpinBox:
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(decimals)
            sp.setSingleStep(step)
            sp.setValue(value)
            return sp

        self.ed_channel = QtWidgets.QLineEdit("ECG")
        form.addRow("Channel", self.ed_channel)
        self.sp_fs = num(MouseECG.FS_DEFAULT, 1, 1_000_000, 0, 100)
        form.addRow("Sampling rate (Hz, if not in the file)", self.sp_fs)
        self.sp_t0 = num(0.0, 0, 1e7, 1)
        self.sp_t1 = num(0.0, 0, 1e7, 1)
        form.addRow("Crop: start (s, 0 = beginning)", self.sp_t0)
        form.addRow("Crop: end (s, 0 = end)", self.sp_t1)
        form.addRow(QtWidgets.QLabel("<b>Filtering</b> (the Filtering box of Signal & Detection)"))
        self.sp_lp = num(MouseECG.BP_LO_HZ, 0.01, 1000, 2, 0.5)
        self.sp_hp = num(MouseECG.BP_HI_HZ, 1, 10000, 1, 10)
        form.addRow("Band-pass low cut-off (Hz)", self.sp_lp)
        form.addRow("Band-pass high cut-off (Hz)", self.sp_hp)
        self.chk_notch = QtWidgets.QCheckBox("50 Hz notch")
        form.addRow("", self.chk_notch)
        self.cb_clean = QtWidgets.QComboBox()
        self.cb_clean.addItems(CLEAN_METHODS)
        form.addRow("NeuroKit2 cleaning", self.cb_clean)
        form.addRow(QtWidgets.QLabel("<b>Detection and HRV</b>"))
        self.sp_minrr = num(MouseECG.MIN_RR_MS, 1, 5000, 1)
        self.sp_maxrr = num(MouseECG.RR_MAX_MS, 1, 10000, 1)
        form.addRow("Min R-R (ms)", self.sp_minrr)
        form.addRow("Max R-R for HRV (ms)", self.sp_maxrr)
        self.chk_invert = QtWidgets.QCheckBox("Invert the signal before detection")
        form.addRow("", self.chk_invert)
        self.chk_artifact = QtWidgets.QCheckBox("Automatic artifact correction on Analyze")
        form.addRow("", self.chk_artifact)
        self.chk_epochs = QtWidgets.QCheckBox("Compute epochs after Analyze")
        form.addRow("", self.chk_epochs)
        form.addRow(QtWidgets.QLabel("<b>SG + Derivative</b>"))
        self.sp_sg_win = num(20.0, 1, 200, 1)
        self.sp_sg_fs = num(10_000, 100, 100_000, 0, 1000)
        form.addRow("SG window (ms)", self.sp_sg_win)
        form.addRow("Resampling (Hz)", self.sp_sg_fs)
        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.hide)
        reset = close.addButton("Reset to mouse ECG defaults",
                                QtWidgets.QDialogButtonBox.ButtonRole.ResetRole)
        reset.clicked.connect(self.reset_defaults)
        form.addRow(close)

        self.win = window
        det = window.detection
        # The Tk widget names FilterParams.from_widgets, the controllers and
        # the session restore read and write.
        host.sw_filtering = Field(det.chk_filter)
        host.cb_det_method = Field(det.cb_method, data=True)
        host.sl_thr = Field(det.sp_thr)
        host.ent_thr = Field(det.sp_thr)
        host.ent_channel = Field(self.ed_channel)
        host.ent_fs = Field(self.sp_fs)
        host.ent_t_start = Field(self.sp_t0)
        host.ent_t_end = Field(self.sp_t1)
        host.ent_lp = Field(self.sp_lp)
        host.ent_hp = Field(self.sp_hp)
        host.sw_notch = Field(self.chk_notch)
        host.cb_clean = Field(self.cb_clean)
        host.ent_minrr = Field(self.sp_minrr)
        host.ent_maxrr = Field(self.sp_maxrr)
        host.sw_invert_signal = Field(self.chk_invert)
        host.sw_artifact = Field(self.chk_artifact)
        host.sw_epoch = Field(self.chk_epochs)
        host.ent_sg_target_fs = Field(self.sp_sg_fs)
        host.ent_sg_window_ms = Field(self.sp_sg_win, text=True)

    def reset_defaults(self) -> None:
        """ECGApp._reset_params: channel, sampling rate, crop, band-pass, RR
        bounds and the SG + Derivative options back to FilterParams'
        defaults (the switches stay as they are, as in Tk)."""
        from ecg.core.models import FilterParams
        fp = FilterParams()
        self.ed_channel.setText(fp.channel)
        self.sp_fs.setValue(float(fp.fs))
        self.sp_t0.setValue(float(fp.t_start))
        self.sp_t1.setValue(float(fp.t_end))
        self.sp_lp.setValue(float(fp.lp))
        self.sp_hp.setValue(float(int(fp.hp)))
        self.sp_minrr.setValue(float(fp.min_rr_ms))
        self.sp_maxrr.setValue(float(fp.max_rr_ms))
        self.sp_sg_fs.setValue(float(fp.sg_target_fs))
        self.sp_sg_win.setValue(20.0)
