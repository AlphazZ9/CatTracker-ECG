"""
ECG Analysis — mouse cardiac electrophysiology toolkit.

Package layout
--------------
ecg/
  core/
    models.py        — physiology constants, dataclasses, typed dicts
    recording.py     — state of one recording (signal, detection, analysis)
    filtering.py     — bandpass, notch, normalise, downsample helpers
    pipeline.py      — signal preparation and R-peak candidates (Detect Peaks)
    peak_editing.py  — accepted peaks after an edit, signal-quality score
    windowed_hrv.py  — HRV per epoch and in a sliding window
    detection.py     — peak detection, RR artifact handling, rhythm /
                       abnormal-event classification
    analysis.py      — HRV time/freq/nonlinear, interval delineation
    wave_template.py — beat-template calibration and wave delineation
    ml_detector.py   — classical-ML R-peak detector, trainable from
                       reviewed recordings
  services/
    analysis_service.py — Analyze without the UI (artifact correction,
                       analysis window, analyse_core)
    events_service.py   — abnormal-event classification (Events tab)
    autonomic_service.py — CSI / CPI and SPWVD LF / HF (Autonomic view)
  io/
    loaders.py       — .mat / HDF5 signal loading
    session.py       — session save/load (JSON)
    export.py        — Excel (openpyxl) and Prism exporters
  ui/
    theme.py         — colour/font globals, ThemeConfig, apply_theme_config
    plots.py         — style_axes, CanvasSlot (matplotlib ↔ Tk bridge)
    dialogs.py       — ThemeDialog, ArtifactReviewDialog, Annotation dialogs
    wave_editor.py   — WaveTemplateMiniEditor
    popover.py       — Popover (Recording sheet, Filters, ⋯ menu, Report > More)
    app.py           — ECGApp (main CTk window)
"""

__version__ = "6.2.0"
