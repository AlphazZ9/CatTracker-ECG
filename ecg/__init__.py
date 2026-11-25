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
    session_service.py  — the recording's side of a session (save / restore)
    agreement_service.py — Check agreement (the other detectors' cross-check)
    annotation_suggest_service.py — annotations from the file's channels
    recording_fields.py — Subject ID / sex from the file name
  io/
    loaders.py       — .mat / HDF5 signal loading
    session.py       — session save/load (JSON)
    export.py        — Excel (openpyxl) and Prism exporters
  figures/           — figures drawn on a Matplotlib Figure (Cohort Trends)
  app/               — the application layer, no toolkit: state, controllers,
                       figures, freshness, theme of the figures; message boxes
                       through dialogs.py (installed by the interface)
  qt/                — the interface (PySide6 + PyQtGraph):  python -m ecg
"""

__version__ = "7.1.0"
