# -*- coding: utf-8 -*-
"""
ecg.qt.session
--------------
Save Session / restore a saved session in the Qt interface.

The session file is written and read by the Tk interface's own code, so a
session saved by one interface opens in the other:

* what is saved: SessionController.collect_session_state, run on the
  window's FigureHost (which publishes the Qt inputs under the Tk widget
  names), then save_session_worker in a background job (session file,
  registry, ML training cache);
* what is restored: ecg.services.session_service.restore_bundle in a
  background job (reload, crop, prepare_signal with the saved parameters),
  then QtSession.apply -- SessionController.on_restore_session_done's
  writes, in the same order, on the Qt state and inputs; the peaks are
  re-derived from the candidates with the saved threshold and hand edits,
  and the saved artifact removals re-applied.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from PySide6 import QtWidgets

from ecg.core.analysis import normalize_ensemble_beats, recompute_qtc_column, ENSEMBLE_BEATS_DEFAULT
from ecg.core.models import normalize_qtc_formula
from ecg.io.session import load_session
from ecg.qt.figure_host import Value
from ecg.services.session_service import restore_bundle
from ecg.app.state import SessionState
from ecg.app.tab_keys import TAB_ORDER

if TYPE_CHECKING:
    from ecg.qt.main_window import MainWindow

log = logging.getLogger("ecg")


class QtSession:
    def __init__(self, window: "MainWindow") -> None:
        from ecg.app.session_controller import SessionController
        self.win = window
        host = window.figures
        host.session = SessionState()
        # What collect_session_state reads with getattr(..., None): absent here.
        # The Recording fields Tk saves: Subject / Sex / Project are the
        # Recording sheet's (ecg.qt.dialogs.recording_dialog); the raw view
        # on, as Tk's default, so a session saved here holds the same keys.
        host.sw_show_raw = Value(lambda: True)
        for name in (
                     "lbl_session_info", "lbl_template_info",
                     "lbl_npeaks", "ent_analysis_t0", "ent_analysis_t1", "cb_qtc_formula",
                     "_show_time_mode", "ent_ensemble"):
            setattr(host, name, None)
        # Compare segments' setup (rows, block, reference, view, stim and the
        # windows compared), saved and restored as Tk's.
        host._compare_setup_snapshot = window.compare.setup_snapshot
        host.tabs = Value(lambda: TAB_ORDER[window.tabs.currentIndex()])
        self.ctrl = SessionController(host)
        self.saved_at: Optional[str] = None

    # ── Save Session ────────────────────────────────────────
    def save(self) -> None:
        win = self.win
        rec = win.recording
        if rec.signal.filepath is None:
            QtWidgets.QMessageBox.warning(win, "No file", "Open a file first.")
            return
        if rec.signal.filtered is None:
            QtWidgets.QMessageBox.warning(win, "Not ready", "Click Detect Peaks first.")
            return
        host = win.figures
        host.session.verified_for_training = self.ctrl.is_training_example(
            rec.signal.filepath, default=host.session.verified_for_training)
        state = self.ctrl.collect_session_state()
        snap = self.ctrl.save_snapshot()
        gen = win.generation
        win.run_job("Save Session", self.ctrl.save_session_worker, state, rec.signal.filepath,
                    rec.signal.filtered, "", snap, progress=False,
                    on_done=lambda result: self._saved(result, gen))

    def save_now(self) -> bool:
        """Save Session on the main thread (the quit / open prompt, which
        cannot wait for a job). True when saved."""
        rec = self.win.recording
        if rec.signal.filepath is None or rec.signal.filtered is None:
            return True
        try:
            state = self.ctrl.collect_session_state()
            result = self.ctrl.save_session_worker(state, rec.signal.filepath,
                                                   rec.signal.filtered, "",
                                                   self.ctrl.save_snapshot())
        except Exception as exc:
            import traceback
            self.win.errors.report("Save Session failed", str(exc), traceback.format_exc())
            return False
        self._saved(result, self.win.generation)
        return True

    def _saved(self, result: dict, gen: int) -> None:
        name = Path(result["out_path"]).name
        if gen != self.win.generation:
            self.win.set_status(f"Session saved — {name} (the recording open before this one)")
            return
        self.win.session_dirty = False
        self.saved_at = result["saved_at"]
        self.win.set_status(f"Session saved — {name}")

    # ── restore ─────────────────────────────────────────────
    def check(self, path: str, ask: bool = True) -> None:
        """After a file is opened: a saved session for it is offered (Yes:
        restored; No: the raw signal, as Tk's try_restore_session)."""
        gen = self.win.generation
        self.win.run_job("Checking for a saved session", load_session, path, progress=False,
                         on_done=lambda state: self._checked(state, path, gen, ask))

    def _checked(self, state: "Optional[dict]", path: str, gen: int, ask: bool) -> None:
        if state is None or gen != self.win.generation:
            return
        if ask:
            answer = QtWidgets.QMessageBox.question(
                self.win, "Session found",
                "A saved analysis was found for this file:\n\n"
                f"  Saved:  {state.get('saved_at', 'unknown time')}\n"
                f"  Beats:  {state.get('n_beats', '?')}\n\n"
                "Restore it?  (No = load raw signal — click Detect Peaks to re-run)")
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        if gen != self.win.generation:
            return
        self.restore(state, path)

    def restore(self, state: dict, path: str) -> None:
        gen = self.win.generation
        saved_at = state.get("saved_at", "unknown time")
        self.win.detection.btn_detect.setEnabled(False)
        self.win.run_job("Restoring session", restore_bundle, state, path, progress=False,
                         on_done=lambda b: self.apply(b, saved_at) if gen == self.win.generation
                         else None,
                         on_end=lambda: self.win.detection.btn_detect.setEnabled(True))

    def apply(self, bundle: dict, saved_at: str) -> None:
        """SessionController.on_restore_session_done on the Qt state and inputs."""
        from ecg.app.session_controller import _TAB_KEY_MAP
        from ecg.app.tab_keys import time_course_mode
        win, host = self.win, self.win.figures
        rec = win.recording
        s, d, a = rec.signal, rec.detection, rec.analysis
        host.ui.figure_cache.clear()
        bundle["current_tab"] = _TAB_KEY_MAP.get(bundle["current_tab"], bundle["current_tab"])

        fs, n_samples = bundle["fs"], bundle["n_samples"]
        s.fs = fs
        s.raw = bundle["signal_raw"]
        s.channel = bundle.get("detected_ch")
        s.time = np.arange(n_samples) / fs
        sb = bundle["signal_bundle"]
        s.raw_norm = sb["signal_raw_norm"]
        s.filtered = sb["signal_flt"]
        d.all_candidates = sb["all_cands"]
        d.all_prominences = sb["all_proms"]
        s.filter_text = sb.get("filter_text")
        s.prepare_params = sb.get("prepare_params")
        s.crop = bundle.get("signal_crop")
        s.markers = bundle.get("markers") or {}
        d.thresh_amp = bundle["thresh_amp"]
        d.manual_excluded = set(bundle["manual_excluded"])
        d.manual_added = set(bundle["manual_added"])
        d.edit_undo.clear()
        d.edit_redo.clear()
        s.inverted = bundle["signal_inverted"]
        s.no_filter_mode = bundle["no_filter_mode"]
        host.session.verified_for_training = self.ctrl.is_training_example(
            s.filepath, default=bundle["verified_for_training"])

        a.results = bundle["results"]
        a.artifact_report = bundle["artifact_report"]
        a.artifact_candidates = bundle["artifact_candidates"]
        a.epoch_df = bundle["epoch_df"]
        a.rolling_hrv_df = bundle["rolling_hrv_df"]
        a.windowed_view = dict(bundle.get("windowed_view") or {})
        a.annotations = list(bundle["annotations"])
        a.noise_regions = list(bundle["noise_regions"])
        a.wave_template = bundle.get("wave_template")
        host.analysis_ctrl.invalidate_autonomic(peaks_changed=False)
        host.analysis_ctrl.apply_autonomic_params(bundle.get("autonomic_params"))
        host.event_ctrl.apply_params(bundle.get("event_locked_params"))
        a.t_start = bundle["analysis_t_start"]
        a.t_end = bundle["analysis_t_end"]
        win.detection.show_window_in_entries()
        a.analysed_window = bundle.get("analysed_window") if bundle["results"] is not None else None

        formula = normalize_qtc_formula(bundle.get("qtc_formula", "mitchell"))
        a.qtc_formula = formula
        cb = win.morphology.cb_qtc
        cb.blockSignals(True)
        cb.setCurrentIndex(max(cb.findData(formula), 0))
        cb.blockSignals(False)
        if bundle.get("ensemble_beats") is not None:
            n = normalize_ensemble_beats(bundle["ensemble_beats"], ENSEMBLE_BEATS_DEFAULT)
            a.ensemble_beats = n
            win.morphology.sp_ensemble.setValue(n)
        if a.results is not None and a.results.get("intervals") is not None:
            try:
                a.results["intervals"] = recompute_qtc_column(a.results["intervals"], formula)
            except Exception as exc:
                log.debug("restore QTc recompute failed: %s", exc, exc_info=True)

        det = win.detection
        det.btn_edit.setEnabled(True)
        det.btn_mark_noise.setEnabled(True)
        det.btn_review.setEnabled(True)
        det.btn_agreement.setEnabled(True)
        det.btn_edit.setChecked(bool(bundle["edit_mode"]))
        thr = float(bundle["thresh_frac"])
        for w in (det.sp_thr, det.sl_thr):
            w.blockSignals(True)
        det.sp_thr.setValue(thr)
        det.sl_thr.setValue(int(round(thr * 100)))
        for w in (det.sp_thr, det.sl_thr):
            w.blockSignals(False)

        fp = bundle.get("filter_params")
        if fp is not None:
            st = win.settings
            st.sp_lp.setValue(fp.lp)
            st.sp_hp.setValue(fp.hp)
            st.sp_minrr.setValue(fp.min_rr_ms)
            st.sp_maxrr.setValue(fp.max_rr_ms)
            st.chk_notch.setChecked(fp.notch_filter)
            det.chk_filter.setChecked(not fp.no_filter)
            st.chk_invert.setChecked(fp.invert_signal)
            host.cb_clean.set(fp.clean_method)
            self.ctrl.restore_detection_widgets(fp, bundle.get("filter_keys") or ())
        fields = bundle.get("recording_fields")
        if isinstance(fields, dict) and isinstance(fields.get("sg_window_ms"), (str, float, int)):
            try:
                win.settings.sp_sg_win.setValue(float(fields["sg_window_ms"]))
            except ValueError:
                pass
        if isinstance(fields, dict):
            # Subject, sex and project (SessionController.apply_recording_fields:
            # a typed value is kept over the file name's; the project only
            # fills an empty field).
            self.ctrl.apply_recording_fields({k: v for k, v in fields.items()
                                              if k in ("subject", "sex", "project_name")})
        win.recording_sheet.notes_restored()
        win.update_file_label()

        # The plot and the peaks: re-derived from the candidates (DetectionController
        # .run_detection), then the saved artifact removals re-applied.
        det.plot.set_signal(s.filtered, fs, s.time)
        det.plot.set_noise_regions(a.noise_regions)
        det.plot.set_annotations(a.annotations)
        det.apply_threshold()
        n_reapplied = host.analysis_ctrl.reapply_artifact_removals()
        if n_reapplied:
            det.refresh()
        quality_note = self.ctrl.restore_beat_quality_times(bundle)
        host.freshness.stamp_restored(bundle.get("freshness_basis"))
        ui = host.ui
        ui.computed_at = bundle.get("computed_at")
        ui.peaks_edited_at = bundle.get("peaks_edited_at")
        include = bundle.get("report_include")
        ui.report_include = dict(include) if isinstance(include, dict) else {}
        det.update_period_counts()
        det._update_undo_buttons()
        win.peaks_changed()
        win.btn_analyze.setEnabled(True)

        # Time course: the ticks and frames saved, on the mode saved.
        self.ctrl.restore_window_ticks(bundle)
        host.analysis_ctrl.show_epochs()
        host.analysis_ctrl.show_rolling()
        mode = time_course_mode(bundle.get("time_course_mode"), None, None)
        pages = [win.time_course.tabText(i) for i in range(win.time_course.count())]
        if mode in pages:
            win.time_course.setCurrentIndex(pages.index(mode))
        if bundle["current_tab"] in TAB_ORDER:
            win.tabs.setCurrentIndex(TAB_ORDER.index(bundle["current_tab"]))

        n_beats = len(d.rpeaks_ok) if d.rpeaks_ok is not None else 0
        legacy = "; ".join(list(bundle.get("legacy_notes") or [])
                           + ([quality_note] if quality_note else []))
        status = (f"Session restored — {n_beats} beats  |  {n_samples / fs:.0f} s recording"
                  + ("  |  results ready" if a.results is not None else "")
                  + (f"  |  older session: {legacy}" if legacy else ""))
        if a.results is not None:
            win.results_restored()
            host.analysis_ctrl.refresh_result_labels(freq_done_label="Loaded from session ✓")
        # Compare segments: the setup saved, its comparison re-run on the
        # restored peaks (SessionController's restore_cmp).
        try:
            win.compare.restore_setup(bundle.get("compare_segments"))
        except Exception as exc:
            log.warning("Compare segments setup not restored: %s", exc, exc_info=True)
        win.session_dirty = False
        self.saved_at = saved_at
        win.set_status(status, warn=bool(legacy))
