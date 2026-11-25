# -*- coding: utf-8 -*-
"""
detection_controller.py
-------------------------
DetectionController -- applying the amplitude threshold to peak candidates,
manual peak edit mode (click-to-exclude/add), undo/redo, hover preview, and
signal-quality scoring.
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
from tkinter import messagebox
from typing import TYPE_CHECKING

import numpy as np

import customtkinter as ctk  # type: ignore[import-untyped]

from ecg.core.models import MouseECG
from ecg.core.peak_editing import (apply_click_edit, apply_peak_edits, edit_at_click,
                                   push_undo, signal_quality_score, undo_redo)
from ecg.core.ml_detector import MLPeakModel
from ecg.services.agreement_service import method_agreement
from ecg.ui.ml_examples import NO_MODEL_STATUS
from ecg.ui.theme import (
    BLUE, BORDER, BORDER2, GREEN, LIGHT, MUTED, ORANGE, ORANGE_DEEP, RED,
)
from ecg.ui.theme import toggle_style
from ecg.ui.dialogs import MethodAgreementDialog

if TYPE_CHECKING:
    from ecg.ui.app import ECGApp

log = logging.getLogger("ecg")


class DetectionController:
    def __init__(self, app: "ECGApp") -> None:
        self.app = app

    def update_undo_btns(self) -> None:
        """Update the toolbar's undo/redo buttons (single global pair, shared
        by the Detection and Events tabs)."""
        n_u, n_r = len(self.app.detection.edit_undo), len(self.app.detection.edit_redo)
        for attr, n in [("btn_toolbar_undo", n_u), ("btn_toolbar_redo", n_r)]:
            try:
                btn = getattr(self.app, attr)
                key = "undo" if "undo" in attr else "redo"
                sym = "↩" if key == "undo" else "↪"
                self.app._set_btn_enabled(btn, bool(n), "secondary")
                btn.configure(
                    text=f"{sym} {key.title()} ({n})" if n else f"{sym} {key.title()}")
            except Exception as e:
                log.debug("undo/redo button configure failed: %s", e)

    def run_detection(self, thresh: float | None = None) -> int:
        """Apply current threshold to pre-computed candidates.

        Parameters
        ----------
        thresh : float | None
            If supplied, use this value directly (safe from a background
            thread).  If None, read ``self.app.sl_thr`` — only safe to call
            from the main thread.

        Returns the number of accepted peaks.  Fast — no signal processing.
        All Tkinter widget *writes* are marshalled through ``after(0, …)``
        so this method is safe to call from either thread.
        """
        if self.app.signal.filtered is None or self.app.detection.all_candidates is None or self.app.detection.all_prominences is None:
            return 0

        if thresh is None:
            thresh = float(self.app.sl_thr.get())   # main-thread-only path  # type: ignore[union-attr]

        det = self.app.detection
        prev_ok = det.rpeaks_ok
        # Artifact removals still in effect, and their matching tolerance
        # (ecg.core.peak_editing.apply_peak_edits, step 4).
        arep = self.app.analysis.artifact_report
        removed = arep.get("removed_samples") if isinstance(arep, dict) else None
        tol = 1
        if removed and prev_ok is not None and len(prev_ok):
            actrl_tol = getattr(self.app, "analysis_ctrl", None)
            tol = (actrl_tol.break_tol_samples() if actrl_tol is not None
                   else max(1, int(round(0.005 * float(self.app.signal.fs or 1.0)))))
        edits = apply_peak_edits(
            self.app.signal.filtered, det.all_candidates, det.all_prominences, thresh,
            self.app.signal.fs, det.manual_excluded, det.manual_added,
            removed_samples=removed, previous_peaks=prev_ok, tol=tol)
        det.rpeaks_manual_excl = edits.manual_excl
        det.rpeaks_manual_added = edits.manual_added
        if det.manual_added:
            # Remove from exclusion set if mistakenly present (can't be both)
            det.manual_excluded -= det.manual_added
        accepted = edits.accepted
        det.rpeaks_ok  = accepted
        det.rpeaks_rej = edits.rejected
        thresh_amp = det.thresh_amp = edits.thresh_amp
        n = len(accepted)

        # A changed peak set makes the Autonomic result stale. The edit
        # paths drop it explicitly before re-detecting, like the core
        # results; this catches every other caller (e.g. the Abnormal
        # Events viewer's edit mode). Widget work stays on the main thread.
        actrl = getattr(self.app, "analysis_ctrl", None)
        if (actrl is not None
                and getattr(self.app.analysis, "autonomic_result", None) is not None
                and not actrl._same_peaks(prev_ok, accepted)):
            if threading.current_thread() is threading.main_thread():
                actrl.invalidate_autonomic()
            else:
                self.app.after(0, actrl.invalidate_autonomic)

        # Widget writes: always on main thread via after(0, ...)
        color = GREEN if n > 10 else RED
        self.app.after(0, lambda _n=n, _c=color: self.app.lbl_npeaks.configure(  # type: ignore[union-attr]
            text=f"Peaks detected: {_n}", text_color=_c))
        # Reflect the resulting normalised amplitude next to the fraction so
        # "Threshold: 0.50" doesn't look disconnected from the amber line on
        # the plot -- both now read the same "fraction -> norm. amplitude".
        if getattr(self.app, "lbl_thr_hint", None) is not None:
            self.app.after(0, lambda _t=thresh, _a=thresh_amp: self.app.lbl_thr_hint.configure(  # type: ignore[union-attr]
                text=f"Current: {_t:.2f} → ≈{_a:.2f} norm. of the median R-peak "
                     "height. Lower = more peaks."))
        # Enable the artifact review button as soon as peaks are available
        self.app.after(0, lambda: self.app._set_review_artifacts_enabled(n > 4))  # type: ignore[union-attr]
        if self.app.btn_check_agreement is not None:
            self.app.after(0, lambda: self.app._set_check_agreement_enabled(n > 4))  # type: ignore[union-attr]
        # Keep the HRV tab's RR/HR tachogram live -- same "no Core Analysis
        # dependency" idea as the Detection tab's own RR/HR/Quality mini-
        # strip, so adding/removing a peak (or a threshold change) doesn't
        # blank it until the next Analyze click.
        self.app.after(0, self.app.plot_ctrl.refresh_live_rr)
        self.update_signal_quality(accepted)
        return n

    def update_signal_quality(self, accepted: np.ndarray) -> None:
        """Store the 0–100 quality score of *accepted* (the KPI gauge reads
        it): ecg.core.peak_editing.signal_quality_score, on the beat-to-template
        correlation of the last analysis when there is one. Unchanged with 5
        peaks or fewer, or no signal loaded."""
        if self.app.signal.time is None:
            return
        beat_corr = None
        if self.app.analysis.results is not None:
            beat_corr = self.app.analysis.results.get("beat_corr")
        quality = signal_quality_score(accepted, self.app.signal.fs,
                                       float(self.app.signal.time[-1]), beat_corr)
        if quality is None:
            return
        self.app.detection.sig_quality = quality

    def on_det_method_change(self, choice: str) -> None:
        """Show/hide SG options frame based on selected detection method."""
        if self.app._sg_frame is not None:
            if "SG" in choice or "Derivative" in choice:
                self.app._sg_frame.pack(fill="x")
            else:
                self.app._sg_frame.pack_forget()

        # ML Detector needs a trained model before it can run. Its section in
        # the Signal quality panel is always there (examples are prepared
        # with the other methods); choosing the method without a model opens
        # it and says what to do (absent from test doubles without that panel).
        # Its status line says so while Method = ML Detector (and stops
        # saying it on another Method): refreshed on every change.
        if getattr(self.app, "lbl_ml_status", None) is not None:
            self.app.refresh_ml_status()
            if choice == "ML Detector" and not MLPeakModel.exists():
                expand = getattr(self.app, "_set_ml_expanded", None)
                if expand is not None:
                    expand(True)
                self.app._set_status(NO_MODEL_STATUS, ORANGE)

    def on_filtering_toggle(self) -> None:
        """Grey out (not hide) FILTER SETTINGS -- notch/band-pass/cleaning
        -- when Filtering is off, and refresh the "Processing" summary.

        Visible either way, so the user can see/set values in advance;
        only interaction is blocked. Distinct from the always-interactive
        DISPLAY & PREVIEW group above it (sw_show_raw/sw_invert_signal/
        sw_filter_preview), which apply regardless of this switch.
        """
        filtering_on = bool(self.app.sw_filtering.get())
        g = self.app._filter_advanced_group
        if g is not None:
            self._set_group_availability(g, filtering_on)
        self.app._update_filter_summary()

    def _set_group_availability(self, frame, available: bool) -> None:
        """Recursively enable/disable every widget under `frame`.

        CTkEntry/CTkComboBox dim automatically via state="disabled". CTkSwitch
        does NOT -- its _draw() only fades the text label, the track/button
        colours stay identical to the enabled look -- so its colours are
        explicitly swapped too: BLUE=enabled+on, BORDER2=enabled+off (the
        _switch() construction defaults), BORDER/LIGHT=unavailable, a third,
        visually distinct "can't touch this" state regardless of the
        switch's underlying on/off value. CTkLabel/CTkFrame don't support
        state= at all -- configure() failures on those are expected no-ops.
        """
        new_state = "normal" if available else "disabled"
        for widget in frame.winfo_children():
            try:
                if isinstance(widget, ctk.CTkSwitch):
                    if available:
                        widget.configure(state=new_state, progress_color=BLUE,
                                          button_color=BORDER2, text_color=MUTED)
                    else:
                        widget.configure(state=new_state, progress_color=BORDER,
                                          button_color=BORDER, text_color=LIGHT)
                else:
                    widget.configure(state=new_state)
            except Exception as e:
                log.debug("widget.configure(state) failed: %s", e)
            self._set_group_availability(widget, available)

    def on_show_raw_toggle(self) -> None:
        """Switch the overview and detail plots between raw and filtered signals.

        The raw signal is normalised (zero-mean, unit-variance) to match
        the amplitude scale of the filtered signal so that peak markers
        remain visually coherent regardless of which view is active.
        No re-processing is needed — both arrays are pre-computed.
        """
        self.app.ui.show_raw = bool(self.app.sw_show_raw.get())
        if self.app.signal.filtered is not None:
            self.app._draw_detail(self.app.ui.nav_pos)

    def mark_peaks_edited(self) -> None:
        """Note the time of a manual peak change, for the context strip."""
        ui = getattr(self.app, "ui", None)
        if ui is not None:
            ui.peaks_edited_at = dt.datetime.now().isoformat(timespec="seconds")

    def push_edit_undo(self) -> None:
        """Snapshot state before a destructive edit action."""
        self.mark_peaks_edited()
        det = self.app.detection
        push_undo(det.edit_undo, det.edit_redo, det.manual_excluded, det.manual_added,
                  limit=self.app._EDIT_UNDO_LIMIT)
        self.update_undo_btns()

    def _drop_results(self) -> None:
        """Discard the results and everything on screen derived from them.
        No-op when there are none. A plot chain still running on them stops by
        itself (run_plot_chain)."""
        if self.app.analysis.results is None:
            return
        self.app.analysis.results  = None
        self.app.analysis.epoch_df = None
        self.app._reset_kpis()
        self.app._reset_result_plots()
        self.app._reset_hrv_readouts()

    def _invalidate_stale_results(self) -> None:
        """Call right before actually committing a peak-set change.

        Rather than leaving SDNN/RMSSD/HRV tables/Poincaré/Report/every
        export showing the pre-edit numbers until the user remembers to
        click Analyze again. Must only be called on a code path that is
        actually about to change the accepted-peak set -- calling it
        unconditionally on click, before knowing whether the click will hit
        a peak, wipes results for a no-op click.
        """
        self._drop_results()
        # Outside the guard: the Autonomic view needs only peaks, so its
        # result can exist without core results.
        self.app.analysis_ctrl.invalidate_autonomic()

    def undo_edit(self, _event=None) -> None:
        """Ctrl+Z — restore previous peak-edit state."""
        if not self.app.detection.edit_undo:
            self.app._set_status("Nothing to undo.", MUTED)
            return
        det = self.app.detection
        det.manual_excluded, det.manual_added = undo_redo(
            det.edit_undo, det.edit_redo, det.manual_excluded, det.manual_added)
        self.mark_peaks_edited()
        self.apply_edit_state()
        n = len(self.app.detection.edit_undo)
        self.app._set_status(f"Undone  ({str(n) + ' left' if n else 'none'})  — Ctrl+Y to redo", ORANGE)
        self.update_undo_btns()

    def redo_edit(self, _event=None) -> None:
        """Ctrl+Y — rétablir après undo."""
        if not self.app.detection.edit_redo:
            self.app._set_status("Nothing to redo.", MUTED)
            return
        det = self.app.detection
        det.manual_excluded, det.manual_added = undo_redo(
            det.edit_redo, det.edit_undo, det.manual_excluded, det.manual_added)
        self.mark_peaks_edited()
        self.apply_edit_state()
        self.app._set_status(f"Redone  ({len(self.app.detection.edit_redo)} remaining)", ORANGE)
        self.update_undo_btns()

    def apply_edit_state(self) -> None:
        if self.app.signal.filtered is not None and self.app.detection.all_candidates is not None:
            # Undo/redo change the accepted-peak set just like a direct
            # edit does -- same stale-HRV invalidation as on_detail_click.
            self._drop_results()
            self.app.analysis_ctrl.invalidate_autonomic()
            self.run_detection(float(self.app.sl_thr.get()))  # type: ignore[union-attr]
            self.app._draw_detail(self.app.ui.nav_pos)
            # Also refresh the arrhythmia ECG viewer if an event is selected
            if self.app.analysis.arr_selected_idx >= 0 and self.app.analysis.arrhythmia_events:
                self.app._draw_arr_detail()

    def toggle_edit_mode(self) -> None:
        """Toggle the click-to-exclude edit mode on/off."""
        self.app.detection.edit_mode = not self.app.detection.edit_mode
        if self.app.detection.edit_mode:
            self.app.btn_edit_mode.configure(text="Edit Mode ON", **toggle_style(True, warn=True))
            self.app.lbl_edit_hint.pack(side="left", padx=(4, 0))  # SPACE_S in app.py
        else:
            self.app.btn_edit_mode.configure(text="Edit Peaks", **toggle_style(False))
            self.app.lbl_edit_hint.pack_forget()
            if self.app.detection.edit_noise_mode:
                self.toggle_noise_mode()
            # Also turn off free placement when leaving edit mode
            if self.app.detection.edit_free_placement:
                self.app.detection.edit_free_placement = False
                try:
                    self.app.btn_free_placement.configure(
                        text="Free Placement", **toggle_style(False))
                except Exception:
                    pass
        if self.app.signal.filtered is not None:
            self.app._draw_detail(self.app.ui.nav_pos)

    # ── Mark Noise: drag over a stretch too noisy to trust ───────────────
    NOISE_MIN_S = 0.05          # a shorter drag is a click, not a region

    def toggle_noise_mode(self) -> None:
        """Mark Noise on/off. On, a left drag on the detail view marks a noisy
        region (its RR intervals left out of HR / HRV) and a right-click on
        one removes it; peak editing waits until it is off. Turns Edit mode
        on when needed."""
        det = self.app.detection
        det.edit_noise_mode = not det.edit_noise_mode
        self._noise_drag = None
        btn = getattr(self.app, "btn_mark_noise", None)
        if det.edit_noise_mode:
            if not det.edit_mode:
                self.toggle_edit_mode()
            if btn is not None:
                btn.configure(text="Mark Noise ON", **toggle_style(True, warn=True))
            self.app._set_status(
                "Mark Noise ON — drag over a noisy stretch to leave its RR intervals out of "
                "HR / HRV; right-click a noisy region to remove it.", ORANGE)
        else:
            if btn is not None:
                btn.configure(text="Mark Noise", **toggle_style(False))
            self.app._set_status("Mark Noise OFF — clicks edit peaks again.", MUTED)

    def _noise_press(self, event) -> None:
        t = float(event.xdata)
        regions = self.app.analysis.noise_regions
        if event.button == 3:
            hit = [i for i, r in enumerate(regions)
                   if float(r["t_start"]) <= t <= float(r["t_end"])]
            if hit:
                for i in reversed(hit):
                    del regions[i]
                self.app._noise_regions_changed()
                self.app._set_status(f"Noisy region at {t:.2f} s removed.", GREEN)
            return
        if event.button != 1:
            return
        ax = event.inaxes
        span = None
        if ax is not None:
            span = ax.axvspan(t, t, facecolor="#9AA0A6", edgecolor="#5F6368", hatch="///",
                              alpha=0.35, linewidth=0, zorder=0.6)
        self._noise_drag = {"t0": t, "ax": ax, "span": span}

    def _noise_motion(self, event) -> bool:
        drag = getattr(self, "_noise_drag", None)
        if not drag or event.xdata is None or drag["span"] is None:
            return bool(drag)
        a, b = sorted((drag["t0"], float(event.xdata)))
        try:
            drag["span"].set_x(a)
            drag["span"].set_width(b - a)
            drag["ax"].figure.canvas.draw_idle()
        except Exception:
            pass
        return True

    def on_detail_release(self, event) -> None:
        drag = getattr(self, "_noise_drag", None)
        self._noise_drag = None
        if not drag:
            return
        if drag["span"] is not None:
            try:
                drag["span"].remove()
            except Exception:
                pass
        t1 = float(event.xdata) if event.xdata is not None else drag["t0"]
        a, b = sorted((drag["t0"], t1))
        if self.app.signal.fs and self.app.signal.filtered is not None:
            b = min(b, len(self.app.signal.filtered) / float(self.app.signal.fs))
        a = max(a, 0.0)
        if b - a < self.NOISE_MIN_S:
            self.app._draw_detail(self.app.ui.nav_pos)
            return
        self.app.analysis.noise_regions.append(
            {"t_start": round(a, 3), "t_end": round(b, 3), "note": ""})
        self.app.analysis.noise_regions.sort(key=lambda r: float(r["t_start"]))
        self.app._noise_regions_changed()
        self.app._set_status(
            f"Noisy region {a:.2f}–{b:.2f} s marked: its RR intervals are left out of HR / "
            "HRV at the next Analyze.", GREEN)

    def toggle_free_placement(self) -> None:
        """Toggle free-placement mode: bypass proximity constraint when adding peaks.

        When active, right-clicking adds a peak at the local max *regardless* of
        how close it is to an existing peak.  This is useful for very high-rate
        signals or for correcting closely-spaced double-peaks.

        Note: edit mode must be active for this to have any effect.
        """
        self.app.detection.edit_free_placement = not self.app.detection.edit_free_placement
        if self.app.detection.edit_free_placement:
            self.app.btn_free_placement.configure(
                text="Free Placement ON", **toggle_style(True, warn=True))
            self.app._set_status(
                "Free Placement ON — R-click places a peak at the exact click position, "
                "no snapping, no proximity guard.", ORANGE)
        else:
            self.app.btn_free_placement.configure(text="Free Placement", **toggle_style(False))
            self.app._set_status("Free Placement OFF — normal proximity guard restored.", MUTED)

    def clear_manual_exclusions(self) -> None:
        """Re-include all manually excluded peaks, remove all manually added peaks."""
        if not self.app.detection.manual_excluded and not self.app.detection.manual_added:
            return
        self.push_edit_undo()
        n_excl  = len(self.app.detection.manual_excluded)
        n_added = len(self.app.detection.manual_added)
        self.app.detection.manual_excluded.clear()
        self.app.detection.manual_added.clear()
        # Invalidate any previous analysis — peaks have changed
        self._drop_results()
        self.app.analysis_ctrl.invalidate_autonomic()
        self.app.detection.rpeaks_manual_excl  = np.array([], dtype=int)
        self.app.detection.rpeaks_manual_added = np.array([], dtype=int)
        if self.app.signal.filtered is not None and self.app.detection.all_candidates is not None:
            self.run_detection(float(self.app.sl_thr.get()))  # type: ignore[union-attr]
            self.app._draw_detail(self.app.ui.nav_pos)
        self.app.lbl_npeaks.configure(  # type: ignore[union-attr]
            text=f"Peaks detected: {len(self.app.detection.rpeaks_ok) if self.app.detection.rpeaks_ok is not None else 0}",
            text_color=GREEN,
        )
        self.app._set_status(
            f"Manual edits cleared — {n_excl} exclusion(s), {n_added} addition(s) removed.",
            GREEN)

    def on_detail_motion(self, event) -> None:
        """Track mouse position in edit mode and compute the preview peak position.

        In normal mode: snaps to the local maximum within ±tol_samp and shows
        an orange/red marker if it would land too close to an existing peak.
        In free placement mode: the preview follows the cursor exactly (no
        snapping) and is always shown in the "ok" colour.

        Redraws are throttled to 30 ms (≈33 fps) via after().
        """
        if self.app.detection.edit_mode and self.app.detection.edit_noise_mode:
            self._noise_motion(event)
            return
        if not self.app.detection.edit_mode:
            if self.app.detection.hover_samp is not None:
                self.app.detection.hover_samp = None
                self.app.detection.hover_samp_near = False
                if self.app.ui.hover_after_id is not None:
                    self.app.after_cancel(self.app.ui.hover_after_id)
                    self.app.ui.hover_after_id = None
                self.app._draw_detail(self.app.ui.nav_pos)
            return

        if event.xdata is None or self.app.signal.filtered is None or self.app.signal.fs is None:
            if self.app.detection.hover_samp is not None:
                self.app.detection.hover_samp = None
                self.app.detection.hover_samp_near = False
                self.app._draw_detail(self.app.ui.nav_pos)
            return

        fs         = self.app.signal.fs
        click_time = float(event.xdata)
        click_samp = int(np.clip(round(click_time * fs), 0, len(self.app.signal.filtered) - 1))

        # Free placement: follow the cursor exactly, no snapping, always "ok"
        if self.app.detection.edit_free_placement:
            new_samp = click_samp
            near     = False
        else:
            try:
                win = float(self.app.ent_window.get())
            except Exception:
                win = 2.0
            tol_samp = int(max(MouseECG.MIN_RR_MS / 1000 / 2, win * 0.03) * fs)

            # Snap to local maximum within ±tol_samp
            sig = self.app.signal.filtered
            lo  = max(0, click_samp - tol_samp)
            hi  = min(len(sig), click_samp + tol_samp + 1)
            new_samp = lo + int(np.argmax(sig[lo:hi]))

            # Show warning colour if this would land too close to an existing peak
            near = False
            if self.app.detection.rpeaks_ok is not None and len(self.app.detection.rpeaks_ok):
                min_sep = int(MouseECG.MIN_RR_MS / 1000 * fs * 0.5)
                near    = int(np.min(np.abs(self.app.detection.rpeaks_ok - new_samp))) < min_sep

        # Only schedule a redraw if the position actually changed
        if self.app.detection.hover_samp == new_samp and self.app.detection.hover_samp_near == near:
            return

        self.app.detection.hover_samp      = new_samp
        self.app.detection.hover_samp_near = near

        # Throttle: cancel any pending redraw, schedule a new one in 30 ms
        if self.app.ui.hover_after_id is not None:
            self.app.after_cancel(self.app.ui.hover_after_id)
        self.app.ui.hover_after_id = self.app.after(30, self.flush_hover_redraw)

    def flush_hover_redraw(self) -> None:
        """Execute the throttled hover redraw on the main thread.

        full=False -- a hover-preview marker only affects draw_detail()
        itself (see plot_controller.py's hover_samp usage); the minimap and
        RR/HR/Quality strip don't read hover state and are unchanged by it,
        so redrawing them here was three full-figure rebuilds for the price
        of one, on every ~30ms hover tick while in Edit mode.
        """
        self.app.ui.hover_after_id = None
        self.app._draw_detail(self.app.ui.nav_pos, full=False)

    def on_detail_click(self, event) -> None:
        """Edit-mode click handler for the detail view.

        Left-click  (button 1) near an existing peak → toggle exclusion
        Right-click (button 3) anywhere              → add peak at local max,
                                                       or remove if clicking
                                                       a manually-added peak

        Only active when ``_edit_mode`` is True.
        """
        if not self.app.detection.edit_mode:
            return
        if event.xdata is None or self.app.signal.filtered is None:
            return
        if self.app.detection.edit_noise_mode:
            self._noise_press(event)
            return
        if self.app.detection.rpeaks_ok is None:
            return

        fs = self.app.signal.fs
        click_time = float(event.xdata)      # seconds
        try:
            win = float(self.app.ent_window.get())
        except Exception:
            win = 2.0
        det = self.app.detection
        # What the click does: ecg.core.peak_editing.edit_at_click (rules there).
        edit = edit_at_click(
            self.app.signal.filtered, fs, click_time,
            "left" if event.button == 1 else "right" if event.button == 3 else "other",
            win, det.rpeaks_ok, det.rpeaks_manual_excl, det.rpeaks_manual_added,
            det.manual_excluded, det.manual_added, free_placement=det.edit_free_placement)
        if not edit.changes_peaks:
            return                      # not near a peak: nothing changes, results kept
        # Right before committing the change (a no-op click keeps the results).
        self._invalidate_stale_results()
        self.push_edit_undo()
        apply_click_edit(edit, det.manual_excluded, det.manual_added)
        self.run_detection(float(self.app.sl_thr.get()))  # type: ignore[union-attr]
        n_ok = len(det.rpeaks_ok) if det.rpeaks_ok is not None else 0
        n_added = len(det.manual_added)
        if edit.action == "add_free":
            self.app._set_status(
                f"[Free] Added peak at {edit.peak / fs:.3f} s  |  "
                f"Total added: {n_added}  |  Accepted: {n_ok}  "
                "— click Analyze to refresh HRV.", BLUE)
        elif edit.action == "remove_added":
            self.app._set_status(
                f"Removed manually added peak at {click_time:.3f} s  |  "
                f"Accepted: {n_ok}  — click Analyze to refresh HRV.", ORANGE)
        elif edit.action == "replace":
            self.app._set_status(
                f"Replaced peak {edit.old_peak/fs:.3f} s → {edit.peak/fs:.3f} s  "
                f"({edit.distance/fs*1000:.1f} ms apart)  |  Accepted: {n_ok}  "
                "— click Analyze to refresh HRV.", ORANGE)
        elif edit.action == "add":
            self.app._set_status(
                f"Added peak at {edit.peak / fs:.3f} s (snapped to local max)  |  "
                f"Total added: {n_added}  |  Accepted: {n_ok}  "
                "— click Analyze to refresh HRV.", ORANGE)
        else:                            # exclude / include
            self.app._set_status(
                f"Manual exclusions: {len(det.manual_excluded)}  |  Accepted peaks: {n_ok}  "
                "— click Analyze to refresh HRV.", ORANGE)
        self.app._draw_detail(self.app.ui.nav_pos)

    def on_threshold_slide(self, value: float) -> None:
        """Called continuously while the slider is being dragged.

        Widget label and entry are updated immediately for visual feedback.
        Detection and redraws are debounced (80 ms) so rapid drag events do
        not flood the rendering pipeline — especially important for long
        recordings where apply_threshold() + two canvas draws take ~50 ms.
        """
        self.app.lbl_thr.configure(text=f"Threshold:  {value:.3f}")
        self.app.ent_thr.delete(0, "end")  # type: ignore[union-attr]
        self.app.ent_thr.insert(0, f"{value:.3f}")  # type: ignore[union-attr]

        if self.app.signal.filtered is None or self.app.detection.all_candidates is None:
            return

        # Cancel any previously scheduled update and reschedule
        if self.app.ui.thr_debounce_id is not None:
            self.app.after_cancel(self.app.ui.thr_debounce_id)
        self.app.ui.thr_debounce_id = self.app.after(80, lambda v=value: self.apply_threshold_ui(v))

    def apply_threshold_ui(self, value: float) -> None:
        """Run detection and refresh plots — called after debounce delay.

        Always executes on the main thread (scheduled via after()), so it is
        safe to read the slider and write widgets directly.
        """
        self.app.ui.thr_debounce_id = None
        # Peaks are about to change — invalidate stale results
        self._drop_results()
        self.app.analysis_ctrl.invalidate_autonomic()
        self.run_detection(value)
        self.app._draw_detail(self.app.ui.nav_pos)

    def on_threshold_entry(self, event=None) -> None:
        """Called when the user types a value in the exact-threshold entry.

        Applies immediately — no debounce — since this is a deliberate commit.
        """
        try:
            value = max(0.01, min(2.0, float(self.app.ent_thr.get())))  # type: ignore[union-attr]
            self.app.sl_thr.set(value)  # type: ignore[union-attr]
            self.app.lbl_thr.configure(text=f"Threshold:  {value:.3f}")
            if self.app.signal.filtered is not None and self.app.detection.all_candidates is not None:
                self.apply_threshold_ui(value)
        except ValueError:
            pass

    def check_method_agreement(self) -> None:
        """Cross-check the active detection result against independently
        implemented detectors run on the same (already filtered) signal.

        A beat several DIFFERENT algorithms all miss is a far stronger
        "this one is questionable" signal than any single method's own
        confidence -- this is complementary to that per-peak confidence,
        not a replacement for it.

        Deliberately excludes "Auto" and whichever method is currently
        active from the comparison set: re-running the active method
        against itself would trivially "agree" on every beat, and Auto's
        candidate generation is entangled with fix_polarity()'s own
        polarity vote in a way that doesn't cleanly separate into a
        standalone re-invocation. Wavelet, SG+Derivative, and Envelope Max
        are independent implementations already; the ML detector joins
        them when a trained model exists. Any method that's unavailable
        (no PyWavelets, no trained model, or a genuine failure) is skipped
        with a reason shown to the user rather than silently dropped.
        """
        if self.app.signal.filtered is None or self.app.detection.rpeaks_ok is None \
                or len(self.app.detection.rpeaks_ok) == 0:
            messagebox.showwarning("Not ready", "Click Detect Peaks first.")
            return

        # Snapshot on the main thread -- the worker runs in a background
        # thread and must not touch Tk widgets or shared self.app.* state
        # that a new file load could reset mid-run.
        sig       = self.app.signal.filtered
        fs        = self.app.signal.fs
        primary   = self.app.detection.rpeaks_ok.copy()
        min_rr_ms = self.app._safe_float(self.app.ent_minrr, MouseECG.MIN_RR_MS)
        active    = (self.app.cb_det_method.get()
                     if self.app.cb_det_method is not None else "").lower()

        if self.app.lbl_agreement_status is not None:
            self.app.lbl_agreement_status.configure(
                text="  Running other detectors…", text_color=ORANGE)

        def _worker() -> dict:
            return method_agreement(sig, fs, primary, min_rr_ms, active)

        def _done(result: dict) -> None:
            if self.app.detection.rpeaks_ok is None:
                return
            if not result.get("ok"):
                reason = "; ".join(result.get("methods_skipped", [])) or "no other methods available"
                if self.app.lbl_agreement_status is not None:
                    self.app.lbl_agreement_status.configure(
                        text=f"  Agreement check: {reason}", text_color=ORANGE)
                self.app._set_status(f"Cross-method agreement: {reason}", ORANGE)
                return

            self.app.detection.agreement_disagree_samples = result["disagree_samples"]
            n_beats     = result["n_beats"]
            n_disagree  = result["n_disagree"]
            pct_confirmed = 100.0 * (n_beats - n_disagree) / max(n_beats, 1)
            summary = (
                f"{result['n_methods']} other method(s) · "
                f"{pct_confirmed:.0f}% beats confirmed by ≥1 other · "
                f"{n_disagree} unconfirmed"
            )
            self.app.detection.agreement_summary = summary
            self.app.detection.agreement_pct = pct_confirmed
            self.app.detection.agreement_peaks = primary
            if self.app.lbl_agreement_status is not None:
                color = (GREEN if n_disagree == 0
                         else ORANGE if pct_confirmed >= 90 else RED)
                self.app.lbl_agreement_status.configure(text=f"  {summary}", text_color=color)
            skipped = result.get("methods_skipped", [])
            skipped_note = ("  ·  skipped: " + ", ".join(skipped)) if skipped else ""
            self.app._set_status(
                f"Cross-method agreement done — {summary}{skipped_note}", GREEN)
            self.app._draw_detail()
            MethodAgreementDialog(self.app, result=result, fs=fs)

        self.app._start_async_result(
            self.app.btn_check_agreement, "Checking…", _worker, _done)
