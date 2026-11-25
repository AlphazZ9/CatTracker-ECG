# -*- coding: utf-8 -*-
"""
ecg.qt.dialogs.wave_editor
--------------------------
The wave-template editor (ecg/ui/wave_editor.py WaveTemplateMiniEditor)
in Qt: the mean beat with one line (centre) and one band (search window)
per landmark; drag a line to move a centre, a dashed edge to resize its
window. Reset / Auto-calibrate / Refine from delineation / Load / Save to
file / Apply to this recording / Save as default, with the same
provenance rules (hit_landmark, dragged_landmark, WaveTemplate).
"""
from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
from PySide6 import QtWidgets

from ecg.core.wave_template import WaveTemplate, _mouse_demo_beat
from ecg.qt.widgets.mpl_canvas import MplCanvas
from ecg.app.wave_landmarks import WaveLandmarks as _Tk, dragged_landmark, hit_landmark

KEYS, COLS, LABELS = _Tk._KEYS, _Tk._COLS, _Tk._LABELS
GRAB_R_MS = _Tk._GRAB_R_MS


class WaveTemplateEditor(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget, template: WaveTemplate,
                 mean_beat: "Optional[np.ndarray]" = None,
                 beat_time: "Optional[np.ndarray]" = None,
                 beat_sd: "Optional[np.ndarray]" = None,
                 intervals: Any = None, recording_name: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Wave Template Editor")
        self.resize(900, 560)
        self.template = template
        self.intervals = intervals
        self.recording_name = recording_name
        self.src, self.origin = template.source, template.origin
        self.saved = False
        self._drag: "tuple[Optional[str], str]" = (None, "center")
        if mean_beat is not None and beat_time is not None:
            self.mb = np.asarray(mean_beat, dtype=float)
            self.bt = np.asarray(beat_time, dtype=float)
            self.sd = np.asarray(beat_sd, dtype=float) if beat_sd is not None else None
        else:
            self.bt, self.mb = _mouse_demo_beat()
            self.sd = None
        self.lm: "dict[str, list[float]]" = {
            k: list(v) for k, v in {**WaveTemplate.DEFAULTS, **template.landmarks}.items()}

        lay = QtWidgets.QVBoxLayout(self)
        self.canvas = MplCanvas(self, 9.0, 4.0)
        lay.addWidget(self.canvas, 1)
        c = self.canvas.canvas
        c.mpl_connect("button_press_event", self._press)
        c.mpl_connect("motion_notify_event", self._motion)
        c.mpl_connect("button_release_event", lambda _e: self._release())
        hint = QtWidgets.QLabel("Drag a coloured line to move a landmark centre.  "
                                "Drag a dashed edge to resize the search window.")
        hint.setProperty("muted", True)
        lay.addWidget(hint)
        bar = QtWidgets.QHBoxLayout()
        for text, fn, show in (("Reset defaults", self.reset_defaults, True),
                               ("Auto-calibrate", self.auto_calibrate, True),
                               ("Refine from delineation", self.refine, intervals is not None),
                               ("Load…", self.load_file, True),
                               ("Save to file…", self.save_file, True)):
            if show:
                b = QtWidgets.QPushButton(text)
                b.clicked.connect(fn)
                bar.addWidget(b)
        bar.addStretch(1)
        for text, fn in (("Save as default template", self.save_default),
                         ("Apply to this recording", self.apply),
                         ("Cancel", self.reject)):
            b = QtWidgets.QPushButton(text)
            if text.startswith("Apply"):
                b.setProperty("primary", True)
            b.clicked.connect(fn)
            bar.addWidget(b)
        lay.addLayout(bar)
        self.draw()

    # ── drawing / dragging ──────────────────────────────────
    def draw(self) -> None:
        def _draw(fig) -> None:
            ax = fig.add_subplot(111)
            if self.sd is not None:
                ax.fill_between(self.bt, self.mb - self.sd, self.mb + self.sd, alpha=0.15)
            ax.plot(self.bt, self.mb, lw=1.5, zorder=2)
            ax.axvline(0, color="#E57373", lw=0.8, alpha=0.6)
            ylo, yhi = ax.get_ylim()
            yspan = yhi - ylo if yhi != ylo else 1.0
            for key in KEYS:
                if key not in self.lm:
                    continue
                c, hw = self.lm[key]
                col = COLS[key]
                ax.axvspan(c - hw, c + hw, alpha=0.12, color=col, zorder=0)
                ax.axvline(c, color=col, lw=1.5, alpha=0.9, zorder=3)
                for edge in (c - hw, c + hw):
                    ax.axvline(edge, color=col, lw=0.8, ls="--", alpha=0.55, zorder=3)
                ax.text(c, yhi - 0.04 * yspan, LABELS[key], ha="center", va="top",
                        fontsize=7, color=col, rotation=90)
            ax.set_xlim(float(self.bt[0]), float(self.bt[-1]))
            ax.set_xlabel("Time from R (ms)")
            ax.set_ylabel("Amplitude (norm.)")
            ax.set_title(f"Wave Template — {self.pending().provenance()}", loc="left",
                         fontsize=9)
        self.canvas.update_figure(_draw)

    def _press(self, event: Any) -> None:
        if event.inaxes is None or event.xdata is None or event.button != 1:
            return
        self._drag = hit_landmark(self.lm, KEYS, float(event.xdata), GRAB_R_MS)

    def _motion(self, event: Any) -> None:
        key, mode = self._drag
        if key is None or event.xdata is None:
            return
        c, hw = self.lm[key]
        self.lm[key] = list(dragged_landmark(c, hw, mode, float(event.xdata)))
        self.mark_edited()
        self.draw()

    def _release(self) -> None:
        self._drag = (None, "center")

    # ── provenance (as the Tk editor) ───────────────────────
    def pending(self) -> WaveTemplate:
        wt = WaveTemplate()
        wt.landmarks = {k: (float(v[0]), float(v[1])) for k, v in self.lm.items()}
        wt.source, wt.origin = self.src, self.origin
        wt.created_at = self.template.created_at if self.src == "saved-default" else ""
        return wt

    def mark_edited(self) -> None:
        if self.src != "edited":
            self.src, self.origin = "edited", self.pending().provenance()

    # ── actions ─────────────────────────────────────────────
    def reset_defaults(self) -> None:
        self.lm = {k: list(v) for k, v in WaveTemplate.DEFAULTS.items()}
        self.src, self.origin = "default", ""
        self.draw()

    def auto_calibrate(self) -> None:
        tmp = WaveTemplate()
        tmp.calibrate_from_beat(self.bt, self.mb)
        self.lm = {k: list(v) for k, v in tmp.landmarks.items()}
        self.src, self.origin = "auto-calibrated", ""
        self.draw()

    def refine(self) -> None:
        from ecg.core.analysis import refine_template_from_intervals
        new = (refine_template_from_intervals(self.intervals, self.pending())
               if self.intervals is not None else None)
        if new is None:
            QtWidgets.QMessageBox.information(
                self, "Not enough beats", "Refining needs at least 10 fully delineated beats "
                "— Measure intervals first.")
            return
        self.lm = {k: list(v) for k, v in {**WaveTemplate.DEFAULTS, **new.landmarks}.items()}
        self.src, self.origin = "refined", ""
        self.draw()

    def load_file(self) -> None:
        path, _f = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load wave template", "", "Wave template (*.json);;All files (*)")
        if not path:
            return
        try:
            wt = WaveTemplate.load(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            return
        self.lm = {k: list(v) for k, v in {**WaveTemplate.DEFAULTS, **wt.landmarks}.items()}
        self.src, self.origin = wt.source, wt.origin
        self.draw()
        problems = wt.plausibility_problems()
        if problems:
            QtWidgets.QMessageBox.warning(
                self, "Check these windows",
                "These search windows look implausible for a mouse beat:\n\n"
                + "\n".join(f"• {p}" for p in problems)
                + "\n\nCheck them on the beat before applying.")

    def save_file(self) -> None:
        path, _f = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save wave template", "", "Wave template (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            self.pending().save(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(exc))

    def commit(self) -> None:
        """The windows and their provenance applied to this recording's
        template, in place; nothing written to disk."""
        p = self.pending()
        t = self.template
        t.landmarks, t.source, t.origin, t.created_at = p.landmarks, p.source, p.origin, \
            p.created_at
        t.confirmed = True
        self.saved = True

    def apply(self) -> None:
        self.commit()
        QtWidgets.QMessageBox.information(
            self, "Template applied", "Template applied to this recording (kept in its "
            "session).\nMeasure intervals again to use these search windows.")
        self.accept()

    def save_default(self) -> None:
        problems = self.pending().plausibility_problems()
        if problems:
            QtWidgets.QMessageBox.warning(
                self, "Not saved as default",
                "These search windows look implausible for a mouse beat, so they would not "
                "be used as the default template:\n\n"
                + "\n".join(f"• {p}" for p in problems)
                + "\n\nAdjust them (or Apply to this recording only).")
            return
        if QtWidgets.QMessageBox.question(
                self, "Save as default template",
                "Use these search windows as the default template for recordings opened "
                "from now on?\n\nRecordings already analysed and saved sessions keep their "
                "own template.") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            self.pending().save(recording=self.recording_name)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(exc))
            return
        self.commit()
        QtWidgets.QMessageBox.information(
            self, "Template saved", "Saved as the default template and applied to this "
            "recording.\nMeasure intervals again to use these search windows.")
        self.accept()


def open_editor(window: Any) -> None:
    """ECGApp._open_wave_template_editor in the Qt interface."""
    rec = window.recording
    r = rec.analysis.results
    wt = window.morphology.wave_template()
    path = rec.signal.filepath
    dlg = WaveTemplateEditor(window, wt,
                             mean_beat=r.get("beat_template") if r else None,
                             beat_time=r.get("beat_time") if r else None,
                             beat_sd=r.get("beat_sd") if r else None,
                             intervals=r.get("intervals") if r else None,
                             recording_name=os.path.basename(path) if path else "")
    dlg.exec()
    if dlg.saved:
        window.session_dirty = True
        if r is not None:
            window.figures.plot_ctrl.plot_beat_template(r)
        window.set_status("Wave template applied to this recording — Measure intervals again "
                          "to use it.")
