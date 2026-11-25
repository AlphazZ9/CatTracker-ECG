# -*- coding: utf-8 -*-
"""
ecg.qt.tabs.compare_more
------------------------
The parts of Compare segments beyond the comparison itself, as in the Tk
tab (ecg/ui/compare_segments.py):

* the configurations relative to the stimulus: the rows of one animal
  saved as offsets from its stim time (ecg.core.segment_plan), applied to
  another at its own stim time -- typed, or one of the file's markers;
* the Prism exports: a new .pzfx file, or a Prism template filled
  (ecg.io.prism_template), the figures (tachograms and Poincaré, 300 dpi
  PNG or PDF / SVG) and Add to group table;
* the GROUP card: the group Poincaré of one pair A -> B of the group table
  the Group tab uses (ecg.ui.group_figure), and its figure;
* the setup (rows, block, reference, view, jitter, stim, the windows of
  the comparison shown) saved in and restored from sessions.

A mixin of CompareTab.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Optional

import numpy as np
from PySide6 import QtWidgets

from ecg.core import segment_plan
from ecg.core.models import MouseECG
from ecg.core.segment_plan import (apply_config, config_from_rows, default_stim, delete_template,
                                   marker_choices, save_template, user_templates)
from ecg.io import prism_template
from ecg.io.prism_template import PrismTemplateError, fill_template, recording_tables
from ecg.io.segment_export import pzfx_notes, write_pzfx
from ecg.qt import theme

log = logging.getLogger("ecg")

NO_CONFIG = "(no configuration saved)"
MARKER_PROMPT = "Stim marker ▾"
PRISM_NEW = "New Prism file…"
PRISM_FILL = "Fill a Prism template…"
PRISM_TIP = ("Fill a Prism template…: make your graphs and tests once in Prism on an "
             "exported file and save it as .pzfx; the app then pours new data into it "
             "(tables matched by title) and Prism redraws the graphs and recomputes the tests.")
FIG_FILTER = "PNG image (300 dpi) (*.png);;PDF (vector) (*.pdf);;SVG (vector) (*.svg)"
GROUP_SHEETS = ("Animals", "Group")
GROUP_FIG_SIZE = (14.0, 4.3)


def fmt_s(v: Any) -> str:
    """A time in seconds for an entry or a message: millisecond precision,
    trailing zeros dropped (the Tk _fmt_s)."""
    s = f"{float(v):.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _same_file(a: str, b: str) -> bool:
    return bool(a) and bool(b) and (os.path.normcase(os.path.abspath(a))
                                    == os.path.normcase(os.path.abspath(b)))


class CompareMoreMixin:
    # ── configurations ──────────────────────────────────────
    def _build_config_bar(self, parent_layout: QtWidgets.QBoxLayout) -> None:
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Stim at (s):"))
        self.ed_stim = QtWidgets.QLineEdit()
        self.ed_stim.setPlaceholderText("e.g. 1803")
        self.ed_stim.setMaximumWidth(90)
        self.ed_stim.setToolTip("Time of the stimulus in this recording (s). A configuration "
                                "is saved as offsets from it and applied at the stim time "
                                "typed here.")
        bar.addWidget(self.ed_stim)
        self.cb_marker = QtWidgets.QComboBox()
        self.cb_marker.setMinimumWidth(220)
        self.cb_marker.setToolTip("Markers of this Spike2 file (the first pulse of each train, "
                                  "×n: its number of pulses): choosing one puts its time in "
                                  "Stim at.")
        self.cb_marker.activated.connect(self._marker_picked)
        self.cb_marker.setVisible(False)
        bar.addWidget(self.cb_marker)
        bar.addWidget(QtWidgets.QLabel("Configuration:"))
        self.cb_cfg = QtWidgets.QComboBox()
        self.cb_cfg.setMinimumWidth(200)
        self.cb_cfg.setToolTip("Your saved configurations: the windows as offsets from the "
                               "stimulus.")
        bar.addWidget(self.cb_cfg)
        self.btn_cfg_apply = QtWidgets.QPushButton("Apply")
        self.btn_cfg_apply.setToolTip("Fill the rows with the chosen configuration, placed at "
                                      "Stim at.")
        self.btn_cfg_save = QtWidgets.QPushButton("Save as…")
        self.btn_cfg_save.setToolTip("Save the rows below as a configuration, relative to "
                                     "Stim at.")
        self.btn_cfg_del = QtWidgets.QPushButton("Delete")
        self.btn_cfg_del.setToolTip("Delete the chosen configuration.")
        for b, fn in ((self.btn_cfg_apply, self.cfg_apply), (self.btn_cfg_save, self.cfg_save),
                      (self.btn_cfg_del, self.cfg_delete)):
            b.clicked.connect(lambda _c=False, f=fn: f())
            bar.addWidget(b)
        self.lbl_cfg = QtWidgets.QLabel("")
        self.lbl_cfg.setWordWrap(True)
        bar.addWidget(self.lbl_cfg, 1)
        parent_layout.addLayout(bar)
        self._cfgs: "dict[str, Any]" = {}
        self._marker_opts: "dict[str, float]" = {}
        self.cfg_refresh()

    def cfg_say(self, text: str, color: "Optional[str]" = None) -> None:
        self.lbl_cfg.setText(text)
        self.lbl_cfg.setStyleSheet(f"color: {color};" if color else "")

    def cfg_refresh(self, select: "Optional[str]" = None) -> None:
        """The saved configurations in the selector (*select* chosen, else
        the one shown when it still exists, else the first)."""
        try:
            temps = user_templates(segment_plan.TEMPLATES_PATH)
        except Exception as exc:
            log.warning("configurations not read: %s", exc)
            temps = []
        self._cfgs = {t.name: t for t in temps}
        names = list(self._cfgs) or [NO_CONFIG]
        shown = self.cb_cfg.currentText()
        cur = select if select in self._cfgs else (shown if shown in self._cfgs else names[0])
        self.cb_cfg.clear()
        self.cb_cfg.addItems(names)
        self.cb_cfg.setCurrentText(cur)
        for b in (self.btn_cfg_apply, self.btn_cfg_del):
            b.setEnabled(bool(self._cfgs))

    def fill_markers(self) -> None:
        """The file's markers (Spike2), for Stim at."""
        markers = self.win.recording.signal.markers or {}
        self._marker_opts = dict(marker_choices(markers))
        self.cb_marker.clear()
        if self._marker_opts:
            self.cb_marker.addItems([MARKER_PROMPT] + list(self._marker_opts))
        self.cb_marker.setVisible(bool(self._marker_opts))

    def _marker_picked(self, _index: int) -> None:
        choice = self.cb_marker.currentText()
        t = self._marker_opts.get(choice)
        if t is None:
            return
        self.ed_stim.setText(fmt_s(t))
        self.cfg_say(f"Stim at {fmt_s(t)} s ({choice.split('  ')[0]})")

    def stim_value(self) -> "Optional[float]":
        """Stim at as a number (None: empty); ValueError when not one."""
        text = self.ed_stim.text().strip().replace(",", ".")
        if not text:
            return None
        v = float(text)
        if not np.isfinite(v):
            raise ValueError(text)
        return v

    def cfg_apply(self) -> None:
        p = theme.current()
        t = self._cfgs.get(self.cb_cfg.currentText())
        if t is None:
            self.cfg_say("⚠  Choose a saved configuration", p.RED)
            return
        try:
            stim = self.stim_value()
        except ValueError:
            self.cfg_say("⚠  Stim at: not a number", p.RED)
            return
        if stim is None:
            self.cfg_say("⚠  Type the stim time in Stim at (s) first", p.RED)
            return
        dur = self._duration()
        res_w = apply_config(t, stim, dur)
        self.set_rows([(w.label, w.lo, w.hi) for w in res_w])
        out = [w.label for w in res_w if w.clamped and not w.hi > w.lo]
        cut = [w.label for w in res_w if w.clamped and w.hi > w.lo]
        if out or cut:
            parts = []
            if out:
                parts.append("outside: " + ", ".join(out))
            if cut:
                parts.append("cut to fit: " + ", ".join(cut))
            self.cfg_say(f"⚠  '{t.name}' at {fmt_s(stim)} s, recording {dur:.1f} s — "
                         + "; ".join(parts), p.RED if out else p.ORANGE)
        else:
            self.cfg_say(f"✓  '{t.name}' applied at {fmt_s(stim)} s", p.GREEN)

    def cfg_save(self, name: "Optional[str]" = None) -> None:
        """Save the rows as a configuration relative to Stim at (*name*: asked
        when not given)."""
        p = theme.current()
        Q = QtWidgets.QMessageBox
        try:
            rows = self.read_rows()
        except ValueError as exc:
            self.cfg_say(f"⚠  {exc}", p.RED)
            return
        bad = [i + 1 for i, (_l, lo, hi) in enumerate(rows) if not hi > lo]
        if bad:
            self.cfg_say(f"⚠  Segment {bad[0]}: start ≥ end", p.RED)
            return
        try:
            stim = self.stim_value()
        except ValueError:
            self.cfg_say("⚠  Stim at: not a number", p.RED)
            return
        if stim is None:
            guess = default_stim(rows)
            if guess is None:
                self.cfg_say("⚠  Type the stim time in Stim at (s): the windows are saved "
                             "relative to it", p.RED)
                return
            post = next(lb for lb, _lo, _hi in rows if lb.strip().lower().startswith("post"))
            if Q.question(self, "Stim time", f"Stim at is empty. Use {fmt_s(guess)} s, the "
                          f"start of '{post}', as the stim time?") != Q.StandardButton.Yes:
                self.cfg_say("Type the stim time in Stim at (s), then Save as… again")
                return
            stim = guess
            self.ed_stim.setText(fmt_s(stim))
        if name is None:
            shown = self.cb_cfg.currentText()
            name, ok = QtWidgets.QInputDialog.getText(
                self, "Save configuration", "Name of this configuration:",
                text=shown if shown in self._cfgs else "")
            if not ok:
                return
        name = (name or "").strip()
        if not name:
            return
        if name in self._cfgs and Q.question(
                self, "Save configuration",
                f"Replace the configuration '{name}'?") != Q.StandardButton.Yes:
            return
        try:
            save_template(config_from_rows(name, rows, stim), segment_plan.TEMPLATES_PATH)
        except (ValueError, OSError) as exc:
            self.cfg_say(f"⚠  Not saved: {exc}", p.RED)
            return
        self.cfg_refresh(select=name)
        self.cfg_say(f"✓  '{name}' saved: {len(rows)} window(s) relative to the stim "
                     f"({fmt_s(stim)} s here)", p.GREEN)

    def cfg_delete(self) -> None:
        Q = QtWidgets.QMessageBox
        name = self.cb_cfg.currentText()
        if name not in self._cfgs:
            return
        if Q.question(self, "Delete configuration", f"Delete the configuration '{name}'? "
                      "This cannot be undone.") != Q.StandardButton.Yes:
            return
        try:
            delete_template(name, segment_plan.TEMPLATES_PATH)
        except OSError as exc:
            self.cfg_say(f"⚠  Not deleted: {exc}", theme.current().RED)
            return
        self.cfg_refresh()
        self.cfg_say(f"'{name}' deleted")

    # ── Prism, figures, group table ─────────────────────────
    def _build_more_exports(self, row: QtWidgets.QBoxLayout) -> "list[QtWidgets.QPushButton]":
        self.btn_prism = QtWidgets.QPushButton("Prism…")
        self.btn_prism.setToolTip(PRISM_TIP + "\n\nGraphPad Prism: the metric table, one table "
                                  "per metric (a column per segment; per pair in Pairs mode), "
                                  "the HR block means, the RR series and the Poincaré pairs; "
                                  "the windows and quality in the project notes.")
        menu = QtWidgets.QMenu(self.btn_prism)
        menu.addAction(PRISM_NEW).triggered.connect(lambda: self.export_prism_new())
        menu.addAction(PRISM_FILL).triggered.connect(lambda: self.export_prism_template())
        self.btn_prism.setMenu(menu)
        self.btn_figures = QtWidgets.QPushButton("Figures…")
        self.btn_figures.setToolTip("The tachograms and the Poincaré plots, 300 dpi PNG or "
                                    "vector PDF / SVG")
        self.btn_figures.clicked.connect(lambda: self.export_figures())
        self.btn_group = QtWidgets.QPushButton("Add to group table…")
        self.btn_group.setToolTip(
            "One row per recording (subject, sex, file, every metric and the quality of each "
            "segment): the table for a paired / repeated-measures test across animals. Every "
            "segment is stored, whatever the Δ % reference. Columns are named after the "
            "segment labels: use the same labels and windows for every animal. A recording "
            "already in it is replaced. Its whole-recording metrics, time course, annotations "
            "and event-locked results go to the same table, for the Group tab.")
        self.btn_group.clicked.connect(lambda: self.add_to_group())
        out = [self.btn_prism, self.btn_figures, self.btn_group]
        for b in out:
            row.addWidget(b)
        return out

    def _stamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    def say_export(self, text: str, ok: bool = True, warn: bool = False) -> None:
        p = theme.current()
        self.lbl_exp.setText(("✓  " if ok and not warn else "⚠  ") + text)
        self.lbl_exp.setStyleSheet(f"color: {p.RED if not ok else p.ORANGE if warn else p.GREEN};")

    def export_prism_new(self, path: "Optional[str]" = None) -> None:
        res = self._result()
        if res is None:
            return
        if path is None:
            path, _f = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export the segment comparison to GraphPad Prism",
                self._base_name() + ".pzfx", "GraphPad Prism (*.pzfx);;All files (*)")
        if not path:
            return
        try:
            n = write_pzfx(res, path, exported_at=self._stamp())
        except Exception as exc:
            log.exception("comparison Prism export")
            QtWidgets.QMessageBox.critical(self, "Prism export", str(exc))
            self.say_export(f"Prism export failed: {exc}", ok=False)
            return
        self.say_export(f"Prism: {n} tables → {os.path.basename(path)}")

    def last_prism_template(self) -> str:
        path = getattr(self, "_prism_template_path", "") or ""
        if path and os.path.exists(path):
            return path
        return prism_template.load_last_template(prism_template.PREFS_PATH)

    def ask_prism_template(self) -> "Optional[str]":
        last = self.last_prism_template()
        path, _f = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose the Prism template (.pzfx saved from Prism)", last or "",
            "GraphPad Prism XML (*.pzfx);;All files (*)")
        if not path:
            return None
        self._prism_template_path = path
        prism_template.save_last_template(path, prism_template.PREFS_PATH)
        return path

    def ask_filled_path(self, template: str, base: "Optional[str]" = None) -> "Optional[str]":
        """'<base>_<template>.pzfx' next to the template by default, never
        the template itself."""
        stem = os.path.splitext(os.path.basename(template))[0]
        base = self._base_name() if base is None else base
        while True:
            path, _f = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save the filled Prism file (the template is not changed)",
                os.path.join(os.path.dirname(template), f"{base}_{stem}.pzfx"),
                "GraphPad Prism (*.pzfx);;All files (*)")
            if not path:
                return None
            if not _same_file(path, template):
                return path
            QtWidgets.QMessageBox.warning(self, "Prism template", "Choose another name: the "
                                          "template itself is never overwritten.")

    def export_prism_template(self, template: "Optional[str]" = None,
                              out: "Optional[str]" = None) -> None:
        res = self._result()
        if res is None:
            return
        tpl = template or self.ask_prism_template()
        if not tpl:
            return
        out = out or self.ask_filled_path(tpl)
        if not out:
            return
        try:
            report = fill_template(tpl, recording_tables(res), out,
                                   notes=pzfx_notes(res, self._stamp())
                                   + [("Prism template", os.path.basename(tpl))])
        except Exception as exc:
            if not isinstance(exc, PrismTemplateError):
                log.exception("comparison Prism template")
            QtWidgets.QMessageBox.critical(self, "Prism template", str(exc))
            self.say_export(f"Prism template: {exc}", ok=False)
            return
        self.say_export(f"Prism template {os.path.basename(tpl)}: {report.summary()} → "
                        f"{os.path.basename(out)}", warn=report.warn)

    def export_figures(self, path: "Optional[str]" = None) -> None:
        """The tachograms and the Poincaré plots: '<name>_tachograms' and
        '<name>_poincare', 300 dpi."""
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from ecg.ui.compare_segments import draw_rr_tachograms, segment_color, tachogram_groups
        from ecg.ui.poincare_figure import draw_poincare_set
        from ecg.qt.tabs.compare_tab import segment_names
        res = self._result()
        if res is None:
            return
        Q = QtWidgets.QMessageBox
        if path is None:
            path, _f = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save the comparison figures (tachograms and Poincaré)",
                self._base_name() + ".png", FIG_FILTER,
                options=QtWidgets.QFileDialog.Option.DontConfirmOverwrite)
        if not path:
            return
        base, ext = os.path.splitext(path)
        ext = ext.lower() if ext.lower() in (".png", ".pdf", ".svg") else ".png"
        targets = [base + sfx + ext for sfx in ("_tachograms", "_poincare")]
        existing = [os.path.basename(t) for t in targets if os.path.exists(t)]
        if existing and Q.question(self, "Figures", "Replace " + " and ".join(existing)
                                   + "?") != Q.StandardButton.Yes:
            return
        segs = res.segs
        colors = [segment_color(k) for k in range(len(segs))]
        p_cols = max(1, min(len(segs), 2))      # a Pre/Post pair side by side
        n_rows = -(-len(segs) // p_cols)
        n_tacho = len(tachogram_groups(len(segs), res.comps))
        comps, names = res.comps, segment_names(segs)
        subject = self.win.subject()
        written = []
        for suffix, figsize, draw in (
                ("_tachograms", (10, 4 if n_tacho == 1 else 3 * n_tacho),
                 lambda fig: draw_rr_tachograms(fig, segs, colors, comps, names)),
                ("_poincare", (4 * p_cols, 4.6 * n_rows),
                 lambda fig: draw_poincare_set(fig, segs, colors, title=subject,
                                               jitter=self.jitter, ncols=p_cols))):
            out = base + suffix + ext
            try:
                fig = Figure(figsize=figsize, dpi=300, facecolor="white", layout="constrained")
                if draw(fig) is not False:
                    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
                    written.append(os.path.basename(out))
                plt.close(fig)
            except Exception as exc:
                log.warning("comparison figure %s: %s", out, exc)
                self.say_export(f"figure {os.path.basename(out)} failed: {exc}", ok=False)
                return
        self.say_export("figures → " + "  +  ".join(written))

    def add_to_group(self, path: "Optional[str]" = None) -> None:
        """The recording open (this comparison's segments with its whole
        recording, time course and events) into a group table (created if
        new): Tk _add_to_group with TabGroupMixin._add_open_recording_to_group_table."""
        from ecg.ui.tab_group import TabGroupMixin
        res = self._result()
        if res is None:
            return
        if path is None:
            path, _f = QtWidgets.QFileDialog.getSaveFileName(
                self, "Group table: add this recording (created if new)", "group_segments.xlsx",
                "Excel workbook (*.xlsx);;All files (*)",
                options=QtWidgets.QFileDialog.Option.DontConfirmOverwrite)
        if not path:
            return
        try:
            _n, replaced, what = TabGroupMixin._add_open_recording_to_group_table(
                self.win.figures, path, comparison=res)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Group table", str(exc))
            return
        group = self.win.group
        if _same_file(group.path, path):
            group.reload()
        if res.pairs:
            j, k = res.pairs[0]
            self._group_default_pair = (res.tags[j], res.tags[k])
        self.group_added = path
        self.group_refresh(force=True)
        self.say_export(f"{res.subject or 'recording'} {'updated in' if replaced else 'added to'} "
                        f"{os.path.basename(path)} — {what.split(': ', 1)[-1]}")

    # ── GROUP card: the group Poincaré ──────────────────────
    def _build_group_card(self, parent_layout: QtWidgets.QBoxLayout) -> None:
        card = QtWidgets.QFrame(objectName="Card")
        gl = QtWidgets.QVBoxLayout(card)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(QtWidgets.QLabel("<b>Group — all animals</b>"))
        note = QtWidgets.QLabel("The tests across animals (one pair or all pairs, by group or "
                                "sex) are in the Group tab: Add to group table above, then "
                                "open it there.")
        note.setProperty("muted", True)
        note.setWordWrap(True)
        head.addWidget(note, 1)
        self.btn_group_fig = QtWidgets.QPushButton("Figure…")
        self.btn_group_fig.setEnabled(False)
        self.btn_group_fig.setToolTip("The group Poincaré (one ellipse per animal, the group mean "
                                      "thick) and the before-after panels of the pair shown: "
                                      "300 dpi PNG or vector PDF / SVG")
        self.btn_group_fig.clicked.connect(lambda: self.save_group_figure())
        head.addWidget(self.btn_group_fig)
        btn_open = QtWidgets.QPushButton("Open the Group tab")
        btn_open.setToolTip("The Group tab, on the group table drawn here")
        btn_open.clicked.connect(self.open_group_tab)
        head.addWidget(btn_open)
        gl.addLayout(head)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Group Poincaré"))
        self.cb_ga, self.cb_gb = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        for cb, sep in ((self.cb_ga, "→"), (self.cb_gb, "")):
            cb.setMinimumWidth(160)
            cb.setToolTip("The pair drawn, A → B: the Group tab's pair by default, else this "
                          "comparison's reference and the first segment compared with it")
            cb.activated.connect(lambda _i: self._group_pair_chosen())
            row.addWidget(cb)
            if sep:
                row.addWidget(QtWidgets.QLabel(sep))
        self.lbl_group = QtWidgets.QLabel("")
        self.lbl_group.setProperty("muted", True)
        self.lbl_group.setWordWrap(True)
        row.addWidget(self.lbl_group, 1)
        gl.addLayout(row)
        self.group_plot = self.win.figures.slot("compare_group", card, *GROUP_FIG_SIZE)
        self.group_plot.setMinimumHeight(430)
        self.group_plot.setVisible(False)
        gl.addWidget(self.group_plot)
        parent_layout.addWidget(card)
        self.group_added = ""
        self._group_cache: "Optional[dict]" = None
        self._group_choice: "Optional[dict]" = None
        self._group_default_pair: "Optional[tuple[str, str]]" = None
        self._group_tags: "dict[str, str]" = {}
        self._group_inputs: Any = None
        self._group_drawn: Any = None
        self.group_shown: Any = None

    def group_path(self) -> str:
        """The Group tab's table, else the one this tab last added to."""
        return self.win.group.path or self.group_added or ""

    def open_group_tab(self) -> None:
        group = self.win.group
        path = self.group_path()
        if path and not _same_file(group.path, path) and os.path.exists(path):
            group.load(path)
        self.win.tabs.setCurrentWidget(group)

    def _group_sheets(self, path: str) -> dict:
        try:
            st = os.stat(path)
            key = (os.path.normcase(os.path.abspath(path)), st.st_mtime_ns, st.st_size)
        except OSError:
            return {"error": f"Group table not found: {os.path.basename(path)}"}
        if self._group_cache is not None and self._group_cache.get("key") == key:
            return self._group_cache
        from ecg.io import group_data as gd
        try:
            cache = {"key": key, "sheets": gd.read_sheets(path, only=GROUP_SHEETS)}
        except Exception as exc:
            cache = {"key": key, "error": f"{os.path.basename(path)}: {exc}"}
        self._group_cache = cache
        return cache

    def _group_tab_pair(self) -> "tuple[str, str]":
        g = self.win.group
        if g.sheets is None:
            return "", ""
        return g.cb_a.currentText(), g.cb_b.currentText()

    def _group_pair(self, tags: "list[str]") -> "tuple[str, str]":
        """The pair chosen here (while the Group tab's stays the one it was
        chosen against), else the Group tab's, else this comparison's
        reference vs the first segment compared with it, else the first two."""
        gp = self._group_tab_pair()
        mine = self._group_choice
        if mine is not None and mine["seen"] == gp and mine["a"] in tags and mine["b"] in tags:
            return mine["a"], mine["b"]
        self._group_choice = None
        if gp[0] in tags and gp[1] in tags and gp[0] != gp[1]:
            return gp
        res = self.win.recording.analysis.last_comparison
        if res is not None and getattr(res, "pairs", None):
            j, k = res.pairs[0]
            a, b = res.tags[j], res.tags[k]
            if a in tags and b in tags and a != b:
                return a, b
        return tags[0], tags[1]

    def _group_pair_chosen(self) -> None:
        self._group_choice = {"a": self._group_tags.get(self.cb_ga.currentText(), ""),
                              "b": self._group_tags.get(self.cb_gb.currentText(), ""),
                              "seen": self._group_tab_pair()}
        self.group_refresh()

    def _group_nothing(self, text: str) -> None:
        self.lbl_group.setText(text)
        self._group_drawn = self.group_shown = None
        self.btn_group_fig.setEnabled(False)
        self.group_plot.setVisible(False)

    def _group_combos(self, choices: "list[tuple[str, str]]", a: str = "", b: str = "") -> None:
        self._group_tags = {v: t for t, v in choices}
        shown = dict(choices)
        for cb, t in ((self.cb_ga, a), (self.cb_gb, b)):
            cb.clear()
            cb.addItems([v for _t, v in choices])
            cb.setEnabled(bool(choices))
            cb.setCurrentText(shown.get(t, ""))

    def group_refresh(self, force: bool = False) -> None:
        """Tk _compare_group_refresh: the table (read again when it changed),
        the pair, the animals used; drawn again only when one changed."""
        from ecg.io import group_compare as gc
        from ecg.io import group_data as gd
        from ecg.ui.compare_segments import group_poincare_data, segment_choices, segment_color
        from ecg.ui.group_figure import draw_group_figure
        if force:
            self._group_cache = None
        path = self.group_path()
        got = self._group_sheets(path) if path else {}
        g = self.win.group
        leave_out = bool(g.chk_leave.isChecked())
        by = g.cb_by.currentText() or gc.BY_GROUP
        res = self.win.recording.analysis.last_comparison
        inputs = (path, got.get("key"), self._group_tab_pair(), leave_out, by,
                  str(self._group_choice),
                  (tuple(res.tags), tuple(res.pairs[:1])) if getattr(res, "pairs", None) else None)
        if inputs == self._group_inputs and not force:
            return
        self._group_inputs = inputs
        if not path:
            self._group_combos([])
            self._group_nothing("No group table yet: Add to group table above, or open one in "
                                "the Group tab.")
            return
        name = os.path.basename(path)
        if "error" in got:
            self._group_combos([])
            self._group_nothing(f"⚠  {got['error']}")
            return
        seg = got["sheets"].get(gd.SHEET_SEGMENTS)
        if seg is None or seg.empty:
            self._group_combos([])
            self._group_nothing(f"{name} has no segments yet (no Group sheet): Add to group "
                                "table above.")
            return
        choices = segment_choices(seg)
        if len(choices) < 2:
            self._group_combos([])
            self._group_nothing(f"{name} holds fewer than two segments: no pair to draw.")
            return
        a, b = self._group_pair([t for t, _v in choices])
        self._group_combos(choices, a, b)
        if a == b:
            self._group_nothing(f"Choose two different segments · table: {name}")
            return
        try:
            rows, out = group_poincare_data(got["sheets"], a, b, leave_out, by)
        except ValueError as exc:
            self._group_nothing(f"⚠  {exc} · table: {name}")
            return
        except Exception as exc:
            log.warning("group Poincaré: %s", exc, exc_info=True)
            self._group_nothing(f"⚠  {name}: {exc}")
            return
        extra = []
        if out["unticked"]:
            extra.append(f"{out['unticked']} unticked")
        if out["ungrouped"]:
            extra.append(f"{out['ungrouped']} " + ("of unknown sex" if by == gc.BY_SEX
                                                    else "without a group"))
        if out["to_check"]:
            extra.append(f"{out['to_check']} to check")
        tail = " (left out: " + ", ".join(extra) + ")" if extra else ""
        if out.get("pooled"):
            tail += " · no " + ("sex" if by == gc.BY_SEX else "group") + " set: all ticked animals"
        tail += f" · table: {name}"
        n = len(rows)
        if n < 2:
            self._group_nothing(f"Only {n} animal{'s' if n != 1 else ''} with a mean RR, SD1 and "
                                f"SD2 for both {a} and {b} (2 needed){tail}")
            return
        self.lbl_group.setText(f"n = {n} animals{tail}")
        self.btn_group_fig.setEnabled(True)
        self.group_plot.setVisible(True)
        shown = dict(choices)
        names = tuple(shown[t].split(" · ", 1)[-1] for t in (a, b))
        tags = list(getattr(res, "tags", None) or [])
        colors = tuple(segment_color(tags.index(t) if t in tags else k)
                       for k, t in enumerate((a, b)))
        self.group_shown = (rows, a, b, names, path, colors)
        sig = (got["key"], a, b, leave_out, by, colors, tuple(map(str, rows.index)))
        if sig != self._group_drawn:
            self._group_drawn = sig
            self.group_plot.update(lambda fig: draw_group_figure(fig, rows, a, b, names, colors))

    def save_group_figure(self, out: "Optional[str]" = None) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from ecg.ui.group_figure import draw_group_figure
        from ecg.ui.theme import light_plot_theme
        if self.group_shown is None:
            return
        rows, a, b, names, table, colors = self.group_shown
        if out is None:
            out, _f = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save the group Poincaré",
                os.path.join(os.path.dirname(table), f"group_poincare_{a}_vs_{b}.png"),
                FIG_FILTER)
        if not out:
            return
        fig = Figure(figsize=GROUP_FIG_SIZE, dpi=300, facecolor="white", layout="constrained")
        try:
            with light_plot_theme():
                draw_group_figure(fig, rows, a, b, names, colors)
                fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Group Poincaré", str(exc))
            return
        finally:
            plt.close(fig)
        self.lbl_group.setText(f"✓  figure → {os.path.basename(out)}")

    # ── the setup, in sessions ──────────────────────────────
    def setup_snapshot(self) -> "Optional[dict]":
        """CompareSegmentsMixin._compare_setup_snapshot: the rows as typed,
        the block, the reference, the view, the jitter, the stim and the
        windows of the comparison shown. None without a recording."""
        from ecg.ui.compare_segments import normalize_compare_setup
        if self.win.recording.signal.filtered is None or self.rows.rowCount() == 0:
            return None

        def _num(text: str) -> Any:
            try:
                f = float(text.strip())
            except ValueError:
                return text
            return f if np.isfinite(f) else text
        rows = [[(self.rows.item(i, 0).text() if self.rows.item(i, 0) else ""),
                 _num(self.rows.item(i, 1).text() if self.rows.item(i, 1) else ""),
                 _num(self.rows.item(i, 2).text() if self.rows.item(i, 2) else "")]
                for i in range(self.rows.rowCount())]
        run = self.win.recording.analysis.last_comparison_run if self.segs else None
        return normalize_compare_setup({
            "rows": rows, "block_s": fmt_s(self.sp_block.value()), "ref": self.ref,
            "stim": self.ed_stim.text().strip(),
            "view": "poincare" if self.cb_view.currentIndex() == 1 else "rr",
            "jitter": bool(self.jitter), "ran": run is not None,
            "run": dict(run) if run is not None else None})

    def restore_setup(self, raw: Any) -> None:
        """CompareSegmentsMixin._restore_compare_setup: the setup into the
        tab, its comparison re-run on the restored peaks."""
        from ecg.ui.compare_segments import normalize_compare_setup
        setup = normalize_compare_setup(raw)
        if setup is None:
            return
        self.rows.setRowCount(0)
        for lbl, lo, hi in setup["rows"]:
            r = self.rows.rowCount()
            self.rows.insertRow(r)
            for c, v in enumerate((lbl, lo, hi)):
                text = fmt_s(v) if isinstance(v, (int, float)) else str(v)
                self.rows.setItem(r, c, QtWidgets.QTableWidgetItem(text))
        self.ed_stim.setText(setup.get("stim") or "")
        try:
            self.sp_block.setValue(float(setup["block_s"]))
        except (TypeError, ValueError):
            self.sp_block.setValue(float(MouseECG.SEGMENT_BLOCK_S))
        self.ref = setup["ref"]
        self.jitter = bool(setup["jitter"])
        self.chk_jitter.setChecked(self.jitter)
        self.cb_view.setCurrentIndex(1 if setup["view"] == "poincare" else 0)
        run = setup.get("run")
        if run is not None and self.win.recording.detection.rpeaks_ok is not None:
            self.run(windows=[tuple(r) for r in run["rows"]], block_s=run["block_s"])
