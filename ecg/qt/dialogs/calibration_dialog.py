# -*- coding: utf-8 -*-
"""
ecg.qt.dialogs.calibration_dialog
---------------------------------
Time course > Autonomic > Calibrate ks / kp (the Tk CalibrationDialog):
the cohort calibration of the CSI / CPI gains, after Candia-Rivera et al.,
J Physiol 2025.

The event-locked rows saved to the cohort registry (label and project
filters, one method / window configuration at a time) give, per event or
per subject mean, the paired Wilcoxon Z(k) of CSI and CPI for k = 1..20
with the knee, the counts and the pre vs post tests at the current gains.
Every statistic is the Tk dialog's pure helper (ecg.app.event_locked_view);
this window only reads its inputs and draws. Apply writes the knees into
the Autonomic view's ks / kp boxes -- after a confirmation when a caveat
applies -- and nothing is recomputed until Compute is pressed there.
Non-modal.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from PySide6 import QtWidgets

from ecg.io.db import export_event_rows_csv, list_event_labels, list_event_rows
from ecg.qt import theme
from ecg.qt.widgets.mpl_canvas import MplCanvas
from ecg.app.event_locked_view import (
    AGGREGATIONS, CALIBRATION_RULE_NOTE, DEFAULT_AGGREGATION, PSEUDO_REPLICATION_NOTE,
    apply_caveats, apply_plan, calibration_text, cohort_calibration, config_label, config_of,
    config_options, default_calibration_label, draw_calibration, in_sample, label_counts,
    registry_frame, row_config, tests_frame,
)

log = logging.getLogger("ecg")

ALL = "All labels"
NO_ROW_TEXT = (
    "  No valid event-locked row in the cohort registry{filt}.\n\n"
    "  For each recording: Time course > Autonomic > Compute, then Event-locked > "
    "Compute event-locked, then Save to cohort registry (with the Subject ID and "
    "Project set in the Recording sheet).")


def _label_counts(project: "Optional[str]") -> "dict[str, int]":
    """Valid registry events per label (of *project*, None = all), most first."""
    try:
        return label_counts(registry_frame(list_event_rows(project_name=project or None,
                                                           valid_only=True)))
    except Exception as exc:
        log.debug("CalibrationDialog: label counts: %s", exc)
        return {}


class CalibrationDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget, ks: float, kp: float,
                 config: "Optional[tuple]" = None, project: str = "",
                 on_apply: "Optional[Callable[[float, float], None]]" = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Calibrate ks / kp — cohort registry")
        self.resize(1080, 860)
        self._ks, self._kp = float(ks), float(kp)
        self._on_apply = on_apply
        self._want_config = config_of(config[0], config[1]) if config is not None else None
        self._rows: "list[dict]" = []
        self._frame = registry_frame([])
        self._configs: "dict[str, tuple]" = {}
        self.cal = None
        self._apply_values: "tuple[float, float]" = (self._ks, self._kp)

        lay = QtWidgets.QVBoxLayout(self)
        note = QtWidgets.QLabel(CALIBRATION_RULE_NOTE + "  " + PSEUDO_REPLICATION_NOTE)
        note.setWordWrap(True)
        note.setProperty("muted", True)
        lay.addWidget(note)

        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Label:"))
        self.cb_label = QtWidgets.QComboBox()
        self.cb_label.addItems([ALL] + list_event_labels())
        lab = default_calibration_label(_label_counts(project.strip() or None))
        self.cb_label.setCurrentText(lab if lab is not None else ALL)
        bar.addWidget(self.cb_label)
        bar.addWidget(QtWidgets.QLabel("Project:"))
        self.ed_project = QtWidgets.QLineEdit(project)
        self.ed_project.setPlaceholderText("(all projects)")
        bar.addWidget(self.ed_project)
        self.btn_load = QtWidgets.QPushButton("↻ Load")
        bar.addWidget(self.btn_load)
        bar.addWidget(QtWidgets.QLabel("Configuration:"))
        self.cb_config = QtWidgets.QComboBox()
        self.cb_config.setMinimumWidth(230)
        bar.addWidget(self.cb_config, 1)
        lay.addLayout(bar)

        agg = QtWidgets.QHBoxLayout()
        agg.addWidget(QtWidgets.QLabel("Aggregation:"))
        self.agg_buttons: "dict[str, QtWidgets.QRadioButton]" = {}
        for key, label in AGGREGATIONS.items():
            rb = QtWidgets.QRadioButton(label)
            rb.setChecked(key == DEFAULT_AGGREGATION)
            rb.toggled.connect(lambda on: self._recompute() if on else None)
            agg.addWidget(rb)
            self.agg_buttons[key] = rb
        agg.addStretch(1)
        self.lbl_counts = QtWidgets.QLabel("")
        agg.addWidget(self.lbl_counts)
        lay.addLayout(agg)
        self.lbl_labels = QtWidgets.QLabel("")
        self.lbl_labels.setWordWrap(True)
        lay.addWidget(self.lbl_labels)

        self.canvas = MplCanvas(self, 9.6, 3.6)
        lay.addWidget(self.canvas, 3)
        self.txt = QtWidgets.QPlainTextEdit(readOnly=True)
        self.txt.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.txt.setStyleSheet("font-family: monospace;")
        lay.addWidget(self.txt, 2)
        self.lbl_apply = QtWidgets.QLabel(
            "Apply writes the knee values into the Autonomic view's ks / kp boxes; it does not "
            "recompute anything until you press Compute there (and then Compute event-locked).")
        self.lbl_apply.setWordWrap(True)
        self.lbl_apply.setProperty("muted", True)
        lay.addWidget(self.lbl_apply)

        bottom = QtWidgets.QHBoxLayout()
        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_apply.setProperty("primary", True)
        self.btn_copy = QtWidgets.QPushButton("Copy results")
        self.btn_export = QtWidgets.QPushButton("Export cohort CSV…")
        for b in (self.btn_apply, self.btn_copy, self.btn_export):
            bottom.addWidget(b)
        self.lbl_status = QtWidgets.QLabel("")
        bottom.addWidget(self.lbl_status, 1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.close)
        bottom.addWidget(close)
        lay.addLayout(bottom)

        self.cb_label.currentIndexChanged.connect(lambda _i: self.reload())
        self.btn_load.clicked.connect(self.reload)
        self.ed_project.returnPressed.connect(self.reload)
        self.cb_config.currentIndexChanged.connect(lambda _i: self._recompute())
        self.btn_apply.clicked.connect(self.apply)
        self.btn_copy.clicked.connect(self.copy)
        self.btn_export.clicked.connect(self.export)
        self.reload()

    # ── data ────────────────────────────────────────────────
    def filters(self) -> "tuple[Optional[str], Optional[str]]":
        lab = self.cb_label.currentText()
        proj = self.ed_project.text().strip()
        return (None if lab == ALL else lab), (proj or None)

    def aggregation(self) -> str:
        return next(k for k, rb in self.agg_buttons.items() if rb.isChecked())

    def reload(self) -> None:
        """Read the registry with the label / project filters, list its
        configurations, then recompute."""
        label, project = self.filters()
        self._rows = list_event_rows(label=label, project_name=project, valid_only=True)
        self._frame = registry_frame(self._rows)
        counts = _label_counts(project)
        pooled = label is None and len(counts) > 1
        self.lbl_labels.setText(
            ("Valid events per label: " + ", ".join(f"{lab} ({n})" for lab, n in counts.items())
             + ("   ⚠ “All labels” pools different condition contrasts — the paper "
                "calibrated on one (rest vs dobutamine)." if pooled else ""))
            if counts else "")
        self.lbl_labels.setStyleSheet(f"color: {theme.current().ORANGE};" if pooled else "")
        # The chosen configuration from the *old* name map: names embed the
        # event count, which changes with the filters.
        current = self._configs.get(self.cb_config.currentText())
        self._configs = {f"{config_label(cfg)}   ({n} events)": cfg
                         for cfg, n in config_options(self._frame)}
        names = list(self._configs) or ["—"]
        pick = next((n for n, c in self._configs.items() if c == current), None)
        if pick is None and self._want_config is not None:
            pick = next((n for n, c in self._configs.items() if c == self._want_config), None)
        self.cb_config.blockSignals(True)
        self.cb_config.clear()
        self.cb_config.addItems(names)
        self.cb_config.setCurrentText(pick or names[0])
        self.cb_config.blockSignals(False)
        try:                                    # labels saved since the dialog opened
            keep = self.cb_label.currentText()
            self.cb_label.blockSignals(True)
            self.cb_label.clear()
            self.cb_label.addItems([ALL] + list_event_labels())
            self.cb_label.setCurrentText(keep)
            self.cb_label.blockSignals(False)
        except Exception as exc:
            log.debug("CalibrationDialog: label list: %s", exc)
        self._recompute()

    def _recompute(self) -> None:
        cfg = self._configs.get(self.cb_config.currentText())
        try:
            cal = cohort_calibration(self._frame, self._ks, self._kp,
                                     aggregation=self.aggregation(), config=cfg)
        except Exception as exc:
            log.exception("CalibrationDialog: calibration failed")
            self.txt.setPlainText(f"  Calibration failed: {exc}")
            return
        self.cal = cal
        c = cal.counts
        self.lbl_counts.setText(f"n events = {c['n_events']}  ·  n subjects = "
                                f"{c['n_subjects']}  ·  n recordings = {c['n_recordings']}")
        self.lbl_counts.setStyleSheet("" if c["n_events"] else
                                      f"color: {theme.current().ORANGE};")
        self.canvas.update_figure(lambda fig: draw_calibration(fig, cal))
        if not self._rows:
            label, project = self.filters()
            self.txt.setPlainText(NO_ROW_TEXT.format(
                filt=" matching the filters" if (label or project) else ""))
        else:
            self.txt.setPlainText(calibration_text(cal))
        ks, kp, text, enabled = apply_plan(cal)
        self._apply_values = (ks, kp)
        self.btn_apply.setText(text)
        self.btn_apply.setEnabled(enabled)

    # ── actions ─────────────────────────────────────────────
    def apply(self) -> None:
        if self.cal is not None:
            caveats = apply_caveats(self.cal)
            if caveats and QtWidgets.QMessageBox.warning(
                    self, "Apply these gains?",
                    "\n\n".join(caveats) + "\n\nWrite them into the Autonomic view anyway?",
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No,
            ) != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        ks, kp = self._apply_values
        if self._on_apply is not None:
            self._on_apply(ks, kp)
        self._ks, self._kp = float(ks), float(kp)
        self._recompute()
        ins = in_sample(self.cal) if self.cal is not None else {}
        p = theme.current()
        self.lbl_apply.setText(
            f"✓ ks = {ks:g}, kp = {kp:g} written to the Autonomic view — not applied yet: "
            "press Compute there to recompute CSI / CPI. The tests below are now at these gains"
            + (" — IN-SAMPLE (chosen on these same events, so their p-values are optimistic): "
               "select another project or label to test them." if any(ins.values()) else "."))
        self.lbl_apply.setStyleSheet(f"color: {p.ORANGE if any(ins.values()) else p.GREEN};")

    def copy(self) -> None:
        if self.cal is None:
            return
        df = tests_frame(self.cal)
        host = getattr(self.parent(), "figures", None)     # the Tk TSV (HelpersMixin)
        tsv = (host._df_to_tsv(df) if hasattr(host, "_df_to_tsv")
               else df.to_csv(sep="\t", index=False))
        knees = "\t".join(f"knee_{k}\t{v:g}" for k, v in self.cal.knees.items())
        QtWidgets.QApplication.clipboard().setText(tsv + "\n" + knees)
        self.lbl_status.setText("Results copied as TSV — paste into Excel ✓")

    def export(self, path: "Optional[str]" = None) -> None:
        cfg = self._configs.get(self.cb_config.currentText())
        rows = [r for r in self._rows if cfg is None or row_config(r) == cfg]
        if not rows:
            QtWidgets.QMessageBox.information(self, "Nothing to export",
                                              "No registry row matches the filters.")
            return
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path, _f = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export the cohort's event-locked rows as CSV",
                f"event_locked_cohort_{ts}.csv", "CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            n = export_event_rows_csv(path, rows)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.lbl_status.setText(f"Exported {n} row(s) → {Path(path).name} ✓")
