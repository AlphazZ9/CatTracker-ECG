"""Report and exports in the Qt interface: the report view, and each export
writes its file (Excel, PDF, figures ZIP, RR CSV) through Qt file dialogs."""
import zipfile
from pathlib import Path

import openpyxl
import pytest

from ecg.core import analysis
from ecg.qt import figure_host
from ecg.qt.main_window import MainWindow

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_mouse_ecg.mat"

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")


def test_report_view_and_exports(qtbot, tmp_path, monkeypatch):
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    w.open_path(str(FIXTURE))
    w.detection.cb_method.setCurrentIndex(0)
    w.detection.detect()
    qtbot.waitUntil(lambda: w.recording.detection.rpeaks_ok is not None, timeout=60000)
    w.analyze()
    qtbot.waitUntil(lambda: w.statusBar().currentMessage() == "Analysis done", timeout=120000)

    # Nothing left running in the background (a figure drawn late, a job
    # still landing) before the exports read the results.
    qtbot.waitUntil(lambda: not w._active, timeout=120000)
    w.jobs.wait(60000)

    w.tabs.setCurrentWidget(w.report)
    tab = w.report
    assert tab.provenance.rowCount() > 3 and tab.lines.rowCount() >= 5
    assert "RMSSD" in tab.text.toPlainText()

    # One file per export, asked for by that export only: an export that
    # stops before its file dialog fails here under its own name, instead of
    # shifting the files of the exports after it.
    monkeypatch.setattr(figure_host.QtFileDialog, "askdirectory",
                        staticmethod(lambda **k: str(tmp_path)))
    for method, name in (("export_excel", "r.xlsx"), ("export_pdf_report", "r.pdf"),
                         ("export_zip", "figs.zip"), ("export_rr_csv", "rr.csv")):
        asked = []
        monkeypatch.setattr(
            figure_host.QtFileDialog, "asksaveasfilename",
            classmethod(lambda cls, _n=name, _a=asked, **k: _a.append(_n) or str(tmp_path / _n)))
        assert tab.buttons[method].isEnabled(), method
        tab.buttons[method].click()
        w.jobs.wait(60000)
        assert asked == [name], (method, asked, w.statusBar().currentMessage())
        qtbot.waitUntil(lambda: (tmp_path / name).exists(), timeout=30000)
    wb = openpyxl.load_workbook(tmp_path / "r.xlsx", read_only=True)
    assert len(wb.sheetnames) >= 3
    assert (tmp_path / "r.pdf").stat().st_size > 10_000
    assert zipfile.ZipFile(tmp_path / "figs.zip").namelist()
    assert (tmp_path / "rr.csv").read_text().count("\n") > 1000
