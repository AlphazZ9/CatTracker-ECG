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

    w.tabs.setCurrentWidget(w.report)
    tab = w.report
    assert tab.provenance.rowCount() > 3 and tab.lines.rowCount() >= 5
    assert "RMSSD" in tab.text.toPlainText()

    targets = iter([tmp_path / "r.xlsx", tmp_path / "r.pdf", tmp_path / "figs.zip",
                    tmp_path / "rr.csv"])
    monkeypatch.setattr(figure_host.QtFileDialog, "asksaveasfilename",
                        classmethod(lambda cls, **k: str(next(targets))))
    monkeypatch.setattr(figure_host.QtFileDialog, "askdirectory",
                        staticmethod(lambda **k: str(tmp_path)))
    for method in ("export_excel", "export_pdf_report", "export_zip", "export_rr_csv"):
        tab.buttons[method].click()
        qtbot.wait(200)
    w.jobs.wait(60000)
    qtbot.wait(500)
    wb = openpyxl.load_workbook(tmp_path / "r.xlsx", read_only=True)
    assert len(wb.sheetnames) >= 3
    assert (tmp_path / "r.pdf").stat().st_size > 10_000
    assert zipfile.ZipFile(tmp_path / "figs.zip").namelist()
    assert (tmp_path / "rr.csv").read_text().count("\n") > 1000
