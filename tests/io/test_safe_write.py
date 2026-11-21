"""Workbooks are written safely (ecg.io.safe_write): never half-written, the
group table's previous version kept as .bak, and a file open in Excel
reported as such with nothing changed."""
from __future__ import annotations

import os

import pandas as pd
import pytest

from ecg.io import safe_write as sw
from ecg.io.safe_write import FileLockedError, replacing


def _leftovers(folder):
    return [f for f in os.listdir(folder) if f.startswith(".~tmp-")]


def test_replaces_in_one_step_and_keeps_a_backup(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("old")
    with replacing(str(p), backup=True) as tmp:
        with open(tmp, "w") as fh:
            fh.write("new")
        assert p.read_text() == "old"                 # untouched until the end
    assert p.read_text() == "new"
    assert (tmp_path / "t.txt.bak").read_text() == "old"
    assert not _leftovers(tmp_path)


def test_an_error_half_way_changes_nothing(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("old")
    with pytest.raises(RuntimeError):
        with replacing(str(p), backup=True) as tmp:
            with open(tmp, "w") as fh:
                fh.write("half")
            raise RuntimeError("disk full")
    assert p.read_text() == "old" and not (tmp_path / "t.txt.bak").exists()
    assert not _leftovers(tmp_path)


def test_a_file_open_in_excel_is_reported_and_left_as_it_was(tmp_path, monkeypatch):
    p = tmp_path / "group.xlsx"
    p.write_bytes(b"old")

    def locked(*_a, **_k):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(sw.os, "replace", locked)
    with pytest.raises(FileLockedError, match="group.xlsx is open in another program.*"
                                              "Nothing was changed"):
        with replacing(str(p)) as tmp:
            with open(tmp, "wb") as fh:
                fh.write(b"new")
    assert p.read_bytes() == b"old" and not _leftovers(tmp_path)


def test_the_group_table_keeps_its_previous_version(tmp_path):
    from ecg.io import group_data as gd
    p = str(tmp_path / "g.xlsx")
    one = {gd.SHEET_WHOLE: pd.DataFrame([{"Mean HR (bpm)": 500.0}])}
    gd.add_recording(p, "1", "F", "a.smr", one)
    assert not os.path.exists(p + ".bak")
    gd.add_recording(p, "2", "M", "b.smr", one)
    old = gd.animals_frame(gd.read_sheets(p + ".bak"))
    assert list(old["subject"]) == ["1"]
    assert list(gd.animals_frame(gd.read_sheets(p))["subject"]) == ["1", "2"]
