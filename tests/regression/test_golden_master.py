"""
Reference ("golden master") outputs of the whole pipeline (tests/regression/
golden.py): what version 6.2.0 computes must still be computed.

* The synthetic recording of tests/fixtures/ is checked in every run against
  the reference committed in tests/regression/reference/.
* Real recordings are checked when the environment variable ECG_GOLDEN_DIR
  names a folder holding the .mat files and, in its reference/ subfolder,
  their reference made with ``python tools/golden_master.py record``. They
  never go into the repository.

A difference here means the numbers changed. When that is intended (a
documented scientific correction), the reference is made again and the
change is written in docs/validation.md -- never by loosening the tolerance.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ecg.core import analysis
from tests.regression import golden

pytestmark = pytest.mark.skipif(not analysis.NK_AVAILABLE, reason="neurokit2 not installed")

HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parent / "fixtures" / "synthetic_mouse_ecg.mat"
REFERENCE_DIR = HERE / "reference"
MAX_SHOWN = 40


def _assert_identical(recording: Path, ref_dir: Path) -> None:
    """Fail on any difference -- unless a scientific library is not the
    version the reference was made with (the CI job on the newest
    libraries): then a difference may be the library's own change, and the
    test is an expected failure that lists the versions and the outputs."""
    diffs, manifest = golden.check(recording, ref_dir)
    if not diffs:
        return
    shown = "\n".join(str(d) for d in diffs[:MAX_SHOWN])
    more = f"\n… and {len(diffs) - MAX_SHOWN} more" if len(diffs) > MAX_SHOWN else ""
    report = f"{recording.name}: {len(diffs)} output(s) differ from the reference:\n{shown}{more}"
    changed = golden.package_changes(manifest)
    if changed:
        versions = ", ".join(f"{p} {a} -> {b}" for p, (a, b) in changed.items())
        pytest.xfail(f"library versions differ from the reference ({versions}); {report}")
    pytest.fail(report)


def test_synthetic_recording_matches_its_reference():
    _assert_identical(FIXTURE, REFERENCE_DIR)


def test_synthetic_fixture_is_the_committed_file():
    """The fixture is read, not regenerated: its SHA-256 is the one the
    reference was recorded from."""
    _, manifest = golden.load(REFERENCE_DIR, "synthetic_mouse_ecg")
    assert golden.file_sha256(FIXTURE) == manifest["recording"]["sha256"]


def _real_recordings() -> "list[Path]":
    folder = os.environ.get("ECG_GOLDEN_DIR")
    if not folder:
        return []
    return sorted(p for p in Path(folder).glob("*.mat")
                  if (Path(folder) / "reference" / f"{p.stem}.json").exists())


@pytest.mark.parametrize("recording", _real_recordings(), ids=lambda p: p.name)
def test_real_recording_matches_its_reference(recording: Path):
    _assert_identical(recording, recording.parent / "reference")


# ── The comparison itself ───────────────────────────────────────

def test_compare_accepts_identical_outputs_and_nan():
    flat = {"a": np.array([1.0, np.nan, np.inf]), "n": 3, "s": "x", "x": float("nan")}
    assert golden.compare(flat, dict(flat)) == []


def test_compare_reports_a_float_change_beyond_the_tolerance():
    ref = {"a": np.array([1.0, 2.0])}
    assert golden.compare(ref, {"a": np.array([1.0, 2.0 * (1 + 1e-12)])}) == []
    diffs = golden.compare(ref, {"a": np.array([1.0, 2.0 * (1 + 1e-6)])})
    assert [d.kind for d in diffs] == ["value"]


def test_compare_wants_integers_exactly():
    diffs = golden.compare({"peaks": np.array([10, 20, 30])}, {"peaks": np.array([10, 21, 30])})
    assert len(diffs) == 1 and diffs[0].kind == "value"


def test_compare_reports_missing_extra_shape_and_type():
    ref = {"gone": 1, "p": np.array([1, 2]), "f": np.array([1.0]), "s": np.array(["a"])}
    new = {"new": 1, "p": np.array([1, 2, 3]), "f": np.array([1]), "s": np.array([1.0])}
    kinds = sorted(d.kind for d in golden.compare(ref, new))
    assert kinds == ["extra", "missing", "shape", "type", "type"]


def test_signal_values_use_an_absolute_tolerance_on_the_signal_scale():
    """A sample near zero of the z-scored signal: 1e-11 off is rounding, not
    a change (seen between Linux and Windows); a non-signal value that close
    to zero must still match to ATOL."""
    ref = {"detect/x/signal_flt/samples": np.array([1e-3, 0.5]), "hr/sd": np.array([1e-3])}
    new = {"detect/x/signal_flt/samples": np.array([1e-3 + 1e-11, 0.5]),
           "hr/sd": np.array([1e-3 + 1e-11])}
    assert [d.name for d in golden.compare(ref, new)] == ["hr/sd"]
    big = {"detect/x/signal_flt/samples": np.array([1e-3 + 1e-6, 0.5]), "hr/sd": ref["hr/sd"]}
    assert [d.name for d in golden.compare(ref, big)] == ["detect/x/signal_flt/samples"]


def test_flatten_keeps_long_signals_as_a_stride_and_statistics():
    sig = np.arange(golden.SIGNAL_MIN_LEN + 5, dtype=float)
    flat = golden.flatten({"sig": sig})
    assert len(flat["sig/samples"]) <= golden.SIGNAL_POINTS
    assert flat["sig/len"] == len(sig) and flat["sig/max"] == sig[-1]


def test_save_and_load_round_trip(tmp_path):
    flat = {"a/b": np.array([1.5, np.nan]), "c": float("nan"), "d": "text", "e": None,
            "f": np.array(["x", "y"]), "g": 7}
    golden.save(flat, tmp_path, "rec", {"recording": {"sha256": "0"}})
    back, _ = golden.load(tmp_path, "rec")
    assert golden.compare(flat, back) == []
