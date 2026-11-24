"""
tools/golden_master.py -- record or check the reference ("golden master")
outputs of the analysis pipeline on real recordings.

Record the reference once, with the version to preserve (6.2.0):

    python tools/golden_master.py record D:/ECG/248_Baseline.mat D:/ECG/247_Footshocks.mat --out D:/ECG/reference

Check any later version against it (exit code 1 when something differs):

    python tools/golden_master.py check D:/ECG/248_Baseline.mat D:/ECG/247_Footshocks.mat --ref D:/ECG/reference

The recordings and their reference stay outside the repository. The pytest
suite checks them too when ECG_GOLDEN_DIR points at a folder holding the
.mat files and their reference/ subfolder (tests/regression/
test_golden_master.py). What is computed and compared: tests/regression/
golden.py and docs/validation.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_SHOWN = 40


def _record(args: argparse.Namespace) -> int:
    from tests.regression import golden
    for rec in args.recordings:
        print(f"{rec}:")
        t0 = time.perf_counter()
        npz, js = golden.record(rec, args.out, channel=args.channel, log=print)
        print(f"  reference written: {js.name} + {npz.name} "
              f"({time.perf_counter() - t0:.0f} s)")
    return 0


def _check(args: argparse.Namespace) -> int:
    from tests.regression import golden
    failed = 0
    for rec in args.recordings:
        print(f"{rec}:")
        t0 = time.perf_counter()
        try:
            diffs, manifest = golden.check(rec, args.ref, channel=args.channel, log=print)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  ERROR: {exc}")
            failed += 1
            continue
        env = manifest.get("environment", {})
        print(f"  reference: version {env.get('app_version')} (commit {env.get('git_commit')}),"
              f" recorded {manifest.get('created')}")
        if diffs:
            failed += 1
            print(f"  {len(diffs)} DIFFERENCE(S):")
            for d in diffs[:MAX_SHOWN]:
                print(f"    {d}")
            if len(diffs) > MAX_SHOWN:
                print(f"    … and {len(diffs) - MAX_SHOWN} more")
        else:
            print(f"  identical to the reference ({time.perf_counter() - t0:.0f} s)")
    return 1 if failed else 0


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="compute and save the reference outputs")
    rec.add_argument("recordings", nargs="+", type=Path)
    rec.add_argument("--out", type=Path, required=True, help="folder for the reference files")
    rec.add_argument("--channel", default="ECG", help="channel name (default: ECG, else auto)")
    rec.set_defaults(func=_record)
    chk = sub.add_parser("check", help="re-compute and compare with the reference")
    chk.add_argument("recordings", nargs="+", type=Path)
    chk.add_argument("--ref", type=Path, required=True, help="folder holding the reference files")
    chk.add_argument("--channel", default=None, help="channel name (default: the recorded one)")
    chk.set_defaults(func=_check)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
