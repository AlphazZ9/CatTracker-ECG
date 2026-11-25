"""Entry point — run with:  python -m ecg  [recording.mat]

Starts the Qt interface (ecg.qt). The CustomTkinter interface it replaced
was removed in version 7."""
from __future__ import annotations

import multiprocessing
import os
import sys


def _ensure_std_streams() -> None:
    """A windowed frozen app (PyInstaller, no console) starts with
    sys.stdout / sys.stderr set to None; a library that prints (progress
    bars, warnings) would then raise. Point them at the null device."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def _set_windows_app_id() -> None:
    """Group the app's windows under its own taskbar entry and icon on
    Windows, instead of python.exe's / pythonw.exe's (run from source or the
    one-click launcher). No-op elsewhere."""
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "TomECG.Analysis")
    except Exception:
        pass


def main(argv: "list[str] | None" = None) -> int:
    # Before anything else: in a frozen build a 'spawn' worker process of
    # ecg.core.delineation_pool re-runs the executable, and this turns that
    # run into the worker rather than a second app. No-op under plain Python.
    multiprocessing.freeze_support()
    _ensure_std_streams()
    _set_windows_app_id()
    from ecg.qt.app import main as qt_main
    return qt_main(argv)


if __name__ == "__main__":
    sys.exit(main())
