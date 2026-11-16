"""Entry point — run with:  python -m ecg"""
from __future__ import annotations
import multiprocessing
import os
import sys
import tkinter as tk
from tkinter import messagebox


def _ensure_std_streams() -> None:
    """A windowed frozen app (PyInstaller, no console) starts with
    sys.stdout / sys.stderr set to None; a library that prints (progress
    bars, warnings) would then raise. Point them at the null device."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def main() -> None:
    # Before anything else: in a frozen build a 'spawn' worker process of
    # ecg.core.delineation_pool re-runs the executable, and this turns that
    # run into the worker rather than a second app. No-op under plain Python.
    multiprocessing.freeze_support()
    _ensure_std_streams()
    from ecg.ui.theme import NK_AVAILABLE
    if not NK_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Missing dependency",
            "Install NeuroKit2:\n  pip install neurokit2",
        )
        root.destroy()
        return
    from ecg.ui.app import ECGApp
    ECGApp().mainloop()


if __name__ == "__main__":
    main()
