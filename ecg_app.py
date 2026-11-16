"""
ecg_app.py -- backward-compatible entry point.

The application has been refactored into the `ecg/` package.
This file exists so existing shortcuts / scripts that call
    python ecg_app.py
continue to work unchanged. It is also the entry script of the standalone
apps built by PyInstaller (packaging/ecg.spec).

To run directly from the package:
    python -m ecg
"""
import multiprocessing
import sys
import os

# Ensure the parent directory is on sys.path so `import ecg` works
# regardless of how this script is invoked.
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)


if __name__ == "__main__":
    # First thing, before the app is imported: in a frozen (PyInstaller)
    # build each 'spawn' worker of ecg.core.delineation_pool re-runs this
    # executable, and freeze_support() turns that run into a worker instead
    # of a second copy of the application. A no-op under plain Python.
    multiprocessing.freeze_support()
    from ecg.__main__ import main
    main()
