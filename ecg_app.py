"""
ecg_app.py -- entry point of the application (the Qt interface, PySide6).

    python ecg_app.py [recording.mat]
    python ecg_app.py --self-test result.txt [recording.mat]   # packaging check

Kept so existing shortcuts and the one-click launchers ("Lancer ECG") keep
working; it is also the entry script of the standalone apps built by
PyInstaller (packaging/ecg.spec). Same as  python -m ecg.
"""
import multiprocessing
import os
import sys

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
    sys.exit(main())
