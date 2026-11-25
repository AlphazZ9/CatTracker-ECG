"""
ecg_qt_app.py -- entry point of the Qt interface (PySide6), as ecg_app.py is
of the Tk one. It is the entry script of the standalone Qt apps built by
PyInstaller (packaging/ecg_qt.spec).

    python ecg_qt_app.py [recording.mat]
    python ecg_qt_app.py --self-test result.txt     # packaging check
"""
import multiprocessing
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)


if __name__ == "__main__":
    # First thing: in a frozen build each 'spawn' worker of
    # ecg.core.delineation_pool re-runs this executable (see ecg_app.py).
    multiprocessing.freeze_support()
    from ecg.qt.app import main
    sys.exit(main())
