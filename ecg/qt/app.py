# -*- coding: utf-8 -*-
"""
ecg.qt.app -- start the Qt interface.

    python -m ecg.qt [recording.mat]
"""
from __future__ import annotations

import multiprocessing
import sys
from typing import Optional


def main(argv: "Optional[list[str]]" = None) -> int:
    # First thing: in a frozen build each 'spawn' worker of
    # ecg.core.delineation_pool re-runs the executable; this turns that run
    # into the worker. No-op under plain Python.
    multiprocessing.freeze_support()
    argv = list(sys.argv if argv is None else argv)

    import pyqtgraph as pg
    from PySide6 import QtWidgets

    from ecg.qt import theme
    from ecg.qt.main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setApplicationName("ECG Analysis")
    pg.setConfigOptions(antialias=False, useOpenGL=False)
    theme.apply(app)
    from ecg.qt.figure_host import sync_figure_theme
    sync_figure_theme(theme.current().is_dark)
    win = MainWindow()
    win.btn_theme.setText("Light mode" if theme.current().is_dark else "Dark mode")
    win.show()
    if len(argv) > 1:
        win.open_path(argv[1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
