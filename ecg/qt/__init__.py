"""
ecg.qt -- the PySide6 interface (migration from CustomTkinter, phase 2).

Run with ``python -m ecg.qt``. It uses the same scientific code as the
CustomTkinter interface (ecg.core, ecg.io, ecg.services), which stays
available (``python -m ecg``) until the migration is validated.

Layout
    app.py           QApplication, theme, main window
    main_window.py   header, tabs, progress
    theme.py         light / dark palettes (the Tk theme's colours) + style sheet
    workers.py       background jobs: QThreadPool, progress, cancellation
    widgets/         ECG plot (PyQtGraph), Matplotlib canvas
    tabs/            one module per tab

Nothing in ecg.core / ecg.io / ecg.services imports this package.
"""
