"""Qt tests run headless (offscreen platform) unless a display is asked
for; they are skipped where PySide6 / pyqtgraph / pytest-qt are missing."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyqtgraph")
pytest.importorskip("pytestqt")
