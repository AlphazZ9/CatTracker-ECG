"""Qt tests run headless (offscreen platform) unless a display is asked
for; they are skipped where PySide6 / pyqtgraph / pytest-qt are missing."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyqtgraph")
pytest.importorskip("pytestqt")


@pytest.fixture(autouse=True)
def _no_unsaved_prompt(request, monkeypatch):
    """Closing a window at the end of a test does not ask about unsaved
    changes (a modal box would block); test_errors tests the prompt."""
    if request.node.get_closest_marker("unsaved_prompt"):
        return
    from ecg.qt.main_window import MainWindow
    monkeypatch.setattr(MainWindow, "confirm_discard", lambda self, action: True)


def pytest_configure(config):
    config.addinivalue_line("markers", "unsaved_prompt: keep MainWindow.confirm_discard")


@pytest.fixture(autouse=True)
def _isolated_profile(tmp_path, monkeypatch):
    """Sessions, the registry and the ML training cache of a Qt test live in
    its own folder: never the user's profile, and no session left by
    another test is offered (a modal "Session found" box would block)."""
    import ecg.core.ml_detector as ml
    import ecg.io.db as db
    import ecg.io.session as io_session
    d = tmp_path / "profile"
    monkeypatch.setattr(io_session, "SESSION_DIR", d / "sessions")
    monkeypatch.setattr(db, "SESSION_DIR", d / "sessions", raising=False)
    monkeypatch.setattr(db, "_DB_PATH", d / "sessions" / "ecg_registry.db")
    monkeypatch.setattr(db, "_schema_ready", False)
    monkeypatch.setattr(db, "_schema_retry_at", 0.0)
    monkeypatch.setattr(ml, "TRAINING_DATA_DIR", d / "ml_training")
    monkeypatch.setattr(ml, "MODEL_PATH", d / "model.skops")
    import ecg.qt.theme as qt_theme
    monkeypatch.setattr(qt_theme, "PREFS_PATH", d / "ecg_qt_prefs.json")
    monkeypatch.setattr(qt_theme, "LEGACY_THEME_PATH", d / "ecg_theme.json")
