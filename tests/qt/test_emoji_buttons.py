"""The emoji beside the buttons' labels (ecg.qt.widgets.emoji): chosen by
the label, drawn as an icon, the label itself unchanged (bar a leading
arrow the emoji replaces), given to the buttons of the window and of the
dialogs as they are first shown."""
import pytest
from PySide6 import QtWidgets

from ecg.qt.widgets import emoji

@pytest.fixture(autouse=True)
def _need_font(qapp):
    if emoji.emoji_font() is None:
        pytest.skip("no colour emoji font on this system")


def test_emoji_by_label():
    assert emoji.emoji_for("Open") == "📂"
    assert emoji.emoji_for("Export…") == emoji.emoji_for("Export") == "📤"
    assert emoji.emoji_for("↶ Undo") == "↩️"
    assert emoji.emoji_for("Compare segments →") == "⚖️"
    assert emoji.emoji_for("Add segment") == "➕"                 # by its first word
    assert emoji.emoji_for("Remove selected") == "🗑️"
    assert emoji.emoji_for("&Open") == "📂"                      # a mnemonic is not text
    for plain in ("Filtered", "2 s", "▸ Text report", "", "OK"):
        assert emoji.emoji_for(plain) is None


def test_decorate_keeps_the_label_and_follows_a_new_one(qtbot):
    b = QtWidgets.QPushButton("Detect Peaks")
    qtbot.addWidget(b)
    assert emoji.decorate(b) and not b.icon().isNull() and b.text() == "Detect Peaks"
    undo = QtWidgets.QPushButton("↶ Undo")
    qtbot.addWidget(undo)
    assert emoji.decorate(undo) and undo.text() == "Undo"      # the arrow gives way
    theme = QtWidgets.QPushButton("Dark mode")
    qtbot.addWidget(theme)
    emoji.decorate(theme)
    theme.setText("Light mode")
    assert emoji.decorate(theme) and theme.property("emoji_set")
    theme.setText("Filtered")                                  # a label without emoji
    assert not emoji.decorate(theme) and theme.icon().isNull()


def test_left_alone(qtbot):
    from PySide6 import QtGui
    own = QtWidgets.QPushButton("Open")
    own.setIcon(QtGui.QIcon(QtGui.QPixmap(8, 8)))             # an icon of its own
    out = QtWidgets.QPushButton("Open")
    out.setProperty("emoji", False)                            # opted out
    box = QtWidgets.QCheckBox("Open")
    for w in (own, out, box):
        qtbot.addWidget(w)
        assert not emoji.decorate(w)
    assert out.icon().isNull() and box.icon().isNull()


def test_tool_buttons_keep_their_label(qtbot):
    tb = QtWidgets.QToolButton()
    tb.setText("Export…")
    qtbot.addWidget(tb)
    assert emoji.decorate(tb)
    assert tb.toolButtonStyle() == emoji.QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon


def test_window_and_dialog_buttons_get_theirs_when_shown(qtbot):
    from ecg.qt.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    qtbot.wait(50)
    assert w.btn_undo_top.property("emoji_set") and w.btn_undo_top.text() == "Undo"
    decorated = [b for b in w.findChildren(QtWidgets.QAbstractButton) if b.property("emoji_set")]
    assert len(decorated) > 40
    dlg = QtWidgets.QDialog(w)
    lay = QtWidgets.QVBoxLayout(dlg)
    later = QtWidgets.QPushButton("Copy table")
    lay.addWidget(later)
    dlg.show()
    qtbot.wait(20)
    assert later.property("emoji_set") and later.text() == "Copy table"
    dlg.close()
