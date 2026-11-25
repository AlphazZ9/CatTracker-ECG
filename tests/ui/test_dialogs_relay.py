"""A function set on the dialog relay (a test's monkeypatch, put back on
undo) never shadows the backend installed afterwards."""
from ecg.app import dialogs


def test_install_drops_functions_set_on_the_relay():
    class Saver:
        @staticmethod
        def asksaveasfilename(**_k):
            return "/tmp/x.xlsx"

    stale = dialogs.filedialog.asksaveasfilename          # the headless one, read through
    dialogs.filedialog.asksaveasfilename = stale          # what a monkeypatch undo leaves
    try:
        dialogs.install(file_dialog=Saver)
        assert dialogs.filedialog.asksaveasfilename() == "/tmp/x.xlsx"
    finally:
        dialogs.uninstall()
    assert dialogs.filedialog.asksaveasfilename() == ""
