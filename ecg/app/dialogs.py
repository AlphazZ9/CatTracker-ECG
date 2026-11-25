# -*- coding: utf-8 -*-
"""
ecg.app.dialogs
---------------
The message boxes and file choosers the controllers show, as the Tk
modules they used (tkinter.messagebox, tkinter.filedialog): the same
calls, sent to the interface that installed itself with install().

Before an interface installs one (a script, the tests), a message is
logged and a question answered "no": no window opens.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("ecg")


class _Headless:
    """messagebox / filedialog without a window."""

    @staticmethod
    def _log(kind: str, title: Any = "", message: Any = "", **_kw: Any) -> None:
        log.info("%s: %s — %s", kind, title, message)

    def showinfo(self, title: Any = "", message: Any = "", **kw: Any) -> str:
        self._log("info", title, message)
        return "ok"

    def showwarning(self, title: Any = "", message: Any = "", **kw: Any) -> str:
        self._log("warning", title, message)
        return "ok"

    def showerror(self, title: Any = "", message: Any = "", **kw: Any) -> str:
        self._log("error", title, message)
        return "ok"

    def askyesno(self, title: Any = "", message: Any = "", **kw: Any) -> bool:
        self._log("question", title, message)
        return False

    def askokcancel(self, title: Any = "", message: Any = "", **kw: Any) -> bool:
        return self.askyesno(title, message)

    def askyesnocancel(self, title: Any = "", message: Any = "", **kw: Any) -> "bool | None":
        self._log("question", title, message)
        return None

    def asksaveasfilename(self, **kw: Any) -> str:
        return ""

    def askopenfilename(self, **kw: Any) -> str:
        return ""

    def askopenfilenames(self, **kw: Any) -> tuple:
        return ()

    def askdirectory(self, **kw: Any) -> str:
        return ""


class _Relay:
    """A module-like object whose calls go to the installed backend."""

    def __init__(self) -> None:
        self.backend: Any = _Headless()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.backend, name)


messagebox = _Relay()
filedialog = _Relay()


def _reset(relay: "_Relay") -> None:
    """Drop the functions set on the relay itself: a test's monkeypatch of
    ``filedialog.asksaveasfilename`` puts back, on undo, the function it
    read through the relay -- the backend of that moment -- and it then
    shadowed every later backend (exports silently saved nowhere)."""
    for name in [n for n in vars(relay) if n != "backend"]:
        delattr(relay, name)


def install(message_box: Any = None, file_dialog: Any = None) -> None:
    """Send the controllers' message boxes / file choosers to these objects
    (each with the tkinter function names); None keeps the current one."""
    if message_box is not None:
        _reset(messagebox)
        messagebox.backend = message_box
    if file_dialog is not None:
        _reset(filedialog)
        filedialog.backend = file_dialog


def uninstall() -> None:
    """Back to no window (tests)."""
    _reset(messagebox)
    _reset(filedialog)
    messagebox.backend = _Headless()
    filedialog.backend = _Headless()
