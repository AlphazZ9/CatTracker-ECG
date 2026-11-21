# -*- coding: utf-8 -*-
"""
ecg.io.safe_write
-----------------
Writing a file the user may have open, safely:

* the new content goes to a temporary file in the same folder, which then
  replaces the target in one step (``os.replace``): a crash, a full disk or
  an exception half-way never leaves a half-written workbook;
* with *backup*, the previous version is kept beside it as ``<name>.bak``
  (the group table: weeks of recordings in one file);
* a file locked by another program (Excel keeps a workbook it shows locked
  on Windows) raises FileLockedError, whose message says to close it and
  that nothing was changed, instead of "[Errno 13] Permission denied".

Nothing here may import from ecg.ui.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from typing import Iterator


class FileLockedError(PermissionError):
    """The target is open in another program (or not writable)."""

    def __init__(self, path: str) -> None:
        name = os.path.basename(path)
        super().__init__(
            f"{name} is open in another program (Excel?) or cannot be written: close it, "
            f"then try again. Nothing was changed.")
        self.path = path


def backup_path(path: str) -> str:
    return path + ".bak"


@contextlib.contextmanager
def replacing(path: str, backup: bool = False) -> Iterator[str]:
    """Yield a temporary path to write the new *path* to; on leaving the
    block without an exception it replaces *path* (and, with *backup*, the
    old *path* is first copied to ``<path>.bak``). On an exception the
    temporary file goes and *path* is untouched."""
    folder = os.path.dirname(os.path.abspath(path)) or "."
    ext = os.path.splitext(path)[1]
    fd, tmp = tempfile.mkstemp(prefix=".~tmp-", suffix=ext, dir=folder)
    os.close(fd)
    try:
        yield tmp
        if os.path.exists(path):
            # A file Excel holds open cannot be replaced: say so before
            # anything moves (the backup is not refreshed for nothing).
            try:
                with open(path, "r+b"):
                    pass
            except PermissionError:
                raise FileLockedError(path) from None
            if backup:
                try:
                    shutil.copy2(path, backup_path(path))
                except OSError:
                    pass                    # no backup beats no save
        try:
            os.replace(tmp, path)
        except PermissionError:
            raise FileLockedError(path) from None
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
