# -*- coding: utf-8 -*-
"""
ecg.app.jobs
------------
The shared guard of background jobs: a result that lands after another
recording was opened is dropped. Moved unchanged from the Tk app_async.
"""
from __future__ import annotations

import logging
from typing import Any

from ecg.app.theme import ORANGE

log = logging.getLogger("ecg")


def recording_changed(app: Any, gen: int, what: str, msg: "str | None" = None) -> bool:
    """True when the recording is no longer the one a job started on.

    The one shared guard for background jobs: *gen* is ``app._generation``
    captured on the main thread when the job started; reset_for_new_file()
    (opening another file) and a fresh load / Detect Peaks bump it. A job
    whose result lands after that must not write it into the new
    recording's state -- it would show (and later save) the old file's
    signal, peaks or results under the new file's name. Logs the drop and
    says so in orange (*msg*, or a generic "discarded" line).

    Module-level (not a mixin method) so controllers driven through a stub
    app in the tests can use it without the stub defining it.
    """
    if getattr(app, "_generation", 0) == gen:
        return False
    log.info("%s: stale result discarded (another recording was opened)", what)
    set_status = getattr(app, "_set_status", None)
    if callable(set_status):
        set_status(msg or f"{what} discarded — another recording was opened while it ran",
                   ORANGE)
    return True

