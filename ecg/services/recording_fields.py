# -*- coding: utf-8 -*-
"""
ecg.services.recording_fields
-----------------------------
The Subject ID and Sex fields filled from the recording's file name
(ecg.io.subject_name), on any object holding them under the Tk names:
ent_subject / ent_sex (get, delete(0, "end"), insert(0, text)) and the
values last filled, _subject_autofill / _sex_autofill.

Moved unchanged from the Tk SignalController.autofill_subject_from_name,
so both interfaces fill (and keep) the same values.
"""
from __future__ import annotations

import logging

from ecg.io.subject_name import DEFAULT_SUBJECT, is_default_subject, subject_from_filename

log = logging.getLogger("ecg")


def autofill_subject_from_name(app, path: str) -> str:
    """Fill the Subject ID and Sex entries from the name of *path* ("Stim
    BNST - 1146 F.smr" gives 1146 and F), each one only while it holds what
    the app put there: empty, the placeholder ("subject_01"), or the value
    auto-filled for the previous file. A value typed by the user is never
    replaced. An auto-filled subject the new name does not give goes back to
    the placeholder, an auto-filled sex to empty: they belonged to the
    previous file. Returns the "Kept as typed" note ("" if none), also
    stored as app._subject_kept_note."""
    subject, sex = subject_from_filename(path)
    kept: "list[str]" = []      # typed values the name contradicts
    ent = getattr(app, "ent_subject", None)
    if ent is not None:
        try:
            cur = ent.get().strip()
            auto = getattr(app, "_subject_autofill", None)
            was_auto = bool(auto) and cur == auto
            if not cur or is_default_subject(cur) or was_auto:
                new = subject or (DEFAULT_SUBJECT if was_auto else cur)
                if new != cur:
                    ent.delete(0, "end")
                    ent.insert(0, new)
                app._subject_autofill = subject
            elif subject and cur != subject:
                kept.append(f"subject {cur} (the file name says {subject})")
        except Exception as exc:
            log.debug("Could not fill the subject from the file name: %s", exc)
    ent = getattr(app, "ent_sex", None)
    if ent is not None:
        try:
            cur = ent.get().strip()
            auto = getattr(app, "_sex_autofill", None)
            if not cur or (bool(auto) and cur == auto):
                new = sex or ""
                if new != cur:
                    ent.delete(0, "end")
                    if new:
                        ent.insert(0, new)
                app._sex_autofill = sex
            elif sex and cur.upper() != sex:
                kept.append(f"sex {cur} (the file name says {sex})")
        except Exception as exc:
            log.debug("Could not fill the sex from the file name: %s", exc)
    app._subject_kept_note = ""
    if kept:
        # The typed value stays, but say so: it may belong to the previous animal.
        app._subject_kept_note = "Kept as typed: " + "; ".join(kept) + "."
    return app._subject_kept_note
