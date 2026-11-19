# -*- coding: utf-8 -*-
"""
ecg.ui.ml_examples
------------------
Plain-language wording of the ML detector's state, kept free of Tk so it can
be tested on its own: the status line of the quality panel's ML DETECTOR
section (short form for its header, full form under it), the one training
example button, and the tooltips that explain the terms.

A *training example* is a recording whose corrected R-peaks were cached as
ML training data (ecg.core.ml_detector, one .npz per recording); the
registry's verified_for_training flag and SessionState.verified_for_training
mirror "this recording is a training example".
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ecg.core.ml_detector import MIN_EXAMPLES_RECOMMENDED, MIN_EXAMPLES_REQUIRED

TONE_OK = "ok"
TONE_MUTED = "muted"
TONE_WARN = "warn"

# One sentence per term (hover tooltips of the ML DETECTOR section).
TIP_HEADER = ("Click to show or hide the ML detector: it learns R-peaks from "
              "recordings whose peaks you corrected (the training examples).")
TIP_STATUS = ("Training examples are recordings whose corrected peaks the model "
              "learns from; held-out data are examples kept aside to check it.")
TIP_ADD = ("Saves this recording's current R-peaks as a training example right "
           "away — correct the peaks first (any method).")
TIP_REMOVE = "Takes this recording out of the training examples."
TIP_UPDATE = ("The peaks were edited after this recording was added: saves the "
              "current peaks in its place.")
TIP_TRAIN = ("How to train, the list of training examples, and the Train button "
             "(retrain after adding or removing examples).")

# Method = ML Detector without a trained model. The status bar is too narrow
# on a laptop screen for the whole how-to: it only points at the ML DETECTOR
# section, whose status line says what to do (status_text(ml_selected=True)).
NO_MODEL_STATUS = "ML Detector: no model trained yet — see ML DETECTOR (Signal quality)"

# A training example cannot be removed while a job runs (it may be writing it).
BUSY_MESSAGE = "Another operation is already running — please wait"

BTN_ADD = "Add recording to training examples"
BTN_REMOVE = "✓ In training examples — Remove"
BTN_UPDATE = "↻ Update example (peaks edited)"
BTN_REMOVE_SHORT = "Remove"
BTN_TRAIN = "Train model…"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def changes_since_training(examples: "list[dict]", meta: "Optional[dict]") -> "tuple[int, int]":
    """(new or updated, removed) examples since the model was trained.

    New/updated: cached after the model's trained_at. Removed: in the model's
    training_fingerprints (models trained before that key: unknown, 0)."""
    if not meta:
        return 0, 0
    try:
        trained = datetime.fromisoformat(str(meta.get("trained_at", ""))).timestamp()
    except ValueError:
        trained = None
    n_new = 0
    if trained is not None:
        n_new = sum(1 for e in examples if float(e.get("mtime", 0.0)) > trained)
    seen = meta.get("training_fingerprints")
    n_removed = 0
    if isinstance(seen, list):
        current = {e.get("fingerprint") for e in examples}
        n_removed = sum(1 for fp in seen if fp not in current)
    return n_new, n_removed


def status_text(examples: "list[dict]", meta: "Optional[dict]",
                ml_selected: bool = False) -> "tuple[str, str, str]":
    """(short, full, tone) for the ML DETECTOR section.

    *examples*: list_training_files(); *meta*: the trained model's metadata,
    None without a model; *ml_selected*: Method = ML Detector (without a
    model, the full line then says what to do instead of detecting)."""
    n = len(examples)
    ex = _plural(n, "training example")
    if meta is None:
        if ml_selected:
            # Three lines in the panel: the header gives the number of examples.
            full = ("ML Detector needs a trained model: detect and correct the "
                    "peaks with another Method, add the recording here; with "
                    f"{MIN_EXAMPLES_RECOMMENDED} or more examples, click Train.")
            return f"No model yet · {_plural(n, 'example')}", full, TONE_WARN
        if n < MIN_EXAMPLES_RECOMMENDED:
            full = (f"No model yet · {ex} — add at least {MIN_EXAMPLES_RECOMMENDED}, "
                    "then Train")
        else:
            full = f"No model yet · {ex} — ready to Train"
        return f"No model yet · {_plural(n, 'example')}", full, TONE_MUTED

    k = meta.get("n_training_files", "?")
    acc = meta.get("holdout_accuracy")
    trained = f"model trained on {k}"
    if isinstance(acc, (int, float)):
        trained += (f" ({acc * 100:.0f} % of candidate peaks classified correctly "
                    "on held-out data)")
    parts = [ex, trained]
    n_new, n_removed = changes_since_training(examples, meta)
    tone = TONE_OK
    short = f"Model ready · {_plural(n, 'example')}"
    if n_new or n_removed:
        why = []
        if n_new:
            why.append(f"{n_new} new or updated example{'' if n_new == 1 else 's'}")
        if n_removed:
            why.append(f"{n_removed} removed")
        parts.append("retrain suggested: " + ", ".join(why))
        short = f"Retrain suggested · {_plural(n, 'example')}"
        tone = TONE_WARN
    return short, " · ".join(parts), tone


def panel_text(examples: "list[dict]", meta: "Optional[dict]",
               ml_selected: bool = False) -> str:
    """The line under the ML DETECTOR header: what to do next, or how good the
    model is, in one line where it can be. The header already gives the
    model's state and the number of examples (status_text's short form), and
    every row here is taken from the quality plots on a 768-px screen; the
    full status_text line is its hover tooltip and the training dialog's.

    Method = ML Detector without a model keeps the whole how-to: there are no
    peaks (nor plots) to make room for then."""
    if meta is None:
        if ml_selected:
            return status_text(examples, meta, ml_selected=True)[1]
        if len(examples) < MIN_EXAMPLES_RECOMMENDED:
            return f"Add at least {MIN_EXAMPLES_RECOMMENDED} training examples, then Train."
        return "Enough training examples: ready to Train."
    k = meta.get("n_training_files", "?")
    acc = meta.get("holdout_accuracy")
    if isinstance(acc, (int, float)):
        return f"Trained on {k} · {acc * 100:.0f} % correct on held-out data"
    return f"Model trained on {k} training examples"


@dataclass(frozen=True)
class ExampleButton:
    """The training-example action for the open recording."""
    mode: str            # "add" | "remove" | "update" | "none"
    text: str
    enabled: bool
    hint: str            # shown under the button ("" for none)
    tip: str
    show_remove: bool    # the separate small Remove next to Update


def example_button(has_file: bool, n_peaks: int, state: str) -> ExampleButton:
    """*state*: ml_detector.training_example_state() for the open recording
    ("absent" / "same" / "changed" / "unknown")."""
    if state in ("same", "unknown"):
        return ExampleButton("remove", BTN_REMOVE, True, "", TIP_REMOVE, False)
    if state == "changed" and n_peaks > 0:
        # No hint row: the label already says "peaks edited" (TIP_UPDATE
        # explains), and every row here is taken from the plots.
        return ExampleButton("update", BTN_UPDATE, True, "", TIP_UPDATE, True)
    if state == "changed":
        return ExampleButton("remove", BTN_REMOVE, True, "", TIP_REMOVE, False)
    if not has_file:
        return ExampleButton("none", BTN_ADD, False, "Open a recording first.", TIP_ADD, False)
    if n_peaks <= 0:
        return ExampleButton("none", BTN_ADD, False,
                             "Detect the peaks first (any method), then correct them.",
                             TIP_ADD, False)
    return ExampleButton("add", BTN_ADD, True, "", TIP_ADD, False)


def train_gate(n_examples: int) -> "tuple[bool, str]":
    """Whether Train is allowed, and the line under it (reason or warning)."""
    if n_examples < MIN_EXAMPLES_REQUIRED:
        return False, (f"Training needs at least {MIN_EXAMPLES_REQUIRED} training examples "
                       f"(you have {n_examples}): one is held out to check the model.")
    if n_examples < MIN_EXAMPLES_RECOMMENDED:
        return True, (f"Only {n_examples} training examples: the model may not generalise. "
                      f"At least {MIN_EXAMPLES_RECOMMENDED}, from different animals / "
                      "sessions, are recommended.")
    return True, ""


def train_result_text(result: dict) -> str:
    """train_model()'s result in plain words."""
    if not result.get("ok"):
        return f"✗ {result.get('message', 'Training failed.')}"
    acc = float(result.get("accuracy", 0.0)) * 100
    rec = float(result.get("recall", 0.0)) * 100
    prec = float(result.get("precision", 0.0)) * 100
    text = (f"✓ Model trained on {_plural(int(result.get('n_files', 0)), 'training example')}. "
            f"On recordings kept aside to check it, {acc:.0f} % of candidate peaks were "
            f"classified correctly; it found {rec:.0f} % of the true R-peaks, and "
            f"{prec:.0f} % of the peaks it marked were real.")
    prev = result.get("previous_accuracy")
    if isinstance(prev, (int, float)):
        delta = acc - prev * 100
        text += (f" Previous model: {prev * 100:.0f} % correct "
                 f"({'+' if delta >= 0 else ''}{delta:.0f} points).")
    return text
