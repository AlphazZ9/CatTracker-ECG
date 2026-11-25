# -*- coding: utf-8 -*-
"""
ecg.qt.widgets.stale_banner
---------------------------
The out-of-date banner of a view (the Tk ones of Time course, Events and
Morphology): what ecg.app.freshness.stale_banner_text says, with an
optional Recompute button; hidden when there is nothing to say.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6 import QtWidgets


class StaleBanner(QtWidgets.QFrame):
    def __init__(self, parent: "Optional[QtWidgets.QWidget]" = None,
                 on_recompute: "Optional[Callable[[], None]]" = None) -> None:
        super().__init__(parent, objectName="Warning")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(14, 8, 10, 8)
        self.label = QtWidgets.QLabel("")
        self.label.setWordWrap(True)
        self.label.setProperty("warn", True)
        lay.addWidget(self.label, 1)
        self.button: "Optional[QtWidgets.QPushButton]" = None
        if on_recompute is not None:
            self.button = QtWidgets.QPushButton("Recompute")
            self.button.clicked.connect(on_recompute)
            lay.addWidget(self.button)
        self.setVisible(False)

    def set_text(self, text: str) -> None:
        if self.label.text() != text:
            self.label.setText(text)
        self.setVisible(bool(text))

    def text(self) -> str:
        return self.label.text() if self.isVisible() else ""
