# -*- coding: utf-8 -*-
"""ecg.app.theme re-syncs colour and font globals through a hard-coded module
list. A module importing those names at module scope and missing from the list
freezes on the old theme after a toggle, with no error anywhere."""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import ecg
from ecg.app import theme

APP_DIR = Path(ecg.__file__).parent / "app"
SYNCED = set(theme._SYNC_COLOUR_NAMES) | set(theme._SYNC_FONT_NAMES)


def _app_modules():
    return sorted(p for p in APP_DIR.glob("*.py") if p.stem not in ("__init__", "theme"))


def _module_scope_theme_imports(path: Path) -> set[str]:
    """Names taken from ecg.app.theme at import time (not inside a function)."""
    names: set[str] = set()

    def visit(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.ImportFrom) and node.module == "ecg.app.theme":
                names.update(a.name for a in node.names)
            for field in ("body", "orelse", "finalbody"):
                visit(getattr(node, field, []) or [])
            for handler in getattr(node, "handlers", []) or []:
                visit(handler.body)

    visit(ast.parse(path.read_text(encoding="utf-8")).body)
    return names


@pytest.mark.parametrize("path", _app_modules(), ids=lambda p: p.stem)
def test_module_importing_colours_is_in_the_sync_list(path):
    imported = _module_scope_theme_imports(path)
    assert "*" not in imported, f"{path.name}: star import from theme"
    if imported & SYNCED:
        assert f"ecg.app.{path.stem}" in theme._SYNC_MODULES, (
            f"{path.name} imports {sorted(imported & SYNCED)[:4]}... from "
            "ecg.app.theme but is missing from theme._SYNC_MODULES")


def test_sync_list_has_no_stale_entries():
    root = Path(ecg.__file__).parent.parent
    for name in theme._SYNC_MODULES:
        assert (root / (name.replace(".", "/") + ".py")).exists(), name


def test_sync_reaches_every_controller_and_figure_module():
    """The real push, on the modules of ecg.app that copy theme names."""
    holders = []
    for path in _app_modules():
        if not (_module_scope_theme_imports(path) & SYNCED):
            continue
        mod = importlib.import_module(f"ecg.app.{path.stem}")
        holders += [(mod, n) for n in SYNCED if hasattr(mod, n)]
    assert holders

    saved = {n: getattr(theme, n) for n in SYNCED if hasattr(theme, n)}
    try:
        for n in saved:
            setattr(theme, n, f"sentinel-{n}")
        theme._sync_to_submodules()
        stale = [f"{m.__name__}.{n}" for m, n in holders if getattr(m, n) != f"sentinel-{n}"]
        assert not stale, stale
    finally:
        for n, value in saved.items():
            setattr(theme, n, value)
        theme._sync_to_submodules()
