"""ecg.core, ecg.services, ecg.figures and ecg.app are free of any user-interface
toolkit (no Qt, no Tk), including imports that only exist for type checking;
nothing imports the removed CustomTkinter interface (ecg.ui)."""
import ast
from pathlib import Path

import ecg
import ecg.app
import ecg.core
import ecg.core.models
import ecg.figures
import ecg.services
from ecg.core.models import FilterParams, FilterWidgetHost


def _ui_imports(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        found += [f"{path.name}:{node.lineno} {n}" for n in names
                  if n == "ecg.ui" or n.startswith("ecg.ui.")]
    return found


TOOLKITS = ("tkinter", "_tkinter", "customtkinter", "PySide6", "PyQt5", "PyQt6", "pyqtgraph",
            "ecg.qt")


def _toolkit_imports(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        found += [f"{path.name}:{node.lineno} {n}" for n in names
                  if any(n == t or n.startswith(t + ".") for t in TOOLKITS)
                  or "backend_tk" in n]
    return found


def test_no_module_in_core_imports_ecg_ui():
    core = Path(ecg.core.__file__).parent
    assert [hit for p in sorted(core.rglob("*.py")) for hit in _ui_imports(p)] == []


def test_no_module_in_services_imports_ecg_ui():
    services = Path(ecg.services.__file__).parent
    assert [hit for p in sorted(services.rglob("*.py")) for hit in _ui_imports(p)] == []


def test_the_application_layer_uses_no_toolkit():
    for pkg in (ecg.core, ecg.services, ecg.figures, ecg.app):
        root = Path(pkg.__file__).parent
        assert [hit for p in sorted(root.rglob("*.py")) for hit in _toolkit_imports(p)] == [], \
            pkg.__name__


def test_nothing_imports_the_removed_tk_interface():
    root = Path(ecg.__file__).parent
    assert not (root / "ui").exists()
    assert [hit for p in sorted(root.rglob("*.py")) for hit in _ui_imports(p)] == []


def test_filter_widget_host_declares_the_attributes_from_widgets_requires():
    # Attributes read through getattr() are optional and stay undeclared.
    src = Path(ecg.core.models.__file__).read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "from_widgets")
    used = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == "app"}
    declared = set(FilterWidgetHost.__annotations__) | {"_safe_float"}
    assert used == declared
    assert FilterParams.from_widgets.__func__.__annotations__["app"] == "FilterWidgetHost"
