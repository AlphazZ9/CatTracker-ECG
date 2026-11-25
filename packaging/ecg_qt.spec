# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the standalone "Tom ECG Analysis Qt" app: the PySide6
interface (ecg_qt_app.py). packaging/ecg.spec builds the Tk one; both are
built while the migration is validated.

Build (from the repository root):
    pyinstaller --noconfirm packaging/ecg_qt.spec
    # optional: --distpath <dir> --workpath <dir> to keep build output elsewhere

Result:
    Windows : dist/Tom ECG Analysis Qt/Tom ECG Analysis Qt.exe   (one-folder build)
    macOS   : dist/Tom ECG Analysis Qt.app                       (app bundle)

Check a build without opening a file (writes "ok ..." to the file):
    "dist/Tom ECG Analysis Qt/Tom ECG Analysis Qt" --self-test result.txt

The Qt interface still uses figure and controller code of ecg.ui (which
imports customtkinter / tkinter) until the Tk interface is removed: both
stay in the bundle meanwhile.

Icon: packaging/icon.ico (Windows) and packaging/icon.icns (macOS), next to
this file, are picked up automatically; ecg/ui/assets (window / taskbar icon
at run time) is bundled as data. All are generated from packaging/icon/*.svg
by tools/build_icons.py.

The app version shown in the macOS bundle comes from the APP_VERSION
environment variable (the CI sets it from the git tag), else setup.py's.
"""
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

HERE = Path(SPECPATH).resolve()          # packaging/
ROOT = HERE.parent                       # repository root
APP_NAME = "Tom ECG Analysis Qt"
BUNDLE_ID = "com.alphazz9.tom-ecg-analysis-qt"
VERSION = os.environ.get("APP_VERSION", "").lstrip("v") or "6.2.0"
IS_MAC = sys.platform == "darwin"


def _have(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


# ── Data files ────────────────────────────────────────────────
datas = []
datas += collect_data_files("customtkinter")      # themes (.json) + fonts
datas += [(str(ROOT / "ecg" / "ui" / "assets"), "ecg/ui/assets")]  # app icon (window/taskbar)
datas += collect_data_files("neurokit2")          # bundled datasets / templates
datas += collect_data_files("matplotlib")         # mpl-data (also done by its hook)
datas += collect_data_files("skops")
datas += collect_data_files("pyqtgraph")          # colour maps, icons
for _opt in ("pzfx",):
    if _have(_opt):
        datas += collect_data_files(_opt)

# ── Hidden imports (imported lazily / by name at runtime) ─────
hiddenimports = []
hiddenimports += collect_submodules("customtkinter")
hiddenimports += collect_submodules("skops")      # skops.io resolves loaders by name
hiddenimports += collect_submodules("ecg.qt")     # tabs / dialogs imported lazily
hiddenimports += collect_submodules("pyqtgraph", filter=lambda m: ".examples" not in m
                                    and ".opengl" not in m and "jupyter" not in m)
hiddenimports += [
    "sklearn.ensemble",
    "sklearn.ensemble._forest",
    "sklearn.tree",
    "sklearn.tree._tree",
    "sklearn.tree._classes",
    "sklearn.model_selection",
    "sklearn.metrics",
    "sklearn.utils._typedefs",
    "sklearn.utils._heap",
    "sklearn.utils._sorting",
    "sklearn.utils._vector_sentinel",
    "sklearn.neighbors._partition_nodes",
    "scipy.signal",
    "scipy.interpolate",
    "scipy.stats",
    "openpyxl",
    "openpyxl.cell._writer",
    "PIL._tkinter_finder",
    "matplotlib.backends.backend_tkagg",
    "matplotlib.backends.backend_qtagg",
    "PySide6.QtSvg",
    "matplotlib.backends.backend_pdf",
    "matplotlib.backends.backend_svg",
    "sqlite3",
]
for _opt in ("h5py", "pzfx"):
    if _have(_opt):
        hiddenimports.append(_opt)
if _have("h5py"):
    hiddenimports += ["h5py.defs", "h5py.utils", "h5py.h5ac", "h5py._proxy"]
if _have("neo"):
    # Spike2 (.smr) loading (ecg.io.loaders); quantities carries unit data.
    hiddenimports += ["neo", "neo.rawio", "neo.rawio.spike2rawio", "quantities"]
    datas += collect_data_files("quantities")

# Not used by the app; leaving them out keeps the bundle smaller.
excludes = [
    "PyQt5", "PyQt6", "PySide2", "wx",
    "IPython", "jupyter", "notebook", "nbformat", "jedi",
    "pytest", "pytestqt", "sphinx", "docutils",
    "matplotlib.backends.backend_wx", "matplotlib.backends.backend_gtk3",
    "pyqtgraph.opengl", "OpenGL",
]

_ico = HERE / "icon.ico"
_icns = HERE / "icon.icns"
icon = None
if IS_MAC and _icns.exists():
    icon = str(_icns)
elif not IS_MAC and _ico.exists():
    icon = str(_ico)


# Linux only: Python builds that keep Tcl / Tk next to the interpreter
# (python-build-standalone, used by uv) -- PyInstaller's tkinter hook does
# not find them there. Windows and macOS builds need nothing here.
binaries = []
if sys.platform.startswith("linux"):
    _lib = Path(sys.base_prefix) / "lib"
    for _so in ("libtcl9.0.so", "libtcl9tk9.0.so", "libtcl8.6.so", "libtk8.6.so"):
        if (_lib / _so).exists():
            binaries.append((str(_lib / _so), "."))

a = Analysis(
    [str(ROOT / "ecg_qt_app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,           # one-folder build
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                       # UPX trips antivirus software; not worth it
    console=False,                   # windowed: no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,                # native arch of the runner (arm64 / x86_64)
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

if IS_MAC:
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=icon,
        bundle_identifier=BUNDLE_ID,
        version=VERSION,
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
            "LSMinimumSystemVersion": "14.0",
        },
    )
