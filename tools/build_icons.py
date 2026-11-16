"""Rebuild every app icon file from the two SVG sources.

    python tools/build_icons.py

Needs resvg-py and Pillow (developer-only; not an app dependency):
    pip install resvg-py pillow

Sources (edit these, never the generated files):
    packaging/icon/icon.svg        the full drawing (thin strokes), 64 px and up
    packaging/icon/icon-small.svg  the same cat with thick strokes, 16-48 px
Outputs:
    ecg/ui/assets/icon_{16,32,48,256}.png   window / taskbar icon at run time
    ecg/ui/assets/icon.ico                  Windows title bars at run time
    packaging/icon.ico                       Windows app (PyInstaller)
    packaging/icon.icns                      macOS app (PyInstaller)
    packaging/icon/icon_1024.png             preview / README
"""
from __future__ import annotations

import io
from pathlib import Path

import resvg_py
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "packaging" / "icon"
BIG, SMALL = SRC / "icon.svg", SRC / "icon-small.svg"
SMALL_MAX = 48   # sizes up to this use the thick-stroke drawing


def render(size: int) -> Image.Image:
    svg = SMALL if size <= SMALL_MAX else BIG
    png = resvg_py.svg_to_bytes(svg_path=str(svg), width=size, height=size)
    return Image.open(io.BytesIO(bytes(png))).convert("RGBA")


def main() -> None:
    assets = ROOT / "ecg" / "ui" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for s in (16, 32, 48, 256):
        render(s).save(assets / f"icon_{s}.png", optimize=True)
    render(1024).save(SRC / "icon_1024.png", optimize=True)

    ico_sizes = [16, 24, 32, 48, 64, 128, 256]
    ico = [render(s) for s in ico_sizes]
    for dest in (ROOT / "packaging" / "icon.ico", assets / "icon.ico"):
        ico[-1].save(dest, format="ICO",
                     sizes=[(s, s) for s in ico_sizes], append_images=ico[:-1])

    icns_sizes = [16, 32, 64, 128, 256, 512, 1024]
    icns = [render(s) for s in icns_sizes]
    icns[-1].save(ROOT / "packaging" / "icon.icns", format="ICNS", append_images=icns[:-1])
    print("icons written")


if __name__ == "__main__":
    main()
