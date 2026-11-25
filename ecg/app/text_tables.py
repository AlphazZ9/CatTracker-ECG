# -*- coding: utf-8 -*-
"""
ecg.app.text_tables
-------------------
Result frames as text: tab-separated for Excel (df_to_tsv) and aligned
for the text boxes (df_to_text). Moved unchanged from the Tk HelpersMixin;
TextTables keeps their names there (_df_to_tsv, _NL_DISPLAY, _df_to_text).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("ecg")


class TextTables:
    @staticmethod
    def _df_to_tsv(df: "pd.DataFrame | None") -> str:
        """Convert a DataFrame to tab-separated values for Excel paste.

        First row = header (column names), subsequent rows = values.
        Pastes into Excel so each metric is in its own column.
        """
        if df is None or df.empty:
            return ""
        rows = ["	".join(str(c).replace("HRV_", "") for c in df.columns)]
        for _, row in df.iterrows():
            cells = []
            for v in row:
                try:
                    fv = float(v)
                    cells.append(f"{fv:.5g}" if np.isfinite(fv) else "")
                except (TypeError, ValueError):
                    cells.append(str(v) if v is not None else "")
            rows.append("	".join(cells))
        return "\n".join(rows)

    _NL_DISPLAY = {
        "SD1":          ("Poincaré SD1 (short-term)", "ms"),
        "SD2":          ("Poincaré SD2 (long-term)",  "ms"),
        "SD1SD2":       ("SD1 / SD2", ""),
        "CSI":          ("Cardiac Sympathetic Index", ""),
        "CVI":          ("Cardiac Vagal Index", ""),
        "CSI_Modified": ("Modified CSI", ""),
        "SampEn":       ("Sample entropy", ""),
        "ApEn":         ("Approximate entropy", ""),
        "DFA_alpha1":   ("DFA α1 (short-term)", ""),
        "DFA_alpha2":   ("DFA α2 (long-term)", ""),
    }

    @staticmethod
    def _df_to_text(df: "pd.DataFrame | None",
                     display_map: "Optional[dict[str, tuple[str, str]]]" = None) -> str:
        """Render every finite numeric column of a NeuroKit2 HRV DataFrame as readable text.

        Format uses dot-leaders for visual alignment without requiring a fixed-width font:
            SDNN ............  12.35
            RMSSD ...........   8.27

        *display_map* optionally maps a column's HRV_-stripped name to a
        (label, unit) pair; columns not in the map fall back to the raw
        name with underscores turned to spaces (still better than the
        literal NeuroKit identifier).
        """
        if df is None or df.empty:
            return "  (not computed)"
        display_map = display_map or {}
        rows = []
        for col in df.columns:
            try:
                v = float(df[col].values[0])
                if np.isfinite(v):
                    name = col.replace("HRV_", "")
                    label, unit = display_map.get(name, (name.replace("_", " "), ""))
                    disp = f"{label} ({unit})" if unit else label
                    # Adaptive precision: integers → 0 dp, small values → 4 dp
                    if abs(v) >= 100:
                        fmt = f"{v:.1f}"
                    elif abs(v) >= 1:
                        fmt = f"{v:.3f}"
                    else:
                        fmt = f"{v:.5f}"
                    # Dot-leader padding to column 30
                    dots = "·" * max(2, 30 - len(disp))
                    rows.append(f"  {disp} {dots}  {fmt:>10}")
            except Exception as exc:
                log.debug("_df_to_text skip '%s': %s", col, exc)
        return "\n".join(rows) or "  (no finite values)"


df_to_tsv = TextTables._df_to_tsv
df_to_text = TextTables._df_to_text
