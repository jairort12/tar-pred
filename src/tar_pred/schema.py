from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

import pandas as pd


# --- Canonical allowed categories (as agreed) ---
ALLOWED_CATEGORIES = {
    "GT": {"BFB", "DG", "FB", "CFB", "DFB", "UG"},
    "GA": {"Air", "Air-Steam", "Steam", "O2-steam", "O2"},
    "MM": {"TP", "SPA", "GC", "GC-MS"},
}

# Categories excluded in training -> must be excluded in production (warning + drop rows)
EXCLUDE = {
    "GT": {"DTBG", "Entrained flow"},
    "GA": {"CO2"},
    "MM": set(),
}

# Columns required for inference (TAR optional)
REQUIRED_COLUMNS = [
    "GT", "GA", "MM",
    "FC", "VM", "Ash",
    "pC", "pH", "pO", "pN", "pS",
    "ER", "Tred",
    "H2", "CH4", "CO", "CO2", "N2",
    "SF", "BF", "SyBR",
]

OPTIONAL_COLUMNS = ["ID", "BM", "TAR"]


@dataclass
class ValidationReport:
    n_rows_in: int
    n_rows_out: int
    n_rows_dropped: int
    dropped_reasons: Dict[str, int]
    warnings: List[str]

    def to_json(self) -> str:
        return json.dumps(
            {
                "n_rows_in": self.n_rows_in,
                "n_rows_out": self.n_rows_out,
                "n_rows_dropped": self.n_rows_dropped,
                "dropped_reasons": self.dropped_reasons,
                "warnings": self.warnings,
            },
            ensure_ascii=False,
            indent=2,
        )


def _clean_cat(s: pd.Series) -> pd.Series:
    # Minimal cleaning: strip, collapse repeated spaces
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def validate_and_filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, ValidationReport]:
    """
    Validate schema and apply training-consistent exclusions.
    Policy: warnings + drop rows (never hard-fail except missing required columns).
    Returns:
      df_ok, df_dropped, report
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df2 = df.copy()

    # Clean categorical columns
    for c in ["GT", "GA", "MM"]:
        df2[c] = _clean_cat(df2[c])

    warnings: List[str] = []
    dropped_reasons: Dict[str, int] = {}

    # Warn on categories outside the allowed set (but do not drop for this reason)
    for c, allowed in ALLOWED_CATEGORIES.items():
        bad = sorted(set(df2[c].dropna().unique()) - allowed - EXCLUDE.get(c, set()))
        if bad:
            warnings.append(
                f"Column {c}: found categories outside template (kept for now, "
                f"but may be ignored by the model encoder): {bad}"
            )

    # Drop rows with excluded categories (training never saw them)
    mask_drop = pd.Series(False, index=df2.index)
    for c, banned in EXCLUDE.items():
        if not banned:
            continue
        m = df2[c].isin(banned)
        n = int(m.sum())
        if n > 0:
            dropped_reasons[f"excluded_{c}"] = n
            warnings.append(f"Dropping {n} rows due to excluded {c} values: {sorted(banned)}")
            mask_drop = mask_drop | m

    df_dropped = df2.loc[mask_drop].copy()
    df_ok = df2.loc[~mask_drop].copy()

    rep = ValidationReport(
        n_rows_in=int(len(df)),
        n_rows_out=int(len(df_ok)),
        n_rows_dropped=int(len(df_dropped)),
        dropped_reasons=dropped_reasons,
        warnings=warnings,
    )
    return df_ok, df_dropped, rep