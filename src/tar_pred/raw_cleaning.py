from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import json

import numpy as np
import pandas as pd


GAS_COLS = ["H2", "CO", "CH4", "CO2", "N2"]
GAS_NO_N2 = ["H2", "CO", "CH4", "CO2"]
PROX_COLS = ["FC", "VM", "Ash"]
CAT_COLS = ["GT", "GA", "MM", "BM"]


@dataclass
class CleaningReport:
    warnings: List[str]

    def to_json(self) -> str:
        return json.dumps({"warnings": self.warnings}, ensure_ascii=False, indent=2)


def _to_numeric(df: pd.DataFrame, cols: List[str], warnings: List[str]) -> pd.DataFrame:
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            before_na = int(df2[c].isna().sum())
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
            after_na = int(df2[c].isna().sum())
            new_na = after_na - before_na
            if new_na > 0:
                warnings.append(f"Column {c}: coerced {new_na} values to NaN (non-numeric inputs).")
    return df2


def renormalize_gas_free_of_n2(df: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
    """
    Re-normalize H2/CO/CH4/CO2 to sum to 100 (excluding N2), as done in training.
    Leaves N2 column unchanged (can be used for checks), but models typically use the normalized 4.
    """
    df2 = df.copy()
    # Ensure numeric
    df2 = _to_numeric(df2, GAS_COLS, warnings)

    denom = df2[GAS_NO_N2].sum(axis=1)

    bad = denom <= 0
    n_bad = int(bad.sum())
    if n_bad > 0:
        warnings.append(
            f"Gas renorm: {n_bad} rows have (H2+CO+CH4+CO2) <= 0; cannot renormalize. Values kept as-is for those rows."
        )

    # Renormalize where possible
    ok = ~bad & denom.notna()
    for c in GAS_NO_N2:
        df2.loc[ok, c] = 100.0 * df2.loc[ok, c] / denom.loc[ok]

    # Optional check: gas totals (with N2) not necessarily 100; that's fine
    return df2


def renormalize_proximate_to_100(df: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
    """
    Re-normalize FC/VM/Ash to sum to 100 (dry basis), as done in training.
    """
    df2 = df.copy()
    df2 = _to_numeric(df2, PROX_COLS, warnings)

    denom = df2[PROX_COLS].sum(axis=1)
    bad = denom <= 0
    n_bad = int(bad.sum())
    if n_bad > 0:
        warnings.append(
            f"Proximate renorm: {n_bad} rows have (FC+VM+Ash) <= 0; cannot renormalize. Values kept as-is for those rows."
        )

    ok = ~bad & denom.notna()
    for c in PROX_COLS:
        df2.loc[ok, c] = 100.0 * df2.loc[ok, c] / denom.loc[ok]

    return df2


def coerce_categories(df: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
    """
    Minimal categorical coercion: strip spaces and keep as string.
    (Actual allowed/excluded validation is handled in schema.py.)
    """
    df2 = df.copy()
    for c in CAT_COLS:
        if c in df2.columns:
            df2[c] = (
                df2[c].astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
    return df2


def clean_raw_for_inference(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningReport]:
    warnings: List[str] = []
    df = df_raw.copy()
    df = coerce_categories(df, warnings)
    df = renormalize_gas_free_of_n2(df, warnings)
    df = renormalize_proximate_to_100(df, warnings)
    return df, CleaningReport(warnings=warnings)