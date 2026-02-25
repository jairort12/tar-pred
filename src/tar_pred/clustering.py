from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import json
from pathlib import Path

import pandas as pd
import joblib


ELEMENTAL_COLS = ["pC", "pH", "pO", "pN", "pS"]


@dataclass
class ClusteringReport:
    warnings: List[str]

    def to_json(self) -> str:
        return json.dumps({"warnings": self.warnings}, ensure_ascii=False, indent=2)


def add_bm_cluster_final(df: pd.DataFrame, artifacts_dir: str) -> Tuple[pd.DataFrame, ClusteringReport]:
    """
    Reproduce training clustering logic:
      - scaler_km + kmeans_family => Family_{0..3}
      - If Family_0 => scaler_sub + kmeans_sub => F0_Sub_{0..8}
      - Else keep Family_k
      - Output: BM_Cluster_Final as category
    Missing elementals are median-imputed (same idea as training).
    """
    warnings: List[str] = []
    df2 = df.copy()

    art = Path(artifacts_dir)
    scaler_km = joblib.load(art / "scaler_km.pkl")
    kmeans_family = joblib.load(art / "kmeans_family.pkl")
    scaler_sub = joblib.load(art / "scaler_sub.pkl")
    kmeans_sub = joblib.load(art / "kmeans_sub.pkl")

    # Ensure required cols exist
    missing = [c for c in ELEMENTAL_COLS if c not in df2.columns]
    if missing:
        raise ValueError(f"Missing elemental columns required for clustering: {missing}")

    X_elem = df2[ELEMENTAL_COLS].copy()
    # Median imputation per-column (training did median fill)
    for c in ELEMENTAL_COLS:
        med = X_elem[c].median(skipna=True)
        if pd.isna(med):
            warnings.append(f"Elemental {c}: median is NaN (all values missing). Filling with 0.")
            med = 0.0
        X_elem[c] = X_elem[c].fillna(med)

    # Family clustering
    X_scaled = scaler_km.transform(X_elem)
    fam_labels = kmeans_family.predict(X_scaled)  # 0..3
    fam_str = pd.Series([f"Family_{i}" for i in fam_labels], index=df2.index)

    # Final label default = family
    final = fam_str.astype(object)

    # Subclustering for Family_0 only
    mask_f0 = fam_labels == 0
    n_f0 = int(mask_f0.sum())
    if n_f0 > 0:
        X_f0 = X_elem.loc[mask_f0]
        X_f0_scaled = scaler_sub.transform(X_f0)
        sub_labels = kmeans_sub.predict(X_f0_scaled)  # 0..8
        final.loc[mask_f0] = [f"F0_Sub_{i}" for i in sub_labels]
    else:
        warnings.append("No rows predicted as Family_0; subclustering not applied.")

    df2["BM_Cluster_Final"] = final.astype("category")
    return df2, ClusteringReport(warnings=warnings)