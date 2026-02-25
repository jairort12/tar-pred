from __future__ import annotations

import json
import pandas as pd
import joblib

from tar_pred.schema import validate_and_filter
from tar_pred.raw_cleaning import clean_raw_for_inference
from tar_pred.clustering import add_bm_cluster_final


DROP_COLS_AFTER_CLUSTER = ["Ash", "VM", "pO", "BF", "BM"]  # training-style removal


def run_predict(args) -> None:
    # 0) read raw CSV (;)
    df_raw = pd.read_csv(args.input, sep=";")

    # 1) schema validation + exclusions (warning + drop rows for excluded categories)
    df_ok, df_dropped, schema_report = validate_and_filter(df_raw)

    # 2) drop ID (keep a copy for output if present)
    id_series = df_ok["ID"].copy() if "ID" in df_ok.columns else None
    if "ID" in df_ok.columns:
        df_ok = df_ok.drop(columns=["ID"])

    # 3) normalizations (gas free N2 and proximate to 100)
    df_clean, cleaning_report = clean_raw_for_inference(df_ok)

    # 4) drop N2 column (as in training)
    if "N2" in df_clean.columns:
        df_clean = df_clean.drop(columns=["N2"])

    # 5) cluster + subcluster => BM_Cluster_Final
    df_clustered, clustering_report = add_bm_cluster_final(df_clean, artifacts_dir=args.artifacts_dir)

    # 6) drop variables list (training)
    cols_to_drop = [c for c in DROP_COLS_AFTER_CLUSTER if c in df_clustered.columns]
    if cols_to_drop:
        df_clustered = df_clustered.drop(columns=cols_to_drop)

    # 7) Load model and predict
    # Compatibility: older pickles reference FeatureBuilder in __main__
    import __main__  # noqa: E402
    from tar_pred.feature_engineering import FeatureBuilder  # noqa: E402
    __main__.FeatureBuilder = FeatureBuilder

    model = joblib.load(args.model_path)
    y_pred = model.predict(df_clustered)

    # 8) Build output
    out_df = pd.DataFrame({"y_pred": y_pred})
    if id_series is not None:
        out_df.insert(0, "ID", id_series.reset_index(drop=True))

    # Save predictions (CSV ;)
    out_df.to_csv(args.out, index=False, sep=";")

    # Save report (schema + cleaning + clustering + dropped rows)
    report = {
        "schema": json.loads(schema_report.to_json()),
        "cleaning": json.loads(cleaning_report.to_json()),
        "clustering": json.loads(clustering_report.to_json()),
        "n_rows_input": int(len(df_raw)),
        "n_rows_after_exclusions": int(len(df_ok)),
        "n_rows_dropped_by_exclusions": int(len(df_dropped)),
        "dropped_rows_preview": df_dropped.head(5).to_dict(orient="records"),
        "columns_dropped_post_cluster": cols_to_drop,
        "model_path": args.model_path,
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved predictions to: {args.out}")
    print(f"Saved report to: {args.report}")