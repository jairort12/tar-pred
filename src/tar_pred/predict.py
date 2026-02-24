from __future__ import annotations

import json
import pandas as pd

from tar_pred.schema import validate_and_filter
from tar_pred.raw_cleaning import clean_raw_for_inference


def run_predict(args) -> None:
    # 1) read raw CSV
    df_raw = pd.read_csv(args.input, sep=";")

    # 2) schema validation + exclusions (warning + drop rows)
    df_ok, df_dropped, schema_report = validate_and_filter(df_raw)

    # 3) deterministic cleaning / renormalizations
    df_clean, cleaning_report = clean_raw_for_inference(df_ok)

    # 4) Save a report (for transparency)
    report = {
        "schema": json.loads(schema_report.to_json()),
        "cleaning": json.loads(cleaning_report.to_json()),
        "n_rows_input": int(len(df_raw)),
        "n_rows_kept": int(len(df_clean)),
        "n_rows_dropped": int(len(df_dropped)),
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 5) For now, just export the cleaned rows (next iteration: load model + predict)
    df_clean.to_csv(args.out, index=False, sep=";")

    print(f"Saved cleaned data to: {args.out}")
    print(f"Saved report to: {args.report}")