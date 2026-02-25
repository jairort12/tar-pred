from __future__ import annotations

from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def _safe_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s.strip("_")


def run_evaluate(args) -> None:
    models_dir = Path(args.models_dir)
    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    # 1) load test data
    X_test = joblib.load(args.x_test)
    y_test = joblib.load(args.y_test)

    # Make sure y_test is 1D array-like
    if isinstance(y_test, (pd.DataFrame, pd.Series)):
        y_true = np.asarray(y_test).reshape(-1)
    else:
        y_true = np.asarray(y_test).reshape(-1)

    # 2) find model files
    model_files = sorted(models_dir.glob("*.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No .pkl models found in: {models_dir.resolve()}")

    rows = []
    failed = []

    # Compatibility: older pickles may reference FeatureBuilder in __main__
    import __main__  # noqa
    from tar_pred.feature_engineering import FeatureBuilder  # noqa
    __main__.FeatureBuilder = FeatureBuilder

    for mp in model_files:
        name = mp.stem
        try:
            model = joblib.load(mp)
            y_pred = model.predict(X_test)
            y_pred = np.asarray(y_pred).reshape(-1)

            r2 = float(r2_score(y_true, y_pred))

            # Save scatter plot y_test vs y_pred
            plt.figure()
            plt.scatter(y_true, y_pred, alpha=0.7)
            plt.xlabel("y_test")
            plt.ylabel("y_pred")
            plt.title(f"{name} | R2={r2:.4f}")
            plot_path = figdir / f"{_safe_name(name)}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            rows.append(
                {
                    "modelo": name,
                    "r2": r2,
                    "archivo": str(mp),
                    "figura": str(plot_path),
                }
            )

        except Exception as e:
            failed.append({"modelo": name, "archivo": str(mp), "error": repr(e)})

    # 3) save metrics
    df = pd.DataFrame(rows).sort_values("r2", ascending=False)
    metrics_path = outdir / "metrics.csv"
    df.to_csv(metrics_path, index=False)

    # 4) save failures (if any)
    if failed:
        failures_path = outdir / "fallos.csv"
        pd.DataFrame(failed).to_csv(failures_path, index=False)
        print(f"[AVISO] Algunos modelos fallaron. Revisa: {failures_path}")

    print(f"Modelos evaluados con éxito: {len(df)} / {len(model_files)}")
    print(f"Métricas guardadas en: {metrics_path}")
    print(f"Figuras guardadas en: {figdir}")