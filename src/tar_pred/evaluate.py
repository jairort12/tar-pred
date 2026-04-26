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

    y_true = np.asarray(y_test).reshape(-1)

    # 2) find model files
    model_files = sorted(models_dir.glob("*.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No .pkl models found in: {models_dir.resolve()}")

    rows = []
    failed = []

    # Compatibility: older pickles may reference FeatureBuilder in __main__
    import __main__  # noqa: E402
    from tar_pred.feature_engineering import FeatureBuilder  # noqa: E402
    __main__.FeatureBuilder = FeatureBuilder

    for i, mp in enumerate(model_files, start=1):
        name = mp.stem
        print(f"[{i}/{len(model_files)}] Evaluting and plotting: {name}")

        try:
            model = joblib.load(mp)
            y_pred = model.predict(X_test)
            y_pred = np.asarray(y_pred).reshape(-1)

            r2 = float(r2_score(y_true, y_pred))

            # Save scatter plot y_test vs y_pred + diagonal y=x
            plt.figure()
            plt.scatter(y_true, y_pred, alpha=0.7)

            vmin = float(min(y_true.min(), y_pred.min()))
            vmax = float(max(y_true.max(), y_pred.max()))
            plt.plot([vmin, vmax], [vmin, vmax], linestyle="--")
            plt.xlim(vmin, vmax)
            plt.ylim(vmin, vmax)

            plt.xlabel("y_test")
            plt.ylabel("y_pred")
            plt.title(f"{name} | R2={r2:.4f}")

            plot_path = figdir / f"{_safe_name(name)}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"   -> R2={r2:.4f}")

            rows.append(
                {
                    "model": name,
                    "r2": r2,
                    "file": str(mp),
                    "figure": str(plot_path),
                }
            )

        except Exception as e:
            failed.append({"model": name, "file": str(mp), "error": repr(e)})

    # 3) save metrics
    df = pd.DataFrame(rows).sort_values("r2", ascending=False)
    metrics_path = outdir / "metrics.csv"
    df.to_csv(metrics_path, index=False)

    # 4) save failures (if any)
    if failed:
        failures_path = outdir / "errors.csv"
        pd.DataFrame(failed).to_csv(failures_path, index=False)
        print(f"[Waring] Some models fail to predict. Check file: {failures_path}")

    print(f"Models succesfuly evaluated: {len(df)} / {len(model_files)}")
    print(f"Metrics stored in: {metrics_path}")
    print(f"Figures stored in: {figdir}")