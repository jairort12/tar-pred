import argparse

def main():
    parser = argparse.ArgumentParser(prog="tar-predict")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("evaluate", help="Evalua modelos con X_test/y_test (pendiente)")
    p_eval.add_argument("--models-dir", default="models")
    p_eval.add_argument("--x-test", default="data/processed/X_test.pkl")
    p_eval.add_argument("--y-test", default="data/processed/y_test.pkl")
    p_eval.add_argument("--outdir", default="reports")

    p_pred = sub.add_parser("predict", help="Predice desde CSV crudo (pendiente)")
    p_pred.add_argument("--input", required=True)
    p_pred.add_argument("--model-path", required=True)
    p_pred.add_argument("--artifacts-dir", default="artifacts")
    p_pred.add_argument("--out", default="predictions.csv")
    p_pred.add_argument("--report", default="prediction_report.json")

    args = parser.parse_args()
    print(f"Command selected: {args.cmd} (CLI wired; implementation in modules)")

if __name__ == "__main__":
    main()