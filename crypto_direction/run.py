"""Run the full experiment and write the results table.

    python -m crypto_direction.run            # CSV files in the repository root
    python -m crypto_direction.run --fast     # smaller Random Forest and XGBoost, for a quick check
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from .data import load_candles
from .features import add_features, add_target, feature_columns
from .models import build_models, evaluate
from .split import time_split, to_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".", help="folder with the BYBIT_*_15m.csv files")
    parser.add_argument("--out", default="reports/results.csv")
    parser.add_argument("--test-days", type=int, default=365)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    data = add_features(add_target(load_candles(args.data_dir)))
    train, test = time_split(data, args.test_days)
    x_train, y_train, x_test, y_test = to_matrix(train, test, feature_columns(data))

    print(f"Rows: {len(data):,}  train: {len(train):,}  test: {len(test):,}")
    print(f"Train ends {train['datetime'].max()}, test starts {test['datetime'].min()}")
    print(f"Share of 'up' candles in the test set: {y_test.mean():.3f}")

    models = build_models()
    if args.fast:
        models["Random Forest"].set_params(n_estimators=50)
        models["XGBoost"].set_params(n_estimators=100)

    rows = []
    for name, model in models.items():
        start = time.time()
        model.fit(x_train, y_train)
        row = {"model": name, **evaluate(model, x_test, y_test), "fit_seconds": round(time.time() - start, 1)}
        rows.append(row)
        print(f"{name:<20} accuracy {row['accuracy']:.4f}  ROC-AUC {row['roc_auc']:.4f}")

    results = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
