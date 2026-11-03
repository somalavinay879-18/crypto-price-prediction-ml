"""Tests on small synthetic candles, so they run in CI without the real data."""
import numpy as np
import pandas as pd
import pytest

from crypto_direction.data import load_candles, symbol_from_path
from crypto_direction.features import LAGS, WINDOWS, add_features, add_target, feature_columns
from crypto_direction.models import build_models, evaluate
from crypto_direction.split import time_split, to_matrix


def make_candles(symbols=("AAAUSDT", "BBBUSDT"), n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for i, sym in enumerate(symbols):
        close = 100 * (i + 1) * np.cumprod(1 + rng.normal(0, 0.01, n))
        frames.append(pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
            "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
            "volume": rng.uniform(1, 10, n), "turnover": rng.uniform(1, 10, n),
            "symbol": sym,
        }))
    return pd.concat(frames, ignore_index=True)


def test_symbol_from_path():
    from pathlib import Path
    assert symbol_from_path(Path("data/BYBIT_BTCUSDT_15m.csv")) == "BTCUSDT"


def test_load_candles_sorts_and_removes_duplicates(tmp_path):
    rows = "Datetime,Open,High,Low,Close,Volume,Turnover\n"
    rows += "2024-01-01 00:15:00,2,2,2,2,1,1\n2024-01-01 00:00:00,1,1,1,1,1,1\n2024-01-01 00:00:00,1,1,1,1,1,1\n"
    (tmp_path / "BYBIT_XUSDT_15m.csv").write_text(rows)
    data = load_candles(tmp_path)
    assert list(data["close"]) == [1, 2]
    assert data["symbol"].unique().tolist() == ["XUSDT"]
    assert str(data["datetime"].dt.tz) == "UTC"


def test_load_candles_without_files_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_candles(tmp_path)


def test_target_compares_with_next_candle_of_same_symbol():
    data = add_target(make_candles(n=10))
    for _, group in data.groupby("symbol"):
        expected = (group["close_next"] > group["close"]).astype(int)
        assert (group["y"] == expected).all()
    # last candle of each symbol has no next candle
    assert len(data) == 2 * 9


def test_features_do_not_look_into_the_future():
    """Changing future prices must not change features computed at an earlier time."""
    candles = make_candles(n=300)
    original = add_features(add_target(candles))

    changed = candles.copy()
    cutoff = candles["datetime"].iloc[200]
    future = changed["datetime"] > cutoff
    changed.loc[future, ["close", "volume"]] *= 3
    altered = add_features(add_target(changed))

    cols = [c for c in feature_columns(original) if c != "close_next"]
    keep = original["datetime"] < cutoff
    a = original.loc[keep].set_index(["symbol", "datetime"])[cols]
    b = altered.set_index(["symbol", "datetime"]).loc[a.index, cols]
    pd.testing.assert_frame_equal(a, b)


def test_lags_stay_within_one_symbol():
    data = add_features(add_target(make_candles(n=200)))
    for _, group in data.groupby("symbol"):
        group = group.reset_index(drop=True)
        assert np.allclose(group["ret_1_lag1"].iloc[1:].values, group["ret_1"].iloc[:-1].values)


def test_feature_rows_start_after_the_longest_window():
    data = add_features(add_target(make_candles(n=200)))
    per_symbol = data.groupby("symbol").size()
    # 1 row lost to the target, 1 to pct_change, then the longest of lags and windows
    assert (per_symbol == 200 - 1 - 1 - max(LAGS, max(WINDOWS)) + 1).all()


def test_feature_columns_exclude_target_and_future():
    data = add_features(add_target(make_candles(n=200)))
    cols = feature_columns(data)
    assert "y" not in cols and "close_next" not in cols and "datetime" not in cols


def test_time_split_has_no_overlap():
    data = add_features(add_target(make_candles(n=400)))
    train, test = time_split(data, test_days=1)
    assert train["datetime"].max() < test["datetime"].min()
    assert len(train) + len(test) == len(data)


def test_time_split_rejects_empty_sets():
    data = add_features(add_target(make_candles(n=200)))
    with pytest.raises(ValueError):
        time_split(data, test_days=365)


def test_matrix_columns_match_between_train_and_test():
    data = add_features(add_target(make_candles(n=400)))
    train, test = time_split(data, test_days=1)
    x_train, y_train, x_test, y_test = to_matrix(train, test, feature_columns(data))
    assert list(x_train.columns) == list(x_test.columns)
    assert "symbol_BBBUSDT" in x_train.columns
    assert len(x_train) == len(y_train) and len(x_test) == len(y_test)


def test_majority_class_baseline_scores_half_balanced_accuracy():
    data = add_features(add_target(make_candles(n=400)))
    train, test = time_split(data, test_days=1)
    x_train, y_train, x_test, y_test = to_matrix(train, test, feature_columns(data))
    model = build_models()["Majority class"].fit(x_train, y_train)
    scores = evaluate(model, x_test, y_test)
    assert scores["balanced_accuracy"] == pytest.approx(0.5)
