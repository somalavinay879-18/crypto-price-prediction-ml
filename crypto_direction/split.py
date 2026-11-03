"""Time-based train/test split: the model never sees the period it is tested on."""
import pandas as pd


def time_split(data: pd.DataFrame, test_days: int = 365) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last `test_days` days (across all symbols) as the test set."""
    test_start = data["datetime"].max() - pd.Timedelta(days=test_days)
    train = data[data["datetime"] < test_start]
    test = data[data["datetime"] >= test_start]
    if train.empty or test.empty:
        raise ValueError("Split produced an empty train or test set")
    return train.copy(), test.copy()


def to_matrix(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    """Features plus one-hot symbol columns, with test columns aligned to train."""
    x_train = pd.get_dummies(train[features + ["symbol"]], columns=["symbol"], drop_first=True, dtype="float32")
    x_test = pd.get_dummies(test[features + ["symbol"]], columns=["symbol"], drop_first=True, dtype="float32")
    x_train, x_test = x_train.align(x_test, join="left", axis=1, fill_value=0)
    return x_train, train["y"].astype(int), x_test, test["y"].astype(int)
