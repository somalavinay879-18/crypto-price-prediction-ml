"""Target and features. Every feature uses only the current and earlier candles of the same symbol."""
import numpy as np
import pandas as pd

LAGS = 16  # 16 x 15 minutes = the last 4 hours
WINDOWS = (16, 64)  # 4 hours and 16 hours


def add_target(data: pd.DataFrame) -> pd.DataFrame:
    """y = 1 if the next candle closes higher than the current one, else 0.

    The last candle of each symbol has no next candle and is dropped.
    """
    data = data.copy()
    data["close_next"] = data.groupby("symbol")["close"].shift(-1)
    data = data.dropna(subset=["close_next"])
    data["y"] = (data["close_next"] > data["close"]).astype("int8")
    return data.reset_index(drop=True)


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Returns, volume changes, their lags and rolling mean and volatility of returns."""
    data = data.copy()
    g = data.groupby("symbol", group_keys=False)

    data["ret_1"] = g["close"].pct_change(1)
    data["vol_chg_1"] = g["volume"].pct_change(1)

    g = data.groupby("symbol", group_keys=False)
    new = {}
    for k in range(1, LAGS + 1):
        new[f"ret_1_lag{k}"] = g["ret_1"].shift(k)
        new[f"vol_lag{k}"] = g["vol_chg_1"].shift(k)
    for w in WINDOWS:
        new[f"ret_mean_{w}"] = g["ret_1"].transform(lambda s, w=w: s.rolling(w).mean())
        new[f"ret_std_{w}"] = g["ret_1"].transform(lambda s, w=w: s.rolling(w).std())
    data = pd.concat([data, pd.DataFrame(new, index=data.index)], axis=1)

    data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return data


def feature_columns(data: pd.DataFrame) -> list[str]:
    """Model inputs: everything except identifiers, the target and the future close."""
    excluded = {"y", "close_next", "datetime", "symbol"}
    return [c for c in data.columns if c not in excluded]
