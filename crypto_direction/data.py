"""Load the Bybit 15-minute candle files into one table."""
from pathlib import Path

import pandas as pd

DTYPES = {
    "Open": "float32",
    "High": "float32",
    "Low": "float32",
    "Close": "float32",
    "Volume": "float32",
    "Turnover": "float32",
}


def symbol_from_path(path: Path) -> str:
    """BYBIT_BTCUSDT_15m.csv -> BTCUSDT"""
    return path.stem.replace("BYBIT_", "").replace("_15m", "")


def load_candles(data_dir: Path | str) -> pd.DataFrame:
    """Read every BYBIT_*_15m.csv in data_dir, add a symbol column and sort by symbol and time."""
    files = sorted(Path(data_dir).glob("BYBIT_*_15m.csv"))
    if not files:
        raise FileNotFoundError(f"No BYBIT_*_15m.csv files found in {data_dir}")

    frames = []
    for path in files:
        df = pd.read_csv(path, dtype=DTYPES)
        df["symbol"] = symbol_from_path(path)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data.columns = data.columns.str.lower()
    data["datetime"] = pd.to_datetime(data["datetime"], utc=True)
    data = data.drop_duplicates(subset=["symbol", "datetime"])
    return data.sort_values(["symbol", "datetime"]).reset_index(drop=True)
