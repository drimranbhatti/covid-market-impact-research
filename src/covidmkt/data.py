"""Dataset loading, asset metadata and COVID-19 period definitions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = pd.DataFrame([
    ("SPY", "US large caps", "Equities"), ("IWM", "US small caps", "Equities"), ("EEM", "EM equities", "Equities"),
    ("TLT", "Long Treasuries", "Fixed income"), ("LQD", "IG credit", "Fixed income"), ("HYG", "High yield", "Fixed income"),
    ("GC=F", "Gold", "Commodities"), ("USO", "Crude oil", "Commodities"), ("HG=F", "Copper", "Commodities"),
    ("DX-Y.NYB", "US dollar", "FX"), ("EURUSD=X", "EUR/USD", "FX"), ("JPY=X", "USD/JPY", "FX"),
    ("BTC-USD", "Bitcoin", "Crypto"), ("ETH-USD", "Ether", "Crypto"), ("^VIX", "VIX", "Volatility"),
], columns=["ticker", "name", "group"]).set_index("ticker")

# Period boundaries (inclusive). The crash window spans the S&P 500 peak (19 Feb 2020)
# to the end of April, after the Fed's emergency programmes stabilised markets.
PERIODS = {
    "Pre-COVID": ("2018-01-01", "2020-02-19"),
    "Crash": ("2020-02-20", "2020-04-30"),
    "Recovery": ("2020-05-01", "2021-12-31"),
    "Post-COVID": ("2022-01-01", "2024-12-31"),
}
PERIOD_COLORS = {"Pre-COVID": "#4C72B0", "Crash": "#C44E52", "Recovery": "#55A868", "Post-COVID": "#8172B2"}


def load_prices(path: str | Path) -> pd.DataFrame:
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return px.ffill().dropna()


def log_returns(px: pd.DataFrame, exclude_vix: bool = True) -> pd.DataFrame:
    cols = [c for c in px.columns if not (exclude_vix and c == "^VIX")]
    return np.log(px[cols]).diff().dropna()


def period_slice(df: pd.DataFrame, period: str) -> pd.DataFrame:
    start, end = PERIODS[period]
    return df.loc[start:end]


def label(tickers) -> list[str]:
    return [ASSETS.loc[t, "name"] if t in ASSETS.index else t for t in tickers]
