"""Per-period return, risk and drawdown statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import PERIODS, period_slice

TRADING_DAYS = 252


def max_drawdown(log_ret: pd.Series) -> float:
    wealth = np.exp(log_ret.cumsum())
    return float((wealth / wealth.cummax() - 1).min())


def period_stats(log_ret: pd.DataFrame) -> pd.DataFrame:
    """Long table: asset x period with annualised return/vol, cumulative return and max drawdown."""
    rows = []
    for p in PERIODS:
        r = period_slice(log_ret, p)
        for a in r.columns:
            x = r[a]
            rows.append({"asset": a, "period": p, "cum. return": np.expm1(x.sum()),
                         "ann. vol": x.std() * np.sqrt(TRADING_DAYS),
                         "ann. return": x.mean() * TRADING_DAYS, "max drawdown": max_drawdown(x),
                         "skew": x.skew(), "kurtosis": x.kurt()})
    return pd.DataFrame(rows)


def crash_drawdowns(px: pd.DataFrame, peak="2020-02-19", window_end="2020-04-30") -> pd.Series:
    """Peak-to-trough move of each asset inside the crash window, measured from the equity peak."""
    w = px.loc[peak:window_end]
    base = w.iloc[0]
    return (w / base - 1).min().where(lambda s: s < 0, 0.0)
