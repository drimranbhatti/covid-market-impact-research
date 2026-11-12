"""Portfolio stress testing: historical scenarios and Monte Carlo VaR / Expected Shortfall."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def historical_scenario(px: pd.DataFrame, weights: pd.Series, start: str, end: str) -> dict:
    """Replay a historical window: cumulative P&L and path drawdown of a buy-and-hold portfolio."""
    w = px.loc[start:end, weights.index]
    growth = (w / w.iloc[0]) @ weights
    return {"total return": float(growth.iloc[-1] - 1), "max drawdown": float((growth / growth.cummax() - 1).min()),
            "worst day": float(growth.pct_change().min()), "path": growth}


def var_es(pnl: np.ndarray, level=0.99) -> tuple[float, float]:
    """Value-at-Risk and Expected Shortfall as positive loss fractions."""
    q = np.quantile(pnl, 1 - level)
    return float(-q), float(-pnl[pnl <= q].mean())


def monte_carlo(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame, horizon_days=10, n_sims=200_000,
                dof: float | None = None, seed=0) -> np.ndarray:
    """Simulate horizon log-returns of the portfolio under normal or multivariate-t shocks."""
    rng = np.random.default_rng(seed)
    w = weights.loc[cov.index].values
    L = np.linalg.cholesky(cov.values * horizon_days)
    z = rng.standard_normal((n_sims, len(w)))
    if dof:
        chi = rng.chisquare(dof, n_sims) / dof
        z = z / np.sqrt(chi)[:, None] * np.sqrt((dof - 2) / dof)     # unit-variance t
    asset_lr = mu.loc[cov.index].values * horizon_days + z @ L.T
    return np.expm1(asset_lr) @ w


def parametric_var(weights, cov, level=0.99, horizon_days=10) -> float:
    s = float(np.sqrt(weights @ cov @ weights * horizon_days))
    return s * stats.norm.ppf(level)
