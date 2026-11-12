"""Diebold & Yilmaz (2012) spillover index from a VAR with generalised FEVD (Pesaran & Shin, 1998)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def generalized_fevd(ma_coefs: np.ndarray, sigma: np.ndarray, horizon: int) -> np.ndarray:
    """Row-normalised generalised forecast-error variance decomposition.

    ma_coefs: array (H, N, N) of MA coefficient matrices A_0..A_{H-1} (A_0 = I).
    theta_ij = sigma_jj^{-1} sum_h (e_i' A_h Sigma e_j)^2 / sum_h (e_i' A_h Sigma A_h' e_i)
    """
    A = ma_coefs[:horizon]
    num = np.zeros_like(sigma)
    den = np.zeros(sigma.shape[0])
    for Ah in A:
        AS = Ah @ sigma
        num += AS**2
        den += np.einsum("ij,ij->i", AS, Ah)
    theta = (num / np.diag(sigma)[None, :]) / den[:, None]
    return theta / theta.sum(axis=1, keepdims=True)


def spillover_table(returns: pd.DataFrame, lags=2, horizon=10) -> tuple[pd.DataFrame, float]:
    """Full Diebold-Yilmaz table (% of forecast-error variance of row i due to shocks in column j).

    Returns (table with 'FROM others' column and 'TO others'/'NET' rows, total spillover index %).
    """
    res = VAR(returns.values).fit(lags)
    theta = generalized_fevd(res.ma_rep(horizon - 1), res.sigma_u, horizon) * 100
    names = list(returns.columns)
    tbl = pd.DataFrame(theta, index=names, columns=names)
    off = tbl.values - np.diag(np.diag(tbl.values))
    tbl["FROM others"] = off.sum(axis=1)
    to = pd.Series(off.sum(axis=0), index=names)
    total = float(off.sum() / len(names))
    tbl.loc["TO others"] = list(to.values) + [total]
    tbl.loc["NET"] = list((to - tbl["FROM others"].iloc[:len(names)]).values) + [np.nan]
    return tbl, total


def rolling_total_spillover(returns: pd.DataFrame, window=200, step=5, lags=2, horizon=10) -> pd.Series:
    out = {}
    for end in range(window, len(returns) + 1, step):
        _, total = spillover_table(returns.iloc[end - window:end], lags, horizon)
        out[returns.index[end - 1]] = total
    return pd.Series(out, name="total spillover (%)")


def rolling_net_spillover(returns: pd.DataFrame, window=200, step=5, lags=2, horizon=10) -> pd.DataFrame:
    rows = {}
    for end in range(window, len(returns) + 1, step):
        tbl, _ = spillover_table(returns.iloc[end - window:end], lags, horizon)
        rows[returns.index[end - 1]] = tbl.loc["NET", returns.columns]
    return pd.DataFrame(rows).T
