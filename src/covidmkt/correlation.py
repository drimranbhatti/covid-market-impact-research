"""Correlation regimes and the Forbes-Rigobon (2002) contagion test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def average_pairwise(corr: pd.DataFrame) -> float:
    c = corr.values
    return float(c[np.triu_indices_from(c, k=1)].mean())


def rolling_corr(x: pd.Series, y: pd.Series, window=63) -> pd.Series:
    return x.rolling(window).corr(y)


def forbes_rigobon(rho_crisis: float, var_crisis: float, var_calm: float) -> float:
    """Heteroskedasticity-adjusted crisis correlation.

    Correlations mechanically rise when the source market's volatility rises. Forbes & Rigobon
    adjust for this:  rho* = rho / sqrt(1 + delta (1 - rho^2)),  delta = var_crisis / var_calm - 1.
    """
    delta = var_crisis / var_calm - 1
    return rho_crisis / np.sqrt(1 + delta * (1 - rho_crisis**2))


def fisher_z_test(rho1: float, n1: int, rho2: float, n2: int) -> tuple[float, float]:
    """Test H0: rho2 = rho1 using Fisher's z. Returns (z statistic, one-sided p-value for rho2 > rho1)."""
    z = (np.arctanh(rho2) - np.arctanh(rho1)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return float(z), float(1 - stats.norm.cdf(z))


def contagion_table(returns: pd.DataFrame, source: str, calm: pd.DataFrame, crisis: pd.DataFrame) -> pd.DataFrame:
    """Calm vs crisis correlation of every asset with ``source``, raw and Forbes-Rigobon adjusted."""
    v_calm, v_crisis = calm[source].var(), crisis[source].var()
    rows = []
    for a in returns.columns.drop(source):
        r0, r1 = calm[source].corr(calm[a]), crisis[source].corr(crisis[a])
        adj = forbes_rigobon(r1, v_crisis, v_calm)
        z, p = fisher_z_test(r0, len(calm), adj, len(crisis))
        rows.append({"asset": a, "calm corr": r0, "crisis corr": r1, "adjusted crisis corr": adj,
                     "z (adjusted vs calm)": z, "p-value": p, "contagion (5%)": p < 0.05})
    return pd.DataFrame(rows).set_index("asset")
