"""GJR-GARCH(1,1) with Student-t errors via the ``arch`` package."""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model


def fit_gjr(returns: pd.Series, dist="t"):
    """Fit GJR-GARCH(1,1) to percentage log returns. Returns the arch result object."""
    am = arch_model(returns * 100, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist=dist, rescale=False)
    return am.fit(disp="off", options={"maxiter": 500})


def summarize(res) -> dict:
    p = res.params
    alpha, gamma, beta = p.get("alpha[1]", np.nan), p.get("gamma[1]", 0.0), p.get("beta[1]", np.nan)
    persistence = alpha + beta + gamma / 2
    return {"omega": p["omega"], "alpha": alpha, "gamma (leverage)": gamma, "beta": beta,
            "persistence": persistence,
            "half-life (days)": np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.inf,
            "nu (t dof)": p.get("nu", np.nan), "log-lik": res.loglikelihood}


def conditional_vol(res, annualize=True) -> pd.Series:
    s = res.conditional_volatility / 100
    return s * np.sqrt(252) if annualize else s


# ---------------------------------------------------------------- forecasting helpers
def forward_realised_vol(returns: pd.Series, horizon: int = 22) -> pd.Series:
    """Annualised realised volatility over days t+1..t+horizon, stamped at t (the forecast target)."""
    r2 = returns.pow(2)
    fwd = r2[::-1].rolling(horizon).sum()[::-1].shift(-1)
    return np.sqrt(252 * fwd / horizon)


def har_features(returns: pd.Series) -> pd.DataFrame:
    """Corsi (2009) HAR regressors from daily squared returns: log realised variance over 1, 5 and 22 days, known at t."""
    r2 = returns.pow(2)
    return pd.DataFrame({"rv1": np.log(r2.clip(lower=1e-10)), "rv5": np.log(r2.rolling(5).mean()), "rv22": np.log(r2.rolling(22).mean())})


def qlike(actual_var: np.ndarray, forecast_var: np.ndarray) -> float:
    """QLIKE loss (Patton, 2011), robust to noise in the volatility proxy. Lower is better; 0 for a perfect forecast."""
    a, f = np.asarray(actual_var), np.asarray(forecast_var)
    return float(np.mean(a / f - np.log(a / f) - 1))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 22) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy with a Newey-West variance (lag h-1). Negative statistic: A is better."""
    from scipy.stats import norm
    d = np.asarray(loss_a) - np.asarray(loss_b); T = len(d); dm = d - d.mean()
    lrv = dm @ dm / T + 2 * sum((1 - k / h) * (dm[k:] @ dm[:-k]) / T for k in range(1, h))
    stat = d.mean() / np.sqrt(lrv / T)
    return float(stat), float(2 * (1 - norm.cdf(abs(stat))))
