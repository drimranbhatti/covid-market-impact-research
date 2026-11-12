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
