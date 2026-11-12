import numpy as np
import pandas as pd
import pytest

from covidmkt import correlation, regimes, spillover, stress, volatility

rng = np.random.default_rng(0)
idx = pd.bdate_range("2018-01-01", periods=1500)
R = pd.DataFrame(rng.multivariate_normal([0, 0, 0], [[1, .5, .2], [.5, 1, .1], [.2, .1, 1]], 1500) * 0.01,
                 index=idx, columns=["A", "B", "C"])


def test_forbes_rigobon_no_vol_change_is_identity():
    assert correlation.forbes_rigobon(0.6, 1.0, 1.0) == pytest.approx(0.6)
    assert correlation.forbes_rigobon(0.6, 4.0, 1.0) < 0.6       # higher crisis variance shrinks rho


def test_gfevd_rows_sum_to_100_and_diagonal_case():
    tbl, total = spillover.spillover_table(R, lags=1, horizon=10)
    assert np.allclose(tbl.iloc[:3, :3].sum(axis=1), 100)
    assert 0 < total < 100
    indep = pd.DataFrame(rng.standard_normal((2000, 3)), columns=list("xyz"))
    _, t0 = spillover.spillover_table(indep, lags=1, horizon=10)
    assert t0 < 2                                                  # independent series -> ~0 spillover


def test_drawdown_and_period_stats():
    lr = pd.Series([0.1, -0.2, 0.05])
    assert regimes.max_drawdown(lr) == pytest.approx(np.exp(-0.2) - 1)


def test_garch_fits_and_persistence_below_one():
    r = pd.Series(rng.standard_t(6, 2000) * 0.01, index=pd.bdate_range("2015-01-01", periods=2000))
    s = volatility.summarize(volatility.fit_gjr(r))
    assert 0 < s["persistence"] < 1


def test_var_es_ordering_and_t_tails():
    w = pd.Series([0.5, 0.3, 0.2], index=R.columns); mu = R.mean(); cov = R.cov()
    pnl_n = stress.monte_carlo(w, mu, cov, n_sims=100_000)
    pnl_t = stress.monte_carlo(w, mu, cov, n_sims=100_000, dof=4)
    v, es = stress.var_es(pnl_n)
    assert es > v > 0
    assert stress.var_es(pnl_t, 0.999)[0] > stress.var_es(pnl_n, 0.999)[0]   # fatter tails
    v0, _ = stress.var_es(stress.monte_carlo(w, mu * 0, cov, n_sims=200_000))   # zero drift, like the formula
    # MC losses are simple returns; the parametric quantile is in log space
    assert v0 == pytest.approx(-np.expm1(-stress.parametric_var(w.values, cov.values)), rel=0.02)
