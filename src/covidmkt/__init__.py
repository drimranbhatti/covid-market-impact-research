"""covidmkt - cross-asset analysis of the COVID-19 market shock.

data         : load prices, asset metadata, period definitions
regimes      : per-period performance statistics and drawdowns
correlation  : rolling / regime correlations and Forbes-Rigobon contagion test
volatility   : GJR-GARCH(1,1) estimation and volatility persistence
spillover    : Diebold-Yilmaz spillover index from a VAR with generalised FEVD
stress       : historical-scenario and Monte Carlo stress tests (VaR / ES)
"""
from . import correlation, data, regimes, spillover, stress, volatility

__all__ = ["data", "regimes", "correlation", "volatility", "spillover", "stress"]
__version__ = "1.0.0"
