"""Download the cross-asset daily dataset (2018-2024) into data/prices.csv.

    python scripts/fetch_data.py
"""
from pathlib import Path

import yfinance as yf

ASSETS = {
    "SPY": ("US large-cap equities", "Equities"), "IWM": ("US small-cap equities", "Equities"),
    "EEM": ("Emerging-market equities", "Equities"),
    "TLT": ("20y+ US Treasuries", "Fixed income"), "LQD": ("IG corporate bonds", "Fixed income"),
    "HYG": ("High-yield bonds", "Fixed income"),
    "GC=F": ("Gold", "Commodities"), "USO": ("Crude oil (USO ETF)", "Commodities"), "HG=F": ("Copper", "Commodities"),
    "DX-Y.NYB": ("US dollar index", "FX"), "EURUSD=X": ("EUR/USD", "FX"), "JPY=X": ("USD/JPY", "FX"),
    "BTC-USD": ("Bitcoin", "Crypto"), "ETH-USD": ("Ether", "Crypto"),
    "^VIX": ("VIX", "Volatility"),
}

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    px = yf.download(list(ASSETS), start="2018-01-01", end="2025-01-01", auto_adjust=True, progress=False)["Close"]
    px = px[px["SPY"].notna()]  # US trading calendar
    px.round(6).to_csv(out / "prices.csv")
    print(px.shape)
