"""
Daily open/high/low/close for the 14 stocks from Yahoo (free), 2013-today -> data/stocks/.

Why: closes alone see only one price a day. The day's range (high vs low) and the overnight gap
(yesterday's close -> today's open) measure volatility much more precisely -- nearly as well as
minute data -- which should help forecast how much a stock will really move.

Yahoo's OHLC are split-adjusted, but every feature below is a ratio within Yahoo's own series,
so splits cancel out. Checked against our Databento closes (same days, same moves).

Features per ticker and day (data/stocks/range_vol.csv), annualized vols:
  rs5, rs21, rs63    Garman-Klass (daytime, from open/high/low/close) + overnight gap, rolling mean
  on_share21         share of the last 21 days' variance that happened overnight (news-driven gaps)

Run:  python fetch_ohlc.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "stocks"
TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "GOOGL", "AMD", "GLD", "TLT", "XLF", "XOM"]


def download():
    raw = yf.download(TICKERS, start="2013-01-01", auto_adjust=False, actions=False, progress=False, group_by="ticker")
    rows = []
    for t in TICKERS:
        d = raw[t][["Open", "High", "Low", "Close"]].dropna().reset_index()
        d.columns = ["date", "open", "high", "low", "close"]
        rows.append(d.assign(ticker=t, date=d["date"].dt.strftime("%Y-%m-%d")))
    o = pd.concat(rows, ignore_index=True)[["ticker", "date", "open", "high", "low", "close"]]
    today = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
    return o[o["date"] < today]                  # today's bar isn't final until the close


def check(o):
    """Stop if the data looks wrong."""
    bad = (o["high"] < o[["open", "close"]].max(axis=1) * 0.9999) | (o["low"] > o[["open", "close"]].min(axis=1) * 1.0001) \
          | (o[["open", "high", "low", "close"]] <= 0).any(axis=1)
    print(f"Rows: {len(o):,}; impossible bars (high < open/close, low > open/close, <= 0): {bad.sum()}")
    flat = (o["high"] == o["low"]).mean()
    print(f"Bars with high == low: {flat:.3%}")
    # daily moves vs our Databento closes (split-corrected), same ticker and day
    db = pd.read_csv(ROOT / "data" / "databento" / "underlying.csv").sort_values(["ticker", "date"])
    sp = pd.read_csv(ROOT / "data" / "events" / "splits.csv")
    db["ret_db"] = db.groupby("ticker")["close"].transform(lambda c: np.log(c / c.shift(1)))
    db = db.merge(sp, on=["ticker", "date"], how="left")
    db["ret_db"] += np.log(db["ratio"].fillna(1.0))
    o = o.sort_values(["ticker", "date"]).copy()
    o["ret_y"] = o.groupby("ticker")["close"].transform(lambda c: np.log(c / c.shift(1)))
    m = o.merge(db[["ticker", "date", "ret_db"]], on=["ticker", "date"]).dropna()
    gap = (m["ret_y"] - m["ret_db"]).abs()
    print(f"Daily moves vs Databento closes on {len(m):,} ticker-days: median gap {gap.median():.5%}, "
          f"99.9th pct {gap.quantile(0.999):.4%}, days > 0.5%: {(gap > 0.005).sum()}")
    missing = db.merge(o[["ticker", "date"]], on=["ticker", "date"], how="left", indicator=True)
    missing = missing[(missing["_merge"] == "left_only") & (missing["date"] >= "2013-04-01")]
    print(f"Databento trading days missing from Yahoo: {len(missing)}")
    assert bad.mean() < 0.001 and gap.median() < 0.0005 and (gap > 0.005).mean() < 0.001, "OHLC data looks wrong"


def features(o):
    o = o.sort_values(["ticker", "date"]).copy()
    lo, lh, lc = (np.log(o[c] / o["open"]) for c in ("low", "high", "close"))
    gk = 0.5 * (lh - lo) ** 2 - (2 * np.log(2) - 1) * lc ** 2           # Garman-Klass daytime variance
    prev_close = o.groupby("ticker")["close"].shift(1)
    on = np.log(o["open"] / prev_close) ** 2                             # overnight gap variance
    o["var"], o["on"] = gk + on, on
    g = o.groupby("ticker")
    for n in (5, 21, 63):
        o[f"rs{n}"] = np.sqrt(g["var"].transform(lambda v: v.rolling(n).mean()) * 252)
    o["on_share21"] = g["on"].transform(lambda v: v.rolling(21).sum()) / g["var"].transform(lambda v: v.rolling(21).sum())
    return o[["ticker", "date", "rs5", "rs21", "rs63", "on_share21"]]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    o = download()
    check(o)
    o.to_csv(OUT / "ohlc_yahoo.csv", index=False)
    f = features(o).dropna()
    f.to_csv(OUT / "range_vol.csv", index=False)
    print(f"Saved {len(o):,} bars and {len(f):,} feature rows, {f['date'].min()} to {f['date'].max()}")
    print("Median 21-day range-based vol, by ticker:")
    print(f.groupby("ticker")["rs21"].median().round(3).to_dict())
