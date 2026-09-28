"""
Save today's real option quotes (bid/ask/volume/open interest) near the market close.
Yahoo's quotes come from the exchanges (~15 min delayed). Massive's free plan has no
bid/ask, so this builds our own day-by-day history of quotes going forward.

One file per day: data/yahoo/YYYY-MM-DD.csv.gz
Run:  python snapshot_yahoo.py      (on GitHub it runs by itself every weekday afternoon)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]
STRIKE_RANGE = 0.30     # keep strikes within +-30% of the stock price
MAX_DAYS = 400          # keep expirations up to ~13 months out
OUT = Path(__file__).parent / "data" / "yahoo"


def retry(f, *args, tries=4):
    """Yahoo sometimes rate-limits; wait and try again."""
    for i in range(tries):
        try:
            return f(*args)
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  retrying after error: {e}")
            time.sleep(20 * (i + 1))


def snapshot(ticker, now):
    tk = yf.Ticker(ticker)
    spot = retry(lambda: tk.history(period="1d")["Close"].iloc[-1])
    tables = []
    for exp in retry(lambda: tk.options):
        days = (pd.Timestamp(exp) - now.tz_localize(None).normalize()).days
        if days < 0 or days > MAX_DAYS:
            continue
        chain = retry(tk.option_chain, exp)
        for kind, t in (("call", chain.calls), ("put", chain.puts)):
            t = t[(t["strike"] / spot - 1).abs() <= STRIKE_RANGE]
            tables.append(pd.DataFrame({
                "snapshot_time": now.strftime("%Y-%m-%d %H:%M %Z"),
                "ticker": ticker, "spot": round(spot, 4), "expiration": exp, "type": kind,
                "contract": t["contractSymbol"], "strike": t["strike"],
                "bid": t["bid"], "ask": t["ask"], "last": t["lastPrice"],
                "last_trade": t["lastTradeDate"].astype(str),
                "volume": t["volume"], "open_interest": t["openInterest"],
                "yahoo_iv": t["impliedVolatility"].round(5),
            }))
        time.sleep(0.5)   # be gentle with Yahoo
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def main():
    now = pd.Timestamp.now(tz="America/New_York")
    if now.weekday() >= 5:
        return print("Weekend -- market closed, nothing to save.")
    frames = []
    for t in TICKERS:
        df = snapshot(t, now)
        live = ((df["bid"] > 0) & (df["ask"] > 0)).sum() if len(df) else 0
        print(f"{t}: {len(df)} options, {live} with a live bid/ask")
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    if ((data["bid"] > 0) & (data["ask"] > 0)).mean() < 0.3:
        return print("Most quotes are empty (holiday or market closed?) -- not saving.")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{now:%Y-%m-%d}.csv.gz"
    data.replace({np.nan: None}).to_csv(path, index=False)
    print(f"Saved {len(data)} quotes to {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
