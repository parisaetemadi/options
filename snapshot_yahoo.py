"""
Save today's real option quotes (bid/ask/volume/open interest) near the market close.
Yahoo's quotes come from the exchanges (~15 min delayed). Massive's free plan has no
bid/ask, so this builds our own day-by-day history of quotes going forward.

One file per day: data/yahoo/YYYY-MM-DD.csv.gz
Run:  python snapshot_yahoo.py      (on GitHub it runs by itself several times every weekday afternoon;
      the last run before the 4pm close wins)

--broad: the rest of the top 100 most-traded option names (86 more), lighter (OTM only, strikes +-20%, expiries up to 100 days), once a day
(the first run after 2:30pm New York time) -> data/yahoo_broad/YYYY-MM-DD.csv.gz.
Breadth for testing signals across many stocks; kept separate from the main 14.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "GOOGL", "AMD", "GLD", "TLT", "XLF"]
# Top 100 US option underlyings by 2026 average daily option volume (Cboe exchanges, via macroption.com,
# data to 2026-08-28): the 29 biggest ETFs (leveraged/inverse ETFs and VXX left out) + the top 71 stocks
# (GOOG left out: same company as GOOGL). The main 14 above are part of it and aren't repeated here.
BROAD = ["SLV", "IBIT", "USO", "SMH", "ETHA", "GDX", "HYG", "XLE", "EEM", "KWEB", "IGV", "EWZ", "EWY", "EFA", "FXI",
         "ARKK", "XLP", "KRE", "DIA", "MAGS", "XLI", "SILJ", "XLB", "INTC", "MU", "PLTR", "MSTR", "NFLX", "SMCI",
         "SOFI", "MARA", "IREN", "SPCX", "ORCL", "AVGO", "HOOD", "NOK", "COIN", "NBIS", "CRWV", "ONDS", "TSM", "BMNR",
         "BABA", "SNDK", "MRVL", "GME", "ASTS", "HIMS", "WULF", "NOW", "BAC", "RKLB", "OPEN", "RIVN", "MRNA", "WMT",
         "NKE", "F", "CRCL", "AI", "PFE", "AAL", "PYPL", "BE", "CIFR", "APLD", "SNAP", "NIO", "UNH", "NU", "QCOM",
         "CORZ", "XOM", "RIOT", "POET", "UBER", "CRM", "RGTI", "IONQ", "AMC", "CVNA", "BA", "EOSE", "SMR", "T"]
IS_BROAD = "--broad" in sys.argv
if IS_BROAD:
    TICKERS = BROAD
STRIKE_RANGE = 0.20 if IS_BROAD else 0.30   # keep strikes within this range of the stock price
MAX_DAYS = 100 if IS_BROAD else 400         # keep expirations up to this many days out
OUT = Path(__file__).parent / "data" / ("yahoo_broad" if IS_BROAD else "yahoo")


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
            if IS_BROAD:                                          # out-of-the-money only
                t = t[t["strike"] > spot] if kind == "call" else t[t["strike"] < spot]
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
    minute = now.hour * 60 + now.minute
    if minute < 9 * 60 + 30:
        return print("Market hasn't opened yet -- nothing to save.")
    path = OUT / f"{now:%Y-%m-%d}.csv.gz"
    if IS_BROAD and (minute < 14 * 60 + 30 or path.exists()):
        return print("Broad list: once a day, first run after 2:30pm -- nothing to do now.")
    if minute >= 16 * 60 and path.exists():
        return print("Already have today's snapshot from before the close -- keeping it.")
    frames = []
    for t in TICKERS:
        try:
            df = snapshot(t, now)
        except Exception as e:                    # one bad ticker shouldn't lose the whole day
            print(f"{t}: FAILED ({str(e)[:100]})")
            continue
        live = ((df["bid"] > 0) & (df["ask"] > 0)).sum() if len(df) else 0
        print(f"{t}: {len(df)} options, {live} with a live bid/ask")
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    if ((data["bid"] > 0) & (data["ask"] > 0)).mean() < 0.3:
        return print("Most quotes are empty (holiday or market closed?) -- not saving.")
    OUT.mkdir(parents=True, exist_ok=True)   # later runs overwrite earlier ones, so the latest wins
    data.replace({np.nan: None}).to_csv(path, index=False)
    print(f"Saved {len(data)} quotes to {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
