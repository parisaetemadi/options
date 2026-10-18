"""
Free INTRADAY option history, built going forward: every ~30 minutes during market hours, save
SPY, QQQ, TSLA, IWM and NVDA option quotes near the money (strikes within +-5%, expiries up to 30 days incl. same-day)
from Yahoo, plus SPY/QQQ/IWM puts 0-15% below the stock up to 45 days out (for the monthly put spread). Each row keeps the snapshot time and the last-trade time (Yahoo can lag ~15 min).

One file per snapshot: data/yahoo_intraday/YYYY-MM-DD/HHMM.csv.gz (New York time)
Run:  python snapshot_intraday.py      (on GitHub: .github/workflows/intraday.yml)
"""
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = ["SPY", "QQQ", "TSLA", "IWM", "NVDA"]   # the cheapest options to trade (spreads 0.4-2.2%)
STRIKE_RANGE, MAX_DAYS = 0.05, 30
# for the monthly put spread (paper_options.py "put-spread-monthly"): puts 0-15% below the stock, expiries up to 45 days
PUT_TICKERS, PUT_RANGE, PUT_MAX_DAYS = ["SPY", "QQQ", "IWM"], (0.85, 1.0), 45
OUT = Path(__file__).parent / "data" / "yahoo_intraday"


def main():
    now = pd.Timestamp.now(tz="America/New_York")
    minute = now.hour * 60 + now.minute
    if now.weekday() >= 5 or not (9 * 60 + 35 <= minute <= 16 * 60 + 5):
        return print(f"{now:%Y-%m-%d %H:%M} -- market closed, nothing to save.")
    rows = []
    for t in TICKERS:
        tk = yf.Ticker(t)
        spot = tk.fast_info["last_price"]
        for exp in tk.options:
            days = (pd.Timestamp(exp) - now.tz_localize(None).normalize()).days
            wide = t in PUT_TICKERS and days <= PUT_MAX_DAYS
            if days < 0 or (days > MAX_DAYS and not wide):
                continue
            ch = tk.option_chain(exp)
            for kind, d in (("call", ch.calls), ("put", ch.puts)):
                m = d["strike"] / spot
                keep = ((m - 1).abs() <= STRIKE_RANGE) & (days <= MAX_DAYS)
                if kind == "put" and wide:
                    keep |= m.between(*PUT_RANGE)
                d = d[keep]
                rows.append(pd.DataFrame({"snapshot_time": now.strftime("%Y-%m-%d %H:%M:%S %Z"), "ticker": t,
                                          "spot": round(spot, 4), "expiration": exp, "type": kind,
                                          "strike": d["strike"], "bid": d["bid"], "ask": d["ask"], "last": d["lastPrice"],
                                          "last_trade": d["lastTradeDate"].astype(str), "volume": d["volume"],
                                          "open_interest": d["openInterest"], "yahoo_iv": d["impliedVolatility"].round(5)}))
            time.sleep(0.3)
    data = pd.concat(rows, ignore_index=True)
    live = ((data["bid"] > 0) & (data["ask"] > 0)).mean()
    if live < 0.3:
        return print(f"Only {live:.0%} live quotes -- not saving.")
    path = OUT / f"{now:%Y-%m-%d}" / f"{now:%H%M}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    print(f"Saved {len(data):,} quotes ({live:.0%} live) to {path} ({path.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
