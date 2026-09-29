"""
Stress-test the one promising backtest rule before believing it:
  SELL options whose market IV is more than 3 vol points above the model's, delta-hedge daily,
  close after 10 trading days (sell at the bid, buy back at the ask).

Checks:
  1. Trades that dropped out of our data (usually big stock moves -- often the LOSERS for a seller)
     are NOT skipped: they're priced with Black-Scholes along the stock's real path, keeping the
     option's entry IV, and charged a full extra bid/ask spread to be conservative.
  2. Overlap: a 10-day trade opened each day overlaps the next 9, which inflates a plain t-stat.
     We use a Newey-West t-stat and also 10 separate non-overlapping samples.
  3. Where the money comes from: by year, ticker, put/call, and real vs parity-rebuilt exits.
  4. Concentration and worst losing stretch.

Run:  python check_edge.py     (after train.py)
"""
import numpy as np
import pandas as pd

import backtest as bt
from pricing import bs_price

H, TH = 10, 0.03


def nw_tstat(x, lags):
    """t-stat of the mean, corrected for overlapping (autocorrelated) observations."""
    x = np.asarray(x, float)
    n, e = len(x), x - x.mean()
    var = (e @ e) / n
    for l in range(1, lags + 1):
        var += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return x.mean() / np.sqrt(var / n)


t = bt.hold(bt.p[bt.p["gap"] > TH], H, keep_dropped=True).reset_index(drop=True)
alive = t["alive"].values
print(f"Rule: sell when market IV > model IV + {TH:.0%}, hold {H} days. Trades: {len(t):,} "
      f"({(~alive).sum():,} dropped out of our data before exit = {(~alive).mean():.1%})")

# ---------- 1. price the dropped trades along the stock's real path ----------
und = pd.read_csv(bt.DATA / "databento" / "underlying.csv").set_index(["ticker", "date"])["close"]
d = t[~alive]
spot_prev, delta_prev = d["spot"].values, d["delta"].values
hedge = np.zeros(len(d))
for k in range(1, H + 1):
    day_k = [bt.days[i + k] for i in d["day_no"]]
    spot_k = und.reindex(list(zip(d["ticker"], day_k))).values
    elapsed = (pd.to_datetime(day_k).values - pd.to_datetime(d["date"]).values) / np.timedelta64(365, "D")
    T_k = np.maximum(d["T"].values - elapsed, 1 / 365)
    hedge += delta_prev * (spot_k - spot_prev)
    k_rows = d.assign(spot=spot_k, T=T_k)
    delta_prev = np.nan_to_num(bt.delta(k_rows))          # keeps the option's entry IV
    spot_prev = spot_k
value = bs_price(d["type"].values, spot_prev, d["strike"].values, T_k, d["tbill"].values, d["iv"].values,
                 d["div_yield"].values)
spread = (d["ask"] - d["bid"]).values
t.loc[~alive, "exit_mid"] = value
t.loc[~alive, "exit_ask"] = value + spread            # conservative: a FULL spread above fair value
t.loc[~alive, "hedge"] = hedge
t["pnl"] = -((t["exit_ask"] - t["bid"]) - t["hedge"]) * 100
t["year"] = t["date"].str[:4]

# ---------- 2. is it luck? ----------
daily = t.groupby("date")["pnl"].mean()
print(f"\nAverage per trade after costs: ${t['pnl'].mean():,.2f}  (only trades that stayed in data: "
      f"${t.loc[alive, 'pnl'].mean():,.2f}; dropped trades, priced conservatively: ${t.loc[~alive, 'pnl'].mean():,.2f})")
print(f"Days with trades: {len(daily)}; profitable days: {(daily > 0).mean():.0%}")
print(f"t-stat, plain: {daily.mean() / daily.std() * np.sqrt(len(daily)):.1f}   "
      f"corrected for overlap (Newey-West): {nw_tstat(daily.values, H):.1f}")
sub = []
for k in range(H):
    s = t[t["day_no"] % H == k].groupby("date")["pnl"].mean()
    sub.append(s.mean() / s.std() * np.sqrt(len(s)))
print(f"10 non-overlapping samples (a trade every 10th day): t-stats {np.round(sub, 1).tolist()}")

# ---------- 3. where does the money come from? ----------
def table(by):
    g = t.groupby(by)["pnl"]
    return pd.DataFrame({"trades": g.size(), "$/trade": g.mean().round(2), "total $": g.sum().round(0),
                         "% winners": (g.apply(lambda x: (x > 0).mean()) * 100).round(0)})
print("\nBy year:\n" + table("year").to_string())
print("\nBy ticker:\n" + table("ticker").to_string())
print("\nBy option type:\n" + table("type").to_string())
t["exit"] = np.where(~alive, "dropped (BS-priced)", np.where(t["exit_real"].astype(bool), "real quote", "rebuilt by parity"))
print("\nBy how the exit was priced:\n" + table("exit").to_string())

# ---------- 4. concentration and worst stretch ----------
by_day_total = t.groupby("date")["pnl"].sum().sort_index()
top = by_day_total.sort_values(ascending=False)
print(f"\nTotal P&L (1 contract per trade): ${by_day_total.sum():,.0f}; "
      f"best 10 entry days = {top.head(10).sum() / by_day_total.sum():.0%} of it")
cum = by_day_total.cumsum()
dd = cum - cum.cummax()
print(f"Worst losing stretch (peak to trough, by entry day): ${dd.min():,.0f}, "
      f"from {cum[:dd.idxmin()].idxmax()} to {dd.idxmin()}")
print("Worst 5 entry days:", by_day_total.nsmallest(5).round(0).to_dict())
