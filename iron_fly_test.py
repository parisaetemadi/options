"""
Selling the 11:15 straddle with a safety net: the IRON FLY.

At 11:15 New York time, on SPY and QQQ options of the nearest expiry (Databento quotes every 15 min):
  SELL the at-the-money call and put (the straddle the paper trader sells), and
  BUY  a call `w` above and a put `w` below the stock price (the "wings").
At the close (the day's last snapshot) everything is closed. Past a wing, any further move is paid by
the wing we own, so the worst loss is fixed in advance: (wing distance - net premium) x 100 + costs.

Costs: sell at the bid, buy at the ask (both ways), IBKR $0.70 per contract each way (8 per fly).
An option with no bid is worth $0; if a short leg has no quote at the close, it is bought back at its
value at expiry from the stock price.

Periods (rules are NOT re-picked here -- every width is shown in every period):
  never-seen earlier days (before 2026-01-23) | picked-on (2026-01-23 .. 06-30) | checked-once (07-01 ..)

Run:  python iron_fly_test.py
"""
import glob
import os
import time

import numpy as np
import pandas as pd

import charts_lib

FRESH_END, SPLIT = "2026-01-23", "2026-07-01"
ENTER = "11:15"
WIDTHS = [0.005, 0.0075, 0.01, 0.015, 0.02]          # wing distance from the stock price
COMM = 0.70
cols = ["date", "snap", "ticker", "type", "strike", "expiration", "bid", "ask"]

# ---------- the 11:15 and closing snapshots of every day ----------
parts = []
for f in sorted(glob.glob("data/databento/intraday/2*.parquet")):
    if time.time() - os.path.getmtime(f) < 120:      # the downloader may still be writing it
        continue
    d = pd.read_parquet(f, columns=cols)
    d["snap"] = pd.to_datetime(d["snap"], utc=True).dt.tz_convert("America/New_York")
    last = d["snap"].max()
    parts.append(d[(d["snap"].dt.strftime("%H:%M") == ENTER) | (d["snap"] == last)].assign(
        when=lambda x: np.where(x["snap"] == last, "close", "enter")))
q = pd.concat(parts, ignore_index=True).astype({"date": str, "ticker": str, "type": str, "expiration": str})
q["bid"] = q["bid"].fillna(0)                       # no bid = nobody will pay anything
q = q[q["ask"] > 0]
q["mid"] = (q["bid"] + q["ask"]) / 2
q = q[q["expiration"] == q.groupby(["date", "ticker"])["expiration"].transform("min")]     # nearest expiry
print(f"{q['date'].nunique()} days, {q['date'].min()} .. {q['date'].max()}")


def parity_spot(g):
    p = g[g["bid"] > 0].pivot_table(index="strike", columns="type", values="mid")
    if not {"call", "put"} <= set(p.columns) or p.dropna().empty:
        return np.nan
    p = p.dropna()
    k = (p["call"] - p["put"]).abs().idxmin()
    return k + p.loc[k, "call"] - p.loc[k, "put"]


closes = pd.read_csv("data/databento/underlying.csv").set_index(["ticker", "date"])["close"].to_dict()
rows = []
for (day, ticker), g in q.groupby(["date", "ticker"]):
    ent, cls = g[g["when"] == "enter"], g[g["when"] == "close"]
    if ent.empty or cls.empty:
        continue
    s_in, s_out = parity_spot(ent), parity_spot(cls)
    if not np.isfinite(s_out):                       # at the bell one side can be all no-bid: use the official close
        s_out = closes.get((ticker, day), np.nan)
    if not np.isfinite(s_in) or not np.isfinite(s_out):
        continue
    book_in = ent.set_index(["type", "strike"])
    book_out = cls.set_index(["type", "strike"])
    live = ent[(ent["bid"] > 0) & (ent["ask"] > ent["bid"])]
    both = set(live.loc[live["type"] == "call", "strike"]) & set(live.loc[live["type"] == "put", "strike"])
    if not both:
        continue
    k0 = min(both, key=lambda k: abs(k - s_in))

    def close_price(kind, k, side):
        """Exit price: buy back shorts at the ask, sell longs at the bid; fall back to the value at expiry."""
        if (kind, k) in book_out.index:
            r = book_out.loc[(kind, k)]
            return float(r["ask"] if side == "short" else r["bid"])
        return max(0.0, (s_out - k) if kind == "call" else (k - s_out))

    short_in = book_in.loc[("call", k0), "bid"] + book_in.loc[("put", k0), "bid"]
    short_out = close_price("call", k0, "short") + close_price("put", k0, "short")
    naked = (short_in - short_out) * 100 - 4 * COMM
    rows.append({"date": day, "ticker": ticker, "width": "none (straddle)", "pnl": naked, "max_loss": np.nan})
    for w in WIDTHS:
        calls = live[(live["type"] == "call") & (live["strike"] >= s_in * (1 + w))]
        puts = live[(live["type"] == "put") & (live["strike"] <= s_in * (1 - w))]
        if calls.empty or puts.empty:
            continue                                  # strikes that far out weren't bought (range +-3%)
        kc, kp = calls["strike"].min(), puts["strike"].max()
        wings_in = book_in.loc[("call", kc), "ask"] + book_in.loc[("put", kp), "ask"]
        wings_out = close_price("call", kc, "long") + close_price("put", kp, "long")
        credit = short_in - wings_in
        pnl = (credit - (short_out - wings_out)) * 100 - 8 * COMM
        max_loss = (max(kc - k0, k0 - kp) - credit) * 100 + 8 * COMM
        rows.append({"date": day, "ticker": ticker, "width": f"{w:.2%}", "pnl": pnl, "max_loss": max_loss})

t = pd.DataFrame(rows)
t["period"] = np.select([t["date"] < FRESH_END, t["date"] < SPLIT], ["never-seen (before 01-23)", "picked-on (01-23..06-30)"],
                        "checked-once (07-01..)")


def stats(g):
    by_day = g.groupby("date")["pnl"].sum()
    return pd.Series({"trades": len(g), "$/trade": g["pnl"].mean(), "% won": (g["pnl"] > 0).mean() * 100,
                      "t (by day)": by_day.mean() / by_day.std() * np.sqrt(len(by_day)) if by_day.std() > 0 else np.nan,
                      "worst trade": g["pnl"].min(), "worst day (both)": by_day.min(), "max loss set": g["max_loss"].mean()})


order = ["none (straddle)"] + [f"{w:.2%}" for w in WIDTHS]
for period in ["never-seen (before 01-23)", "picked-on (01-23..06-30)", "checked-once (07-01..)"]:
    res = t[t["period"] == period].groupby("width").apply(stats).reindex(order)
    print(f"\n===== {period}: sell at 11:15, close at the close, full bid/ask + fees =====")
    print(res.to_string(float_format=lambda x: f"{x:,.1f}"))
print("\nAll periods together:")
print(t.groupby("width").apply(stats).reindex(order).to_string(float_format=lambda x: f"{x:,.1f}"))

day = t.groupby(["width", "date"])["pnl"].sum().unstack(0)
p = charts_lib.cumulative({w: day[w].dropna() for w in ["none (straddle)", "1.00%", "0.50%"] if w in day},
                          "Selling the 11:15 straddle (SPY+QQQ, 1 each): plain vs with wings", "iron_fly.png")
print(f"Chart: {p}")
