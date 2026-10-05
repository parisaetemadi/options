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
        buyable = ent[ent["ask"] > 0]                 # wings are bought: an ask is enough (far ones often have no bid)
        calls = buyable[(buyable["type"] == "call") & (buyable["strike"] >= s_in * (1 + w))]
        puts = buyable[(buyable["type"] == "put") & (buyable["strike"] <= s_in * (1 - w))]
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


# ---------- publish for the website's Validation tab (merged into data/paper_options/validation.json) ----------
import json
from pathlib import Path


def brief(g):
    if len(g) < 5:
        return {"n": int(len(g))}
    st = stats(g)
    return {"n": int(st["trades"]), "per_trade": round(float(st["$/trade"]), 2), "won": round(float(st["% won"]), 1),
            "t": round(float(st["t (by day)"]), 2)}


def card(name, title, width):
    g = t[t["width"] == width]
    per = {p: g[g["period"] == p] for p in g["period"].unique()}
    c = {"name": name, "title": title, "tried": 1,
         "selection": brief(per.get("picked-on (01-23..06-30)", g.iloc[:0])),
         "confirmation": brief(per.get("checked-once (07-01..)", g.iloc[:0])),
         "fresh": brief(per.get("never-seen (before 01-23)", g.iloc[:0]))}
    pos = [x.get("per_trade", 0) > 0 for x in (c["confirmation"], c["fresh"])]
    c["verdict"] = ("positive in both checks but weak — not proven" if all(pos) else
                    "positive in one check only — not proven" if any(pos) else "fails the checks")
    c["worst_day"] = round(float(g.groupby("date")["pnl"].sum().min()), 0)
    return c


vf = Path("data/paper_options/validation.json")
v = json.loads(vf.read_text()) if vf.exists() else {"rules": []}
mine = [card("iron-fly-1115 (paper-traded)", "sell the 11:15 straddle with wings 1% away, close everything at the close "
             f"(worst day so far ${t[t['width'] == '1.00%'].groupby('date')['pnl'].sum().min():,.0f}; loss capped)", "1.00%"),
        card("plain 11:15 straddle (dropped)", "the same without wings "
             f"(worst day ${t[t['width'] == 'none (straddle)'].groupby('date')['pnl'].sum().min():,.0f}, 2025-04-09)", "none (straddle)")]
v["rules"] = mine + [r for r in v.get("rules", []) if not r["name"].startswith(("iron-fly", "plain 11:15", "sell-straddle-1115"))]
v["fresh"] = f"{t['date'].min()} .. {FRESH_END}"
v["data"] = f"Databento SPY/QQQ option quotes every 15 min, {t['date'].min()} .. {t['date'].max()}"
vf.write_text(json.dumps(v, indent=1))
print(f"Updated {vf}")
