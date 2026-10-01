"""
Long/short VOLATILITY across stocks -- all trades in options.

Each day, rank the tickers by how expensive their options look:
  vrp:    log(30-day implied vol) - log(forecast of real movement), forecast trained on earlier years only
  iv_rv:  log(30-day implied vol / last 21 days' real vol)
SELL the near-the-money put+call on the k most expensive, BUY them on the k cheapest. Every pair is
sized to the same vega ($100 per vol point), so a market-wide jump in volatility (2018, 2020, 2022)
hits the short side and pays the long side about equally -- that's what sank the sell-only tests.
Delta-hedged daily with the stock. Held to expiry (spread paid once) or 10 days.

Rules picked on 2016-2021 at full bid/ask, checked ONCE on 2022-today (full and half spread).
Result per day = the pairs entered that day, $ for $100-per-vol-point positions.

Run:  python vol_longshort.py
"""
import itertools

import numpy as np
import pandas as pd

import opt_lib as ol
import strategy_search as ss

GROUPS = ("core", "extra", "more")
SELECT = range(2016, 2022)
ss.PANEL_GROUPS = GROUPS

legs = ol.load_legs(GROUPS)
sig = ol.vrp_signal(legs).rename(columns={"signal": "vrp"})
legs = legs.merge(sig, on=["date", "ticker"])
legs["iv_rv"] = np.log(legs["atm30"] / legs["rv21"])
legs["pair_vega"] = legs.groupby(["date", "ticker"])["vega"].transform("sum")
print(f"{legs['ticker'].nunique()} tickers, {legs['date'].nunique():,} days ({legs['date'].min()} to {legs['date'].max()})")

GRID = {"signal": ["vrp", "iv_rv"], "k": [2, 4], "hold": ["expiry", 10]}
rules = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]


def picks(sig_name, k):
    r = legs.groupby("date")[sig_name].rank(method="dense", ascending=False)
    n = legs.groupby("date")[sig_name].transform(lambda x: x.nunique())
    return legs[r <= k], legs[r > n - k]          # (most expensive -> sell, cheapest -> buy)


def scale(t):
    """$ for a pair sized to $100 of vega per vol point: contracts = 1 / pair vega per share."""
    return t.assign(pnl=t["pnl"] / t["pair_vega"])


results = {}
for r in rules:
    rich, cheap = picks(r["signal"], r["k"])
    out = {}
    for name, f in [("full", 1.0), ("half", 0.5)]:
        if r["hold"] == "expiry":
            s = scale(ol.hold_to_expiry(rich, True, f, "sell"))
            b = scale(ol.hold_to_expiry(cheap, True, f, "buy"))
        else:
            parts_s, parts_b = [], []
            for year in sorted(legs["date"].str[:4].unique()):
                ry, cy = rich[rich["date"].str[:4] == year], cheap[cheap["date"].str[:4] == year]
                days = pd.concat([ry["date"], cy["date"]])
                first, last = ss.day_no[days.min()], min(ss.day_no[days.max()] + r["hold"], len(ss.all_days) - 1)
                panel = ss.panel_for(ss.all_days[first:last + 1])
                parts_s.append(scale(ss.trade_pnl(ry, r["hold"], panel, "sell", fill=f)))
                parts_b.append(scale(ss.trade_pnl(cy, r["hold"], panel, "buy", fill=f)))
                del panel
            s, b = pd.concat(parts_s), pd.concat(parts_b)
        day = pd.DataFrame({"short": s.groupby("date")["pnl"].sum(), "long": b.groupby("date")["pnl"].sum()}).fillna(0)
        day["both"] = day["short"] + day["long"]
        out[name] = day
    results[str(r)] = (r, out)
    print(f"{r} done", flush=True)


def stats(day, lags):
    x = day["both"]
    return {"days": len(x), "$/day": x.mean(), "short side $/day": day["short"].mean(), "long side $/day": day["long"].mean(),
            "% days +": (x > 0).mean() * 100, "t": ss.nw_tstat(x.values, lags)}


rows = []
for key, (r, out) in results.items():
    d = out["full"]
    lags = 21 if r["hold"] == "expiry" else r["hold"]
    rows.append({**r, **{f"sel {k}": v for k, v in stats(d[d.index.str[:4].astype(int).isin(SELECT)], lags).items()}})
res = pd.DataFrame(rows)
print("\nRules on 2016-2021 (full bid/ask), $ per day for $100-vega positions:")
print(res.sort_values("sel t", ascending=False).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
best = res.loc[res["sel t"].idxmax()]
r, out = list(results.values())[int(res["sel t"].idxmax())]
lags = 21 if r["hold"] == "expiry" else r["hold"]
print(f"Chosen: {r}")
for name, label in [("full", "1.0 (full bid/ask)"), ("half", "0.5 (half the spread)")]:
    d = out[name]
    c = stats(d[d.index >= "2022-01-01"], lags)
    print(f"  CONFIRMATION 2022-today, fill {label}: {c['days']} days, ${c['$/day']:,.1f}/day "
          f"(short side ${c['short side $/day']:,.1f}, long side ${c['long side $/day']:,.1f}), "
          f"{c['% days +']:.0f}% of days +, t {c['t']:.1f}")
d = out["full"]
print("  by year ($/day, full bid/ask):", d.groupby(d.index.str[:4])["both"].mean().round(1).to_dict())

# ---------- chart: cumulative result of the chosen rule ----------
import charts_lib
p = charts_lib.cumulative({"full bid/ask": out["full"]["both"], "half the spread": out["half"]["both"],
                           "short side only (full)": out["full"]["short"], "long side only (full)": out["full"]["long"]},
                          f"Long/short volatility, chosen rule {r}", "vol_longshort.png")
print(f"Chart: {p}")
