"""
Search for a trading rule the honest way, on out-of-sample predictions from walkforward.py:
  1. try a small grid of rules on 2016-2021 ("selection" years)
  2. pick the best by overlap-corrected t-stat
  3. check that ONE rule once on 2022-today ("confirmation" years) -- no more tuning after that

A rule = which side (sell options the model finds expensive / buy ones it finds cheap),
how big the IV gap must be, how long to hold, and a cap on trades per ticker per day
(the biggest gaps first). Every trade is delta-hedged daily and pays the real bid/ask both ways.
Trades whose option leaves our data before the exit (in-the-money beyond +-20%, etc.) are priced
with Black-Scholes along the stock's real path at their entry IV, plus a full extra spread.

Run:  python strategy_search.py
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from pricing import bs_price, implied_vol_many

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PRED = ROOT / "models" / "walkforward_predictions.parquet"
SELECT_YEARS = range(2016, 2022)
MIN_ABS_LOG_M = 0.03
KEY = ["ticker", "type", "strike", "expiration"]
GRID = {"side": ["sell", "buy"], "gap": [0.02, 0.03, 0.05], "hold": [5, 10, 20], "cap": [None, 5, 1]}


def delta(d):
    T, s = d["T"], d["iv"]
    d1 = (np.log(d["spot"] / d["strike"]) + (d["tbill"] - d["div_yield"] + 0.5 * s**2) * T) / (s * np.sqrt(T))
    disc = np.exp(-d["div_yield"] * T)
    return np.where(d["type"] == "call", disc * norm.cdf(d1), disc * (norm.cdf(d1) - 1))


def nw_tstat(x, lags):
    x = np.asarray(x, float)
    if len(x) < 3:
        return np.nan
    n, e = len(x), x - x.mean()
    var = (e @ e) / n
    for l in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return x.mean() / np.sqrt(var / n)


rates = pd.read_csv(DATA / "market" / "vix_rates.csv")[["date", "tbill"]]
divs = pd.read_csv(DATA / "events" / "div_yield.csv")
closes = pd.read_csv(DATA / "databento" / "underlying.csv").set_index(["ticker", "date"])["close"]
all_days = np.array(sorted(f.stem for f in (DATA / "databento" / "quotes" / "core").glob("*/*.parquet")))
day_no = {d: i for i, d in enumerate(all_days)}


def panel_for(days):
    """Daily quotes (incl. rebuilt in-the-money ones via put-call parity) for these trading days."""
    f = pd.concat([pd.read_parquet(DATA / "databento" / "quotes" / "core" / d[:4] / f"{d}.parquet") for d in days],
                  ignore_index=True).astype({"date": str, "ticker": str, "type": str, "expiration": str})
    f = f.merge(rates, on="date").merge(divs, on=["ticker", "date"], how="left").fillna({"div_yield": 0})
    f["T"] = (pd.to_datetime(f["expiration"]) - pd.to_datetime(f["date"])).dt.days / 365
    f["mid"] = (f["bid"] + f["ask"]) / 2
    f["iv"] = implied_vol_many(f["mid"], f["spot"], f["strike"], f["T"], f["tbill"].values, f["type"], q=f["div_yield"].values)
    f["delta"] = np.nan_to_num(delta(f))
    f["day_no"] = f["date"].map(day_no)
    carry = f["spot"] * np.exp(-f["div_yield"] * f["T"]) - f["strike"] * np.exp(-f["tbill"] * f["T"])
    is_call = f["type"] == "call"
    sign = np.where(is_call, -1.0, 1.0)
    mirror = f.assign(type=np.where(is_call, "put", "call"), bid=f["bid"] + sign * carry, ask=f["ask"] + sign * carry,
                      mid=f["mid"] + sign * carry,
                      delta=f["delta"] + np.where(is_call, -1.0, 1.0) * np.exp(-f["div_yield"] * f["T"]))
    return (pd.concat([f, mirror]).drop_duplicates(KEY + ["day_no"], keep="first")
            .set_index(KEY + ["day_no"])[["bid", "ask", "mid", "spot", "delta"]])


def trade_pnl(entry, h, panel, side):
    """P&L in $ per contract for each entry, held h days, delta-hedged daily, real bid/ask both ways."""
    entry = entry[(entry["day_no"] + h < len(all_days)) & (entry["T"] * 365 >= 7 + np.ceil(h * 7 / 5) + 3)]
    if entry.empty:
        return entry.assign(pnl=[])
    hedge, alive = np.zeros(len(entry)), np.ones(len(entry), bool)
    prev_spot, prev_delta = entry["spot"].values, entry["delta"].values
    for k in range(1, h + 1):
        nd = panel.reindex(pd.MultiIndex.from_frame(entry[KEY].assign(day_no=entry["day_no"] + k)))
        ok = nd["mid"].notna().values
        alive &= ok
        # fall back to the stock's real close and a Black-Scholes delta at entry IV when the quote is missing
        spot_k = np.where(ok, nd["spot"].values,
                          closes.reindex(list(zip(entry["ticker"], all_days[entry["day_no"] + k]))).values)
        elapsed = (pd.to_datetime(all_days[entry["day_no"] + k]) - pd.to_datetime(entry["date"].values)).days.values / 365
        T_k = np.maximum(entry["T"].values - elapsed, 1 / 365)
        bs_delta = np.nan_to_num(delta(entry.assign(spot=spot_k, T=T_k)))
        hedge += prev_delta * (spot_k - prev_spot)
        prev_spot, prev_delta = spot_k, np.where(ok, nd["delta"].values, bs_delta)
    value = bs_price(entry["type"].values, prev_spot, entry["strike"].values, T_k, entry["tbill"].values,
                     entry["iv"].values, entry["div_yield"].values)
    spread = (entry["ask"] - entry["bid"]).values
    exit_bid = np.where(alive, nd["bid"].values, value - spread)      # dropped trades: a full extra spread
    exit_ask = np.where(alive, nd["ask"].values, value + spread)
    if side == "buy":
        pnl = (exit_bid - entry["ask"].values) - hedge
    else:
        pnl = -((exit_ask - entry["bid"].values) - hedge)
    return entry.assign(pnl=pnl * 100, dropped=~alive)


def pick(p, side, gap, cap):
    sel = p[(p["gap"] > gap) if side == "sell" else (p["gap"] < -gap)]
    if cap:
        sel = (sel.assign(size=sel["gap"].abs()).sort_values("size", ascending=False)
               .groupby(["date", "ticker"]).head(cap))
    return sel


def stats(t, h):
    if t.empty:
        return {"trades": 0}
    daily = t.groupby("date")["pnl"].mean()
    return {"trades": len(t), "$/trade": t["pnl"].mean(), "% days +": (daily > 0).mean(),
            "t (overlap-corrected)": nw_tstat(daily.values, h), "dropped": t["dropped"].mean()}


if __name__ == "__main__":
    p = pd.read_parquet(PRED)
    p = p[~p["atm_used"] & (np.abs(np.log(p["strike"] / p["spot"])) >= MIN_ABS_LOG_M)].copy()
    p["gap"] = p["iv"] - p["iv_pred"]
    p["delta"] = delta(p)
    p["day_no"] = p["date"].map(day_no)
    p = p.dropna(subset=["day_no"]).astype({"day_no": int})
    configs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    trades = {i: [] for i in range(len(configs))}
    max_h = max(GRID["hold"])
    for year in sorted(p["date"].str[:4].unique()):
        py = p[p["date"].str[:4] == year]
        first, last = day_no[py["date"].min()], min(day_no[py["date"].max()] + max_h, len(all_days) - 1)
        panel = panel_for(all_days[first:last + 1])
        for i, c in enumerate(configs):
            trades[i].append(trade_pnl(pick(py, c["side"], c["gap"], c["cap"]), c["hold"], panel, c["side"])
                             [["date", "ticker", "type", "pnl", "dropped"]])
        print(f"{year} done", flush=True)
        del panel

    rows = []
    for i, c in enumerate(configs):
        t = pd.concat(trades[i], ignore_index=True)
        t["period"] = np.where(t["date"].str[:4].astype(int).isin(SELECT_YEARS), "select", "confirm")
        rows.append({**c, **{f"select {k}": v for k, v in stats(t[t["period"] == "select"], c["hold"]).items()},
                     "_trades": t})
    res = pd.DataFrame(rows)
    show = res.drop(columns="_trades").sort_values("select t (overlap-corrected)", ascending=False)
    print("\nAll rules on the SELECTION years 2016-2021 (best first):")
    print(show.head(15).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    best = res.loc[show.index[0]]
    t = best["_trades"]
    conf = t[t["period"] == "confirm"]
    s = stats(conf, best["hold"])
    print(f"\nChosen rule: {best['side']} when |gap| > {best['gap']:.0%}, hold {best['hold']} days, "
          f"cap {best['cap'] or 'none'} per ticker per day")
    print(f"CONFIRMATION 2022-today (checked once): {s['trades']:,} trades, ${s.get('$/trade', np.nan):,.2f}/trade, "
          f"{s.get('% days +', np.nan):.0%} of days profitable, t = {s.get('t (overlap-corrected)', np.nan):.1f}")
    print("By year:", t.groupby(t["date"].str[:4])["pnl"].mean().round(2).to_dict())
    print("By ticker (confirmation):", conf.groupby("ticker")["pnl"].mean().round(2).to_dict())
