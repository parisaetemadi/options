"""
Two ideas for the 11:15 trade on SPY/QQQ (Databento quotes every 15 min, May 2023 - Sep 2026).

1. CALM-DAY FILTER. Selling the 11:15 straddle (or iron fly) wins on calm afternoons and loses on wild
   ones. Can what is known at 11:15 tell them apart? Inputs, all known by 11:15:
     imp        the 11:15 straddle price / stock price (the move the market expects for the rest of the day)
     vix1d_open Cboe's VIX1D at the open (expected move today, free from cboe.com), and its previous close
     vix_prev   VIX at the previous close
     rv_morn    how much the stock moved 9:45-11:15 (15-min steps), scaled to the rest of the day
     range_morn high-low 9:45-11:15 / price;  gap  |9:45 price / previous close - 1|;  prev_move  |yesterday|
     ratio_rv   imp / rv_morn  (is today's remaining move priced above what the morning suggests?)
     ratio_vix  imp / VIX1D's move for the rest of the day
     fomc       Fed decision day (2pm, after the entry);  cpi_jobs (released before the open);  weekday
   Tested WALK-FORWARD: each month from 2024-01 on is decided by a model/threshold fitted only on the
   months before it, then the filtered trades are compared with trading every day.

2. PINNING (a free stand-in for open interest). Open interest piles up at round strikes ($5/$10 levels);
   if dealers' hedging pulls prices toward them near expiry, closes should land near round numbers
   more often than chance (a $5 grid: 20% of closes within $0.50 by chance).

Run:  python calm_filter_test.py
"""
import contextlib
import glob
import io
import os
import runpy
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

NY = "America/New_York"
OOS_FROM = "2024-01"

# ---------- P&L of the 11:15 trades (plain straddle, iron flies) from iron_fly_test.py ----------
with contextlib.redirect_stdout(io.StringIO()):
    fly = runpy.run_path("iron_fly_test.py")["t"]
pnl = fly.pivot_table(index=["date", "ticker"], columns="width", values="pnl").reset_index()
pnl = pnl.rename(columns={"none (straddle)": "straddle", "1.00%": "fly1", "2.00%": "fly2"})[["date", "ticker", "straddle", "fly1", "fly2"]]

# ---------- the stock price at every snapshot (put-call parity, nearest expiry) and the 11:15 straddle ----------
rows, imp_rows = [], []
for f in sorted(glob.glob("data/databento/intraday/2*.parquet")):
    if time.time() - os.path.getmtime(f) < 120:
        continue
    d = pd.read_parquet(f, columns=["date", "snap", "ticker", "type", "strike", "expiration", "bid", "ask"])
    d = d.astype({"date": str, "ticker": str, "type": str, "expiration": str})
    d = d[(d["bid"] > 0) & (d["ask"] > d["bid"])]
    d = d[d["expiration"] == d.groupby("ticker")["expiration"].transform("min")]
    d["mid"] = (d["bid"] + d["ask"]) / 2
    p = d.pivot_table(index=["snap", "ticker", "strike"], columns="type", values="mid").dropna().reset_index()
    if p.empty or not {"call", "put"} <= set(p.columns):
        continue
    p["gap"] = (p["call"] - p["put"]).abs()
    p = p.loc[p.groupby(["snap", "ticker"])["gap"].idxmin()]
    p["spot"] = p["strike"] + p["call"] - p["put"]
    p["date"] = d["date"].iloc[0]
    rows.append(p[["date", "snap", "ticker", "spot"]])
    at = p[pd.to_datetime(p["snap"], utc=True).dt.tz_convert(NY).dt.strftime("%H:%M") == "11:15"]
    for _, r in at.iterrows():           # ATM straddle at 11:15: the strike nearest the stock with both sides live
        g = d[(d["snap"] == r["snap"]) & (d["ticker"] == r["ticker"])].pivot_table(index="strike", columns="type", values="mid").dropna()
        if len(g):
            k = (g.index.to_series() - r["spot"]).abs().idxmin()
            imp_rows.append({"date": r["date"], "ticker": r["ticker"], "imp": (g.loc[k, "call"] + g.loc[k, "put"]) / r["spot"]})
spots = pd.concat(rows, ignore_index=True)
spots["t"] = pd.to_datetime(spots["snap"], utc=True).dt.tz_convert(NY)
spots["hhmm"] = spots["t"].dt.strftime("%H:%M")
imp = pd.DataFrame(imp_rows)
print(f"{spots['date'].nunique()} days of 15-minute prices, {spots['date'].min()} .. {spots['date'].max()}")

# ---------- features known by 11:15 ----------
closes = pd.read_csv("data/databento/underlying.csv").pivot(index="date", columns="ticker", values="close")
def cboe(name):
    v = pd.read_csv(f"data/market/{name}_History.csv")
    v.index = pd.to_datetime(v["DATE"]).dt.strftime("%Y-%m-%d")
    return v
vix1d, vix = cboe("VIX1D"), cboe("VIX")
ev = pd.read_csv("data/events/macro.csv")
fomc = set(ev.loc[ev["type"] == "fomc", "reaction_day"])
cpi_jobs = set(ev.loc[ev["type"] != "fomc", "reaction_day"])

feat = []
for (day, ticker), g in spots.groupby(["date", "ticker"]):
    m = g[(g["hhmm"] >= "09:45") & (g["hhmm"] <= "11:15")].sort_values("t")
    if len(m) < 5 or m["hhmm"].iloc[-1] != "11:15" or ticker not in closes:
        continue
    prev_days = closes.index[closes.index < day]
    if len(prev_days) < 2 or prev_days[-1] not in vix1d.index:
        continue
    pc, ppc = closes.loc[prev_days[-1], ticker], closes.loc[prev_days[-2], ticker]
    r = np.diff(np.log(m["spot"].values))
    feat.append({"date": day, "ticker": ticker,
                 "vix1d_open": vix1d["OPEN"].get(day, np.nan), "vix1d_prev": vix1d.loc[prev_days[-1], "CLOSE"],
                 "vix_prev": vix["CLOSE"].get(prev_days[-1], np.nan),
                 "rv_morn": np.sqrt((r ** 2).sum()) * np.sqrt(19 / max(len(r), 1)),
                 "range_morn": (m["spot"].max() - m["spot"].min()) / m["spot"].iloc[-1],
                 "gap": abs(m["spot"].iloc[0] / pc - 1), "prev_move": abs(pc / ppc - 1),
                 "fomc": int(day in fomc), "cpi_jobs": int(day in cpi_jobs), "weekday": pd.Timestamp(day).weekday()})
X = pd.DataFrame(feat).merge(imp, on=["date", "ticker"]).merge(pnl, on=["date", "ticker"])
X["ratio_rv"] = X["imp"] / X["rv_morn"].clip(lower=1e-4)
X["ratio_vix"] = X["imp"] / (X["vix1d_open"] / 100 / np.sqrt(252) * np.sqrt(4.75 / 6.5))
X["is_qqq"] = (X["ticker"] == "QQQ").astype(int)
FEATS = ["imp", "vix1d_open", "vix1d_prev", "vix_prev", "rv_morn", "range_morn", "gap", "prev_move",
         "ratio_rv", "ratio_vix", "fomc", "cpi_jobs", "weekday", "is_qqq"]
X = X.dropna(subset=FEATS + ["straddle"]).sort_values(["date", "ticker"]).reset_index(drop=True)
print(f"{len(X)} day-ticker rows with every input, {X['date'].min()} .. {X['date'].max()}")


def stats(p, dates):
    if len(p) < 5:
        return {"trades": len(p)}
    by_day = p.groupby(dates).sum()
    return {"trades": len(p), "$/trade": p.mean(), "% won": (p > 0).mean() * 100,
            "t (by day)": by_day.mean() / by_day.std() * np.sqrt(len(by_day)), "worst day": by_day.min(), "total": p.sum()}


# ---------- 1a. one input at a time: result by fifth of that input (all months, descriptive) ----------
print("\n===== 1a. Sold straddle / 1% fly by fifth of each input (all months; descriptive, not a test) =====")
for f in ["imp", "vix1d_open", "ratio_vix", "ratio_rv", "rv_morn", "range_morn", "gap", "prev_move"]:
    b = pd.qcut(X[f], 5, labels=["lowest", "2", "3", "4", "highest"])
    s = X.groupby(b, observed=True)[["straddle", "fly1"]].mean().round(1)
    print(f"{f:11s} straddle {s['straddle'].tolist()}   fly1 {s['fly1'].tolist()}")
for f in ["fomc", "cpi_jobs"]:
    print(f"{f:11s}", X.groupby(f)[["straddle", "fly1"]].mean().round(1).to_dict("index"))

# ---------- 1b. walk-forward: model trained only on earlier months decides each month ----------
months = sorted(X["date"].str[:7].unique())
oos = [m for m in months if m >= OOS_FROM]
out = []
for m in oos:
    train, test = X[X["date"].str[:7] < m], X[X["date"].str[:7] == m]
    for target in ["straddle", "fly1"]:
        mdl = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.03, max_depth=3, num_leaves=7, min_child_samples=40,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
        mdl.fit(train[FEATS], train[target].clip(train[target].quantile(0.02), train[target].quantile(0.98)))
        pred = mdl.predict(test[FEATS])
        # simple rule, also chosen on earlier months only: trade when ratio_vix is above its q-quantile
        best_q, best = None, -np.inf
        for q in (0.0, 0.2, 0.4, 0.6):
            thr = train["ratio_vix"].quantile(q)
            v = train.loc[train["ratio_vix"] >= thr, target].mean()
            if v > best:
                best, best_q = v, q
        thr = train["ratio_vix"].quantile(best_q)
        out.append(test.assign(target=target, pnl=test[target], model_says=pred > 0, rule_says=test["ratio_vix"] >= thr))
o = pd.concat(out)
print(f"\n===== 1b. WALK-FORWARD {OOS_FROM} .. {X['date'].max()}: each month decided by earlier months only =====")
res = []
for target in ["straddle", "fly1"]:
    g = o[o["target"] == target]
    for name, mask in [("every day (no filter)", g["pnl"].notna()), ("model says trade", g["model_says"]),
                       ("ratio_vix rule says trade", g["rule_says"])]:
        res.append({"trade": target, "filter": name, **stats(g.loc[mask, "pnl"], g.loc[mask, "date"])})
print(pd.DataFrame(res).to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
for target in ["straddle", "fly1"]:
    g = o[(o["target"] == target) & o["model_says"]]
    print(f"  {target}, model-filtered, by year:", g.groupby(g["date"].str[:4])["pnl"].agg(["count", "mean"]).round(1).to_dict("index"))

# ---------- 2. pinning: do closes land near round strikes more than chance? ----------
print("\n===== 2. PINNING: share of prices within $0.50 of a $5 strike level (chance = 20%) =====")
pin = []
for ticker in ["SPY", "QQQ"]:
    c = closes[ticker].dropna()
    c = c[(c.index >= X["date"].min()) & (c.index <= X["date"].max())]
    for label, series in [("official close (4pm)", c),
                          ("3:00pm price", spots[(spots["ticker"] == ticker) & (spots["hhmm"] == "15:00")].set_index("date")["spot"]),
                          ("12:00pm price", spots[(spots["ticker"] == ticker) & (spots["hhmm"] == "12:00")].set_index("date")["spot"])]:
        for grid in (5, 10):
            d = np.abs(series - np.round(series / grid) * grid)
            pin.append({"ticker": ticker, "when": label, "grid $": grid, "days": len(d),
                        "% within $0.50": (d <= 0.5).mean() * 100, "chance %": 100 / grid,
                        "avg distance $": d.mean(), "chance avg $": grid / 4})
print(pd.DataFrame(pin).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
# conditional: within $1 of a $5 level at 3pm -> closer at the close? Control: the same for the in-between
# levels ($2.50 off the grid), where nothing should pull.
for ticker in ["SPY", "QQQ"]:
    s3 = spots[(spots["ticker"] == ticker) & (spots["hhmm"] == "15:00")].set_index("date")["spot"]
    c = closes[ticker].reindex(s3.index)
    for name, off in [("$5 levels", 0.0), ("in-between levels (control)", 2.5)]:
        lvl = np.round((s3 - off) / 5) * 5 + off
        near = (s3 - lvl).abs() <= 1.0
        d3, dc = (s3 - lvl).abs()[near], (c - lvl).abs()[near]
        print(f"{ticker} {name:28s}: {int(near.sum())} days within $1 at 3pm; distance 3pm ${d3.mean():.2f} -> close ${dc.mean():.2f}; "
              f"closer at the close on {(dc < d3).mean() * 100:.0f}% of them")
