"""
Do option prices predict the STOCK's next move? Trading the stock itself costs ~0.01-0.02%
(vs 1-10% for options), so even a small signal could survive costs.

Signals per ticker and day, from the closing option quotes (expiry ~30 days out):
  put_skew    IV of puts ~1 standard deviation below the price / ATM IV - 1   (crash-protection demand)
  call_skew   IV of calls ~1 standard deviation above / ATM IV - 1           (upside demand)
  vrp         30-day ATM IV - last 21 days' realized vol
  term        ATM IV ~60-90 days out / ~30 days out - 1                      (< 0: short-term stress)
  d_put_skew, d_atm   5-day changes in put skew and in log ATM IV
Target: the stock's return over the next 1 or 5 days, split-corrected.

A small LightGBM model combines the signals, retrained every year on earlier years only.
Portfolio: each day, go long the tickers with the highest forecast and short the lowest (positions
sum to zero, so a rising market alone doesn't make money), hold h days, pay 0.02% per unit traded.
Rules are picked on 2014-2021 and checked once on 2022-today.

Run:  python stock_signals.py [--no-tickers] [--all]   (--all: all 14 tickers; needs models/dataset_extra.parquet)
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

import strategy_search as ss
import train as tr

SELECT = range(2014, 2022)
COST = 0.0002                 # per unit of stock traded (bid/ask + fees, conservative for these names)
SIG = ["put_skew", "call_skew", "vrp", "term", "d_put_skew", "d_atm"]

# ---------- signals from the option smile ----------
GROUPS = ["core", "extra"] if "--all" in sys.argv else ["core"]     # --all: all 14 tickers
q = pd.concat([pd.read_parquet(tr.Path(tr.__file__).parent / "models" / f"dataset_{g}.parquet",
                               columns=["date", "ticker", "type", "expiration", "T", "x", "iv", "atm_iv", "atm30", "rv21"])
               .astype({"date": str, "ticker": str, "type": str, "expiration": str}) for g in GROUPS], ignore_index=True)
sig = tr.smile_signals(q)
del q
sig = sig.sort_values(["ticker", "date"])
g = sig.groupby("ticker")
sig["d_put_skew"] = sig["put_skew"] - g["put_skew"].shift(5)
sig["d_atm"] = np.log(sig["atm"] / g["atm"].shift(5))

# ---------- what the stock did next ----------
s = pd.read_csv(tr.DATA / "databento" / "underlying.csv").sort_values(["ticker", "date"])
s["ret"] = s.groupby("ticker")["close"].transform(lambda c: np.log(c / c.shift(1)))
s = s.merge(pd.read_csv(tr.DATA / "events" / "splits.csv"), on=["ticker", "date"], how="left")
s["ret"] = np.exp(s["ret"] + np.log(s["ratio"].fillna(1.0))) - 1
s["next1"] = s.groupby("ticker")["ret"].shift(-1)
s["next5"] = s.groupby("ticker")["ret"].transform(lambda r: (1 + r).rolling(5).apply(np.prod, raw=True).shift(-5) - 1)
d = sig.merge(s[["ticker", "date", "next1", "next5"]], on=["ticker", "date"]).dropna(subset=SIG + ["next1"])
d = d[d["date"] >= "2013-06-01"].reset_index(drop=True)
d["year"] = d["date"].str[:4].astype(int)
tickers = sorted(d["ticker"].unique())
print(f"{len(d):,} ticker-days, {d['date'].min()} to {d['date'].max()}")

# each signal alone: does it line up with the next day's return? (rank correlation across tickers, per day)
print("\nSingle signals: average daily rank correlation with the next-day return (t, overlap-free):")
for c in SIG:
    ic = d.groupby("date").apply(lambda x: x[c].rank().corr(x["next1"].rank()))
    for name, part in [("2014-21", ic[ic.index < "2022"]), ("2022+", ic[ic.index >= "2022"])]:
        print(f"  {c:11s} {name}: {part.mean():+.4f} (t {ss.nw_tstat(part.dropna().values, 1):+.1f})", end="")
    print()

# ---------- combined forecast, retrained every year on earlier years ----------
NO_TICKERS = "--no-tickers" in sys.argv     # signals only: the model can't learn "always long NVDA"

def X(x):
    if NO_TICKERS:
        return x[SIG].values
    return np.column_stack([x[SIG], pd.get_dummies(x["ticker"]).reindex(columns=tickers, fill_value=0)])

for target in ["next1", "next5"]:
    h = int(target[-1])
    d[f"fc_{target}"] = np.nan
    for y in range(2014, d["year"].max() + 1):
        past = d[(d["date"] < f"{y}-01-01")].dropna(subset=[target])
        past = past[past["date"] < sorted(past["date"].unique())[-h]]            # outcome known before year y
        # demeaned per day: learn which tickers beat the others, not where the market goes
        yv = past[target] - past.groupby("date")[target].transform("mean")
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.02, num_leaves=7, min_child_samples=200,
                              subsample=0.8, subsample_freq=1, verbose=-1, random_state=0).fit(X(past), yv)
        cur = d["year"] == y
        d.loc[cur, f"fc_{target}"] = m.predict(X(d[cur]))


# save the walk-forward forecasts (each year's from a model trained on earlier years only) for option_strategies.py
d[["date", "ticker", "fc_next1", "fc_next5"]].to_parquet(
    tr.Path(tr.__file__).parent / "models" / f"stock_forecasts_{'_'.join(GROUPS)}{'_notk' if NO_TICKERS else ''}.parquet")


def backtest(fc, h):
    """Dollar-neutral: weight = rank of forecast, centred, sums to |1|; held h days (1/h rebalanced daily)."""
    w = d.pivot(index="date", columns="ticker", values=fc)
    w = w.rank(axis=1).sub(w.rank(axis=1).mean(axis=1), axis=0)
    w = w.div(w.abs().sum(axis=1), axis=0).fillna(0)
    w = w.rolling(h, min_periods=1).mean()                                         # h overlapping books
    r = d.pivot(index="date", columns="ticker", values="next1").reindex_like(w).fillna(0)
    gross = (w * r).sum(axis=1)
    cost = w.diff().abs().sum(axis=1).fillna(0) * COST
    return gross, gross - cost


print("\nCombined model, dollar-neutral portfolio (daily returns, % per year, Sharpe, t):")
rows = []
for target in ["next1", "next5"]:
    for h in [1, 5, 10]:
        gross, net = backtest(f"fc_{target}", h)
        for name, mask in [("2014-21", net.index < "2022"), ("2022+", net.index >= "2022")]:
            n, gr = net[mask], gross[mask]
            rows.append({"forecast": target, "hold": h, "period": name, "gross %/yr": gr.mean() * 252 * 100,
                         "net %/yr": n.mean() * 252 * 100, "Sharpe": n.mean() / n.std() * np.sqrt(252),
                         "t": ss.nw_tstat(n.values, h)})
res = pd.DataFrame(rows)
print(res.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
sel = res[res["period"] == "2014-21"].sort_values("t", ascending=False).iloc[0]
conf = res[(res["period"] == "2022+") & (res["forecast"] == sel["forecast"]) & (res["hold"] == sel["hold"])].iloc[0]
print(f"\nChosen on 2014-21: forecast {sel['forecast']}, hold {sel['hold']} (t {sel['t']:.1f}) -> "
      f"CONFIRMATION 2022+: net {conf['net %/yr']:.1f}%/yr, Sharpe {conf['Sharpe']:.2f}, t {conf['t']:.1f}")

# ---------- robustness of the chosen rule ----------
fc, h = f"fc_{sel['forecast']}", int(sel["hold"])
w = d.pivot(index="date", columns="ticker", values=fc)
w = w.rank(axis=1).sub(w.rank(axis=1).mean(axis=1), axis=0)
w = w.div(w.abs().sum(axis=1), axis=0).fillna(0).rolling(h, min_periods=1).mean()
r = d.pivot(index="date", columns="ticker", values="next1").reindex_like(w).fillna(0)
late = w.index >= "2022"
print("\nAverage position per ticker 2022+ (+ long, - short):", w[late].mean().round(3).to_dict())
print("Profit per ticker 2022+ (%/yr):", ((w * r)[late].mean() * 252 * 100).round(1).to_dict())
by_year = (w * r).sum(axis=1) - w.diff().abs().sum(axis=1).fillna(0) * COST
print("Net %/yr by year:", (by_year.groupby(by_year.index.str[:4]).mean() * 252 * 100).round(1).to_dict())

# 1. trade one day LATER (signal at today's close, trade at tomorrow's close): no timing luck
lag = (w.shift(1).fillna(0) * r).sum(axis=1) - w.shift(1).diff().abs().sum(axis=1).fillna(0) * COST
# 2. remove each ticker's persistent tilt (only day-to-day changes in the bets count)
tilt = w - w.expanding().mean().shift(1).fillna(0)
tl = (tilt * r).sum(axis=1) - tilt.diff().abs().sum(axis=1).fillna(0) * COST
for name, x in [("trade 1 day later", lag), ("without persistent tilts", tl)]:
    for per, m in [("2014-21", x.index < "2022"), ("2022+", x.index >= "2022")]:
        print(f"  {name:25s} {per}: net {x[m].mean() * 252 * 100:+.1f}%/yr, Sharpe {x[m].mean() / x[m].std() * np.sqrt(252):.2f}, "
              f"t {ss.nw_tstat(x[m].values, h):.1f}")

# 3. is it just market exposure? (long volatile stocks vs short the index = net long the market)
spy = r["SPY"]
net = (w * r).sum(axis=1) - w.diff().abs().sum(axis=1).fillna(0) * COST
for per, m in [("2014-21", net.index < "2022"), ("2022+", net.index >= "2022")]:
    beta = np.cov(net[m], spy[m])[0, 1] / spy[m].var()
    alpha = net[m] - beta * spy[m]
    print(f"  market beta {per}: {beta:.2f}; return left after removing it: {alpha.mean() * 252 * 100:+.1f}%/yr, "
          f"t {ss.nw_tstat(alpha.values, h):.1f}")
