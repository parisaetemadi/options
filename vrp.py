"""
Volatility risk premium (VRP): do options price in more movement than actually happens --
and can we tell WHEN they're too expensive (or too cheap)?

1. Each day, for each ticker: the 30-day at-the-money implied vol (what options expect) vs the
   volatility that actually happened over the next 21 trading days.
2. Forecast that realized vol with a small model retrained every year on past data only
   (inputs: recent realized vols, the implied vol itself, VIX, upcoming earnings/Fed/CPI/jobs).
   signal = log(implied vol) - log(forecast)   (> 0: options look expensive)
3. Trade the two options closest to the stock price (the OTM put just below and the OTM call
   just above; 2-week, monthly or quarterly expiries; held 1 to 20 days): SELL both when the signal is high, BUY both when it's low.
   Delta-hedged daily, real bid/ask both ways, trades crossing a stock split skipped.
4. Rules are chosen on 2016-2021 and checked ONCE on 2022-today.

Run:  python vrp.py          (needs models/dataset_core.parquet from train.py)
"""
import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd

import strategy_search as ss
import train as tr

H_RV = 21                        # realized vol measured over the next 21 trading days
SELECT_YEARS = range(2016, 2022)
GRID = {"side": ["sell", "buy"], "threshold": [0.0, 0.1, 0.2], "hold": [1, 2, 3, 5, 10, 20],
        "expiry": ["2 weeks", "month", "quarter"]}
# which options to trade: (min, max, target) days to expiry
EXPIRY = {"2 weeks": (12, 18, 14), "month": (20, 45, 30), "quarter": (60, 100, 75)}

# ---------- 1. daily table per ticker: implied vs realized ----------
cols = ["date", "ticker", "type", "strike", "expiration", "spot", "bid", "ask", "mid", "T", "tbill", "div_yield",
        "iv", "atm_iv", "atm30", "atm_used", "rv5", "rv21", "rv63", "vix_prev", "days_to_event",
        "days_to_fomc", "days_to_cpi", "days_to_jobs"]
q = pd.read_parquet(tr.Path(tr.__file__).parent / "models" / "dataset_core.parquet", columns=cols)
q = q.astype({"date": str, "ticker": str, "type": str, "expiration": str})
q = q[q["date"] >= "2014-01-01"]
day = q.drop_duplicates(["date", "ticker"])[["date", "ticker", "atm30", "rv5", "rv21", "rv63", "vix_prev",
                                             "days_to_event", "days_to_fomc", "days_to_cpi", "days_to_jobs"]]

# realized vol over the NEXT 21 trading days (split-corrected returns)
s = tr.stock_features()[["ticker", "date"]].merge(pd.read_csv(tr.DATA / "databento" / "underlying.csv"), on=["ticker", "date"])
s = s.sort_values(["ticker", "date"])
s["ret"] = s.groupby("ticker")["close"].transform(lambda c: np.log(c / c.shift(1)))
sp = pd.read_csv(tr.DATA / "events" / "splits.csv")
s = s.merge(sp, on=["ticker", "date"], how="left")
s["ret"] += np.log(s["ratio"].fillna(1.0))
s["rv_fwd"] = s.groupby("ticker")["ret"].transform(
    lambda r: np.sqrt((r**2).rolling(H_RV).sum().shift(-H_RV) * 252 / H_RV))
day = day.merge(s[["ticker", "date", "rv_fwd"]], on=["ticker", "date"]).dropna(subset=["atm30", "rv21", "rv63"])
day["year"] = day["date"].str[:4].astype(int)

print("How often options priced MORE movement than actually happened (30-day ATM IV > next 21 days' realized):")
known = day.dropna(subset=["rv_fwd"])
print((known.assign(rich=known["atm30"] > known["rv_fwd"]).groupby("ticker")["rich"].mean() * 100).round(0).to_dict())
print("Average IV minus realized (vol points):",
      ((known["atm30"] - known["rv_fwd"]).groupby(known["ticker"]).mean() * 100).round(1).to_dict())

# ---------- 2. forecast realized vol, year by year (past data only) ----------
F = ["rv5", "rv21", "rv63", "atm30", "vix_prev", "days_to_event", "days_to_fomc", "days_to_cpi", "days_to_jobs"]
day["log_fc"] = np.nan
for y in range(2016, day["year"].max() + 1):
    # train only on days whose 21-day outcome was already known before year y started
    past = known[known["date"] < f"{y}-01-01"]
    past = past[past["date"] < sorted(past["date"].unique())[-H_RV]]
    X = lambda d: np.column_stack([np.log(d[["rv5", "rv21", "rv63", "atm30"]].clip(lower=0.01)), d[F[4:]],
                                   pd.get_dummies(d["ticker"]).reindex(columns=sorted(day["ticker"].unique()), fill_value=0)])
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                          verbose=-1, random_state=0).fit(X(past), np.log(past["rv_fwd"]))
    cur = day["year"] == y
    day.loc[cur, "log_fc"] = m.predict(X(day[cur]))
day["signal"] = np.log(day["atm30"]) - day["log_fc"]
chk = day.dropna(subset=["log_fc", "rv_fwd"])
err_model = np.median(np.abs(np.exp(chk["log_fc"]) - chk["rv_fwd"]))
err_iv = np.median(np.abs(chk["atm30"] - chk["rv_fwd"]))
err_rv = np.median(np.abs(chk["rv21"] - chk["rv_fwd"]))
print(f"\nForecast of next-21-day realized vol, 2016-today (median error, vol points): model {err_model * 100:.1f}, "
      f"implied vol itself {err_iv * 100:.1f}, last 21 days' realized {err_rv * 100:.1f}")

# ---------- 3. trade the near-the-money pair, for several expiries ----------
atm = q[q["atm_used"]].copy()
atm["dte"] = (pd.to_datetime(atm["expiration"]) - pd.to_datetime(atm["date"])).dt.days
leg_sets = {}
for name, (lo, hi, target) in EXPIRY.items():
    legs = atm[atm["dte"].between(lo, hi)].copy()
    # one expiry per ticker and day: the one closest to the target
    legs["dist"] = (legs["dte"] - target).abs()
    legs = legs[legs["dist"] == legs.groupby(["date", "ticker"])["dist"].transform("min")]
    legs = legs.merge(day[["date", "ticker", "signal"]], on=["date", "ticker"]).dropna(subset=["signal"])
    legs["delta"] = ss.delta(legs)
    legs["day_no"] = legs["date"].map(ss.day_no)
    leg_sets[name] = legs.dropna(subset=["day_no"]).astype({"day_no": int})
del atm

configs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
trades = {i: [] for i in range(len(configs))}
years = sorted(leg_sets["month"]["date"].str[:4].unique())
for year in years:
    days = pd.concat([l.loc[l["date"].str[:4] == year, "date"] for l in leg_sets.values()])
    first, last = ss.day_no[days.min()], min(ss.day_no[days.max()] + max(GRID["hold"]), len(ss.all_days) - 1)
    panel = ss.panel_for(ss.all_days[first:last + 1])
    for i, c in enumerate(configs):
        ly = leg_sets[c["expiry"]]
        ly = ly[ly["date"].str[:4] == year]
        sel = ly[ly["signal"] > c["threshold"]] if c["side"] == "sell" else ly[ly["signal"] < -c["threshold"]]
        t = ss.trade_pnl(sel, c["hold"], panel, c["side"])
        # both legs together = one trade per ticker per day
        trades[i].append(t.groupby(["date", "ticker"]).agg(pnl=("pnl", "sum"), dropped=("dropped", "max")).reset_index())
    print(f"{year} done", flush=True)
    del panel

rows = []
for i, c in enumerate(configs):
    t = pd.concat(trades[i], ignore_index=True)
    sel_t = t[t["date"].str[:4].astype(int).isin(SELECT_YEARS)]
    rows.append({**c, **{f"select {k}": v for k, v in ss.stats(sel_t, c["hold"]).items()}, "_t": t})
res = pd.DataFrame(rows)
show = res.drop(columns="_t").sort_values("select t (overlap-corrected)", ascending=False)
print("\nRules on the SELECTION years 2016-2021 (one trade = the put+call pair, $ per pair):")
print(f"{len(configs)} rules tried -- with this many, a selection t below ~2.5 is likely luck.")
print(show.head(20).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

best = res.loc[show.index[0]]
t = best["_t"]
conf = t[~t["date"].str[:4].astype(int).isin(SELECT_YEARS)]
s2 = ss.stats(conf, best["hold"])
print(f"\nChosen: {best['side']} {best['expiry']} options when signal {'>' if best['side'] == 'sell' else '<'} "
      f"{'' if best['side'] == 'sell' else '-'}{best['threshold']}, hold {best['hold']} days")
print(f"CONFIRMATION 2022-today (checked once): {s2['trades']:,} trades, ${s2.get('$/trade', np.nan):,.2f}/trade, "
      f"{s2.get('% days +', np.nan):.0%} of days profitable, t = {s2.get('t (overlap-corrected)', np.nan):.1f}, "
      f"dropped {s2.get('dropped', np.nan):.0%}")
print("By year:", t.groupby(t["date"].str[:4])["pnl"].mean().round(1).to_dict())
print("By ticker (confirmation):", conf.groupby("ticker")["pnl"].mean().round(1).to_dict())
