"""
Option strategies designed around the cost that sank the earlier tests (paying the full bid/ask twice).
All trades are in OPTIONS. 14 tickers, 2014-today. Every rule is chosen on the earlier years and
checked ONCE on 2022-today, at full bid/ask (fill 1.0). The same chosen rule is also shown at
fill 0.5 (paying half the spread: what patient limit orders on liquid options often get) --
an execution assumption, never used to pick rules.

B. DIRECTIONAL: the stock-signal model (stock_signals.py; forecasts made only from earlier years)
   ranks the tickers each day. Option on the k best / k worst, ~30 days to expiry, closest to the money:
     buy:  calls on the best, puts on the worst       sell: puts on the best, calls on the worst
   Held 5 or 10 days, not hedged.
C. SELL AND HOLD TO EXPIRY: sell the near-the-money put+call (~30 days) when options look expensive
   (vrp.py's signal: implied vol vs a forecast of real movement, from earlier years only), keep to
   expiry -> the spread is paid once. With or without a daily stock hedge (0.02% per hedge trade).
   Settled at the stock's closing price on expiry day.

Run:  python option_strategies.py
"""
import itertools

import lightgbm as lgb
import numpy as np
import pandas as pd

import strategy_search as ss
import train as tr
from pricing import bs_price

MODELS = tr.Path(tr.__file__).parent / "models"
GROUPS = ("core", "extra")
STOCK_COST = 0.0002
all_days, KEY = ss.all_days, ss.KEY

# ---------- the near-the-money put and call, ~30 days out, per ticker and day ----------
cols = ["date", "ticker", "type", "strike", "expiration", "spot", "bid", "ask", "mid", "T", "tbill", "div_yield", "iv",
        "atm30", "rv5", "rv21", "rv63", "vix_prev", "days_to_event", "days_to_fomc", "days_to_cpi", "days_to_jobs"]
q = pd.concat([pd.read_parquet(MODELS / f"dataset_{g}.parquet", columns=cols, filters=[("atm_used", "==", True)])
               .astype({"date": str, "ticker": str, "type": str, "expiration": str}) for g in GROUPS], ignore_index=True)
q = q.astype({c: "float64" for c in q.columns if q[c].dtype == "float32"})
dte = q["T"] * 365
legs = q[dte.between(20, 45)].copy()
legs["dist"] = (legs["T"] * 365 - 30).abs()
legs = legs[legs["dist"] == legs.groupby(["date", "ticker"])["dist"].transform("min")]
legs = legs[legs.groupby(["date", "ticker"])["type"].transform("nunique") == 2]          # both legs present
legs["delta"] = ss.delta(legs)
legs["day_no"] = legs["date"].map(ss.day_no)
legs = legs.dropna(subset=["day_no"]).astype({"day_no": int}).reset_index(drop=True)
legs["half"] = (legs["ask"] - legs["bid"]) / 2
print(f"{len(legs):,} near-the-money options, {legs['date'].min()} to {legs['date'].max()}; "
      f"median spread {(2 * legs['half'] / legs['mid']).median():.1%} of the price")


def stats(t, lags):
    if len(t) < 10:
        return {"n": len(t)}
    daily = t.groupby("date")["pnl"].mean()
    return {"n": len(t), "$/trade": t["pnl"].mean(), "% of premium": t["pnl"].sum() / t["premium"].sum() * 100,
            "% won": (t["pnl"] > 0).mean() * 100, "t": ss.nw_tstat(daily.values, lags)}


def show(name, rows, select_years, lags):
    """Pick the best rule on the selection years (fill 1.0), confirm it once on 2022+ at fill 1.0 and 0.5."""
    res = pd.DataFrame([{**r["rule"], **{f"sel {k}": v for k, v in stats(r["full"][r["full"]["year"].isin(select_years)], lags).items()}}
                        for r in rows])
    print(f"\n===== {name}: rules on {select_years[0]}-{select_years[-1]} (full bid/ask) =====")
    print(res.sort_values("sel t", ascending=False).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    best = rows[int(res["sel t"].idxmax())]
    print(f"Chosen: {best['rule']}")
    for fill, t in [("1.0 (full bid/ask)", best["full"]), ("0.5 (half the spread)", best["half"])]:
        c = stats(t[~t["year"].isin(select_years) & (t["year"] >= 2022)], lags)
        print(f"  CONFIRMATION 2022-today, fill {fill}: {c['n']:,} trades, ${c.get('$/trade', np.nan):,.1f}/trade, "
              f"{c.get('% of premium', np.nan):+.1f}% of premium, {c.get('% won', np.nan):.0f}% won, t {c.get('t', np.nan):.1f}")
    t = best["full"]
    print("  by year (fill 1.0, $/trade):", t.groupby("year")["pnl"].mean().round(0).to_dict())


# ---------- B. directional option trades on the stock-signal ranking ----------
def hold_unhedged(entry, h, panel, side, fill):
    entry = entry[(entry["day_no"] + h < len(all_days)) & (entry["T"] * 365 >= 7 + np.ceil(h * 7 / 5) + 3)]
    entry = entry[~ss.crosses_split(entry, h)]
    if entry.empty:
        return entry.assign(pnl=[], premium=[], dropped=[])
    nd = panel.reindex(pd.MultiIndex.from_frame(entry[KEY].assign(day_no=entry["day_no"] + h)))
    ok = nd["mid"].notna().values
    exit_day = all_days[entry["day_no"].values + h]
    spot = np.where(ok, nd["spot"].values, ss.closes.reindex(list(zip(entry["ticker"], exit_day))).values)
    elapsed = (pd.to_datetime(exit_day) - pd.to_datetime(entry["date"].values)).days.values / 365
    value = bs_price(entry["type"].values, spot, entry["strike"].values, np.maximum(entry["T"].values - elapsed, 1 / 365),
                     entry["tbill"].values, entry["iv"].values, entry["div_yield"].values)
    xmid = np.where(ok, nd["mid"].values, value)
    xhalf = np.where(ok, (nd["ask"] - nd["bid"]).values / 2, 2 * entry["half"].values)    # lost from data: extra spread
    emid, ehalf = entry["mid"].values, entry["half"].values
    pnl = (xmid - fill * xhalf) - (emid + fill * ehalf) if side == "buy" else (emid - fill * ehalf) - (xmid + fill * xhalf)
    return entry.assign(pnl=pnl * 100, premium=emid * 100, dropped=~ok)


fc = pd.read_parquet(MODELS / "stock_forecasts_core_extra_notk.parquet")[["date", "ticker", "fc_next5"]].dropna()
fc["rank"] = fc.groupby("date")["fc_next5"].rank(ascending=False, method="first")
fc["n"] = fc.groupby("date")["fc_next5"].transform("size")
B = {"side": ["buy", "sell"], "k": [1, 3], "hold": [5, 10]}
rulesB = [dict(zip(B, v)) for v in itertools.product(*B.values())]
outB = {i: {"full": [], "half": []} for i in range(len(rulesB))}
legsB = legs.merge(fc, on=["date", "ticker"])
for year in sorted(legsB["date"].str[:4].unique()):
    ly = legsB[legsB["date"].str[:4] == year]
    first, last = ss.day_no[ly["date"].min()], min(ss.day_no[ly["date"].max()] + max(B["hold"]), len(all_days) - 1)
    panel = ss.panel_for(all_days[first:last + 1])
    for i, r in enumerate(rulesB):
        best, worst = ly["rank"] <= r["k"], ly["rank"] > ly["n"] - r["k"]
        if r["side"] == "buy":      # calls on the best, puts on the worst
            pick = ly[(best & (ly["type"] == "call")) | (worst & (ly["type"] == "put"))]
        else:                       # sell puts on the best, calls on the worst
            pick = ly[(best & (ly["type"] == "put")) | (worst & (ly["type"] == "call"))]
        for name, f in [("full", 1.0), ("half", 0.5)]:
            outB[i][name].append(hold_unhedged(pick, r["hold"], panel, r["side"], f)[["date", "ticker", "pnl", "premium"]])
    print(f"B {year} done", flush=True)
    del panel
rowsB = []
for i, r in enumerate(rulesB):
    full, half = (pd.concat(outB[i][k], ignore_index=True) for k in ("full", "half"))
    rowsB.append({"rule": r, "full": full.assign(year=full["date"].str[:4].astype(int)),
                  "half": half.assign(year=half["date"].str[:4].astype(int))})
show("B. directional options on the stock-signal ranking", rowsB, list(range(2014, 2022)), 10)


# ---------- C. sell near-the-money options when they look expensive, hold to expiry ----------
# the forecast of real movement over the next 21 trading days (as in vrp.py), year by year
day = legs.drop_duplicates(["date", "ticker"])[["date", "ticker", "atm30", "rv5", "rv21", "rv63", "vix_prev",
                                                "days_to_event", "days_to_fomc", "days_to_cpi", "days_to_jobs"]].copy()
s = pd.read_csv(tr.DATA / "databento" / "underlying.csv").sort_values(["ticker", "date"])
s["ret"] = s.groupby("ticker")["close"].transform(lambda c: np.log(c / c.shift(1)))
s = s.merge(pd.read_csv(tr.DATA / "events" / "splits.csv"), on=["ticker", "date"], how="left")
s["ret"] += np.log(s["ratio"].fillna(1.0))
s["rv_fwd"] = s.groupby("ticker")["ret"].transform(lambda r: np.sqrt((r**2).rolling(21).sum().shift(-21) * 252 / 21))
day = day.merge(s[["ticker", "date", "rv_fwd"]], on=["ticker", "date"], how="left")
F = ["rv5", "rv21", "rv63", "atm30", "vix_prev", "days_to_event", "days_to_fomc", "days_to_cpi", "days_to_jobs"]
tickers = sorted(day["ticker"].unique())
X = lambda d: np.column_stack([np.log(d[F[:4]].clip(lower=0.01)), d[F[4:]],
                               pd.get_dummies(d["ticker"]).reindex(columns=tickers, fill_value=0)])
day["log_fc"] = np.nan
known = day.dropna(subset=["rv_fwd"])
for y in range(2016, int(day["date"].max()[:4]) + 1):
    past = known[known["date"] < f"{y}-01-01"]
    past = past[past["date"] < sorted(past["date"].unique())[-21]]
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                          verbose=-1, random_state=0).fit(X(past), np.log(past["rv_fwd"]))
    cur = day["date"].str[:4] == str(y)
    day.loc[cur, "log_fc"] = m.predict(X(day[cur]))
day["signal"] = np.log(day["atm30"]) - day["log_fc"]
legsC = legs.merge(day[["date", "ticker", "signal"]], on=["date", "ticker"]).dropna(subset=["signal"])


def hold_to_expiry(entry, hedge, fill):
    e = entry.copy()
    e["exp_no"] = np.searchsorted(all_days, e["expiration"].values, side="right") - 1    # last trading day <= expiry
    e = e[(e["exp_no"] > e["day_no"]) & (e["expiration"] <= all_days[-1])]
    steps = (e["exp_no"] - e["day_no"]).values
    e = e[~ss.crosses_split(e, steps)]
    steps = (e["exp_no"] - e["day_no"]).values
    n, kmax = len(e), int(steps.max()) if len(e) else 0
    S_prev, hedge_pnl, cost = e["spot"].values.copy(), np.zeros(n), np.zeros(n)
    d_prev = e["delta"].values.copy()                         # hedge = +delta shares against the short option
    cost += np.abs(d_prev) * S_prev * STOCK_COST
    for k in range(1, kmax + 1):
        act = k <= steps
        day_k = all_days[np.minimum(e["day_no"].values + k, len(all_days) - 1)]
        S_k = np.where(act, ss.closes.reindex(list(zip(e["ticker"], day_k))).values, S_prev)
        hedge_pnl += np.where(act, d_prev * (S_k - S_prev), 0)
        T_k = np.maximum(e["T"].values - (pd.to_datetime(day_k) - pd.to_datetime(e["date"].values)).days.values / 365, 1e-6)
        d_k = np.nan_to_num(ss.delta(e.assign(spot=S_k, T=T_k)))
        last = k == steps                                         # expiry: close the hedge
        d_new = np.where(last, 0.0, np.where(act, d_k, d_prev))
        cost += np.where(act, np.abs(d_new - d_prev) * S_k * STOCK_COST, 0)
        d_prev, S_prev = d_new, S_k
    payoff = np.where(e["type"].values == "call", np.maximum(S_prev - e["strike"].values, 0),
                      np.maximum(e["strike"].values - S_prev, 0))
    pnl = (e["mid"].values - fill * e["half"].values) - payoff + (hedge_pnl - cost if hedge else 0)
    return e.assign(pnl=pnl * 100, premium=e["mid"].values * 100)


C = {"threshold": [None, 0.0, 0.1, 0.2], "hedge": [True, False]}
rulesC = [dict(zip(C, v)) for v in itertools.product(*C.values())]
rowsC = []
for r in rulesC:
    pick = legsC if r["threshold"] is None else legsC[legsC["signal"] > r["threshold"]]
    res = {}
    for name, f in [("full", 1.0), ("half", 0.5)]:
        t = hold_to_expiry(pick, r["hedge"], f)
        t = t.groupby(["date", "ticker"]).agg(pnl=("pnl", "sum"), premium=("premium", "sum")).reset_index()  # the pair
        res[name] = t.assign(year=t["date"].str[:4].astype(int))
    rowsC.append({"rule": r, **res})
    print(f"C {r} done", flush=True)
show("C. sell the near-the-money pair, hold to expiry", rowsC, list(range(2016, 2022)), 21)
