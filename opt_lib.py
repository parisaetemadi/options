"""
Shared pieces for the option-strategy tests (option_strategies.py, vol_longshort.py):
the near-the-money option pair per ticker and day, the "options look expensive" signal, and
trade simulators that pay real bid/ask (fill = share of the half-spread paid; 1.0 = full).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm

import strategy_search as ss
import train as tr

MODELS = tr.Path(tr.__file__).parent / "models"
STOCK_COST = 0.0002
all_days = ss.all_days

LEG_COLS = ["date", "ticker", "type", "strike", "expiration", "spot", "bid", "ask", "mid", "T", "tbill", "div_yield",
            "iv", "atm30", "rv5", "rv21", "rv63", "vix_prev", "days_to_event", "days_to_fomc", "days_to_cpi", "days_to_jobs"]


def load_legs(groups, target_dte=30, lo=20, hi=45):
    """The OTM put just below and OTM call just above the stock, the expiry closest to target_dte."""
    q = pd.concat([pd.read_parquet(MODELS / f"dataset_{g}.parquet", columns=LEG_COLS, filters=[("atm_used", "==", True)])
                   .astype({"date": str, "ticker": str, "type": str, "expiration": str})
                   for g in groups if (MODELS / f"dataset_{g}.parquet").exists()], ignore_index=True)
    q = q.astype({c: "float64" for c in q.columns if q[c].dtype == "float32"})
    legs = q[(q["T"] * 365).between(lo, hi)].copy()
    legs["dist"] = (legs["T"] * 365 - target_dte).abs()
    legs = legs[legs["dist"] == legs.groupby(["date", "ticker"])["dist"].transform("min")]
    legs = legs[legs.groupby(["date", "ticker"])["type"].transform("nunique") == 2]          # both legs present
    legs["delta"] = ss.delta(legs)
    legs["vega"] = vega(legs)
    legs["day_no"] = legs["date"].map(ss.day_no)
    legs = legs.dropna(subset=["day_no"]).astype({"day_no": int}).reset_index(drop=True)
    legs["half"] = (legs["ask"] - legs["bid"]) / 2
    return legs


def vega(d):
    """$ change in one option's price (per share) for a 1 vol-point move in implied vol."""
    T, s = d["T"], d["iv"]
    d1 = (np.log(d["spot"] / d["strike"]) + (d["tbill"] - d["div_yield"] + 0.5 * s**2) * T) / (s * np.sqrt(T))
    return d["spot"] * np.exp(-d["div_yield"] * T) * norm.pdf(d1) * np.sqrt(T) / 100


def vrp_signal(legs):
    """log(30-day implied vol) - log(forecast of the next 21 days' real vol); the forecast for each
    year is trained only on earlier years (as in vrp.py). > 0: options look expensive."""
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
        past = past[past["date"] < sorted(past["date"].unique())[-21]]           # outcome known before year y
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                              verbose=-1, random_state=0).fit(X(past), np.log(past["rv_fwd"]))
        cur = day["date"].str[:4] == str(y)
        day.loc[cur, "log_fc"] = m.predict(X(day[cur]))
    day["signal"] = np.log(day["atm30"]) - day["log_fc"]
    return day[["date", "ticker", "signal"]].dropna()


def hold_to_expiry(entry, hedge, fill, side="sell"):
    """Sell (or buy) each option and keep it to expiry: the spread is paid once. Optional daily delta
    hedge with the stock (STOCK_COST per trade). Settled at the stock's close on the last trading day
    on or before expiry. Trades whose life includes a stock split are skipped. $ per contract."""
    e = entry.copy()
    e["exp_no"] = np.searchsorted(all_days, e["expiration"].values, side="right") - 1
    e = e[(e["exp_no"] > e["day_no"]) & (e["expiration"] <= all_days[-1])]
    steps = (e["exp_no"] - e["day_no"]).values
    e = e[~ss.crosses_split(e, steps)]
    steps = (e["exp_no"] - e["day_no"]).values
    n, kmax = len(e), int(steps.max()) if len(e) else 0
    S_prev, hedge_pnl, cost = e["spot"].values.copy(), np.zeros(n), np.zeros(n)
    d_prev = e["delta"].values.copy()                          # shares held against a SHORT option
    cost += np.abs(d_prev) * S_prev * STOCK_COST
    for k in range(1, kmax + 1):
        act = k <= steps
        day_k = all_days[np.minimum(e["day_no"].values + k, len(all_days) - 1)]
        S_k = np.where(act, ss.closes.reindex(list(zip(e["ticker"], day_k))).values, S_prev)
        hedge_pnl += np.where(act, d_prev * (S_k - S_prev), 0)
        T_k = np.maximum(e["T"].values - (pd.to_datetime(day_k) - pd.to_datetime(e["date"].values)).days.values / 365, 1e-6)
        d_k = np.nan_to_num(ss.delta(e.assign(spot=S_k, T=T_k)))
        d_new = np.where(k == steps, 0.0, np.where(act, d_k, d_prev))           # expiry: close the hedge
        cost += np.where(act, np.abs(d_new - d_prev) * S_k * STOCK_COST, 0)
        d_prev, S_prev = d_new, S_k
    payoff = np.where(e["type"].values == "call", np.maximum(S_prev - e["strike"].values, 0),
                      np.maximum(e["strike"].values - S_prev, 0))
    if side == "sell":
        pnl = (e["mid"].values - fill * e["half"].values) - payoff + (hedge_pnl - cost if hedge else 0)
    else:                                                       # long option, hedge is the mirror image
        pnl = payoff - (e["mid"].values + fill * e["half"].values) + (-hedge_pnl - cost if hedge else 0)
    return e.assign(pnl=pnl * 100 - ss.COMMISSION, premium=e["mid"].values * 100)  # one commission: held to expiry
