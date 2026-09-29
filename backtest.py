"""
Does the model find options that are really too cheap or too expensive?

Uses ONLY the test period (dates the model never saw while training), from train.py's
models/test_predictions.parquet.

Each day at the close, for every option (except the two used to measure the ATM level):
  gap = market IV - model IV
  gap very negative -> option looks CHEAP -> buy it at the ask
  gap very positive -> option looks EXPENSIVE -> sell it at the bid
Each trade is delta-hedged with the stock, re-balanced daily (so we bet on the option's price,
not the stock's direction), and closed 1, 5 or 10 trading days later: sell at the bid / buy
back at the ask. Rules compared: a big IV gap, or the model's price being outside the bid/ask.

Reported with and without trading costs:
  - "mid to mid": no costs, shows if the signal exists at all
  - "real (bid/ask)": what a trader would actually have made

Run:  python backtest.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from pricing import bs_price, implied_vol_many

PRED = Path(__file__).parent / "models" / "test_predictions.parquet"
HOLD_DAYS = [1, 5, 10]
MIN_ABS_LOG_M = 0.03                # skip options within 3% of the stock price (they often drop out of our data next day)


def delta(d):
    """Black-Scholes delta, using the market's own IV."""
    T, s = d["T"], d["iv"]
    d1 = (np.log(d["spot"] / d["strike"]) + (d["tbill"] - d["div_yield"] + 0.5 * s**2) * T) / (s * np.sqrt(T))
    disc = np.exp(-d["div_yield"] * T)
    return np.where(d["type"] == "call", disc * norm.cdf(d1), disc * (norm.cdf(d1) - 1))


p = pd.read_parquet(PRED)
p = p[~p["atm_used"]].copy()
p["gap"] = p["iv"] - p["iv_pred"]
p["delta"] = delta(p)
p["model_price"] = bs_price(p["type"].values, p["spot"], p["strike"], p["T"], p["tbill"].values,
                            p["iv_pred"].values, p["div_yield"].values)
p = p[np.abs(np.log(p["strike"] / p["spot"])) >= MIN_ABS_LOG_M]
key = ["ticker", "type", "strike", "expiration"]
days = np.sort(p["date"].unique())
day_no = {d: i for i, d in enumerate(days)}
p["day_no"] = p["date"].map(day_no)
print(f"Test period {days[0]} to {days[-1]} ({len(days)} days), {len(p):,} option-days.\n")

# every contract's daily path: price and stock, so the hedge can be re-balanced each day.
# We only downloaded out-of-the-money options, so when an option moves IN the money it would
# vanish from our data -- and dropping those trades would hide the losers. Put-call parity
# rebuilds its price exactly from the out-of-the-money option at the same strike:
#     call - put = stock * e^(-qT) - strike * e^(-rT)
DATA = Path(__file__).parent / "data"
test_files = [f for f in sorted((DATA / "databento" / "quotes").glob("*/*.parquet")) if days[0] <= f.stem <= days[-1]]
full = pd.concat([pd.read_parquet(f) for f in test_files], ignore_index=True)   # ALL quotes, incl. zero bids
full = full.astype({"date": str, "ticker": str, "type": str, "expiration": str})
full = full[full["date"].isin(day_no)]
full = full.merge(pd.read_csv(DATA / "market" / "vix_rates.csv")[["date", "tbill"]], on="date")
full = full.merge(pd.read_csv(DATA / "events" / "div_yield.csv"), on=["ticker", "date"], how="left").fillna({"div_yield": 0})
full["T"] = (pd.to_datetime(full["expiration"]) - pd.to_datetime(full["date"])).dt.days / 365
full["mid"] = (full["bid"] + full["ask"]) / 2
full["iv"] = implied_vol_many(full["mid"], full["spot"], full["strike"], full["T"], full["tbill"].values,
                              full["type"], q=full["div_yield"].values)
full["delta"] = np.nan_to_num(delta(full))     # worthless far-away options: delta ~ 0
full["day_no"] = full["date"].map(day_no)
carry = full["spot"] * np.exp(-full["div_yield"] * full["T"]) - full["strike"] * np.exp(-full["tbill"] * full["T"])
is_call = full["type"] == "call"
sign = np.where(is_call, -1.0, 1.0)            # a call gives a put (subtract carry), a put gives a call (add)
mirror = full.assign(type=np.where(is_call, "put", "call"),
                     bid=full["bid"] + sign * carry, ask=full["ask"] + sign * carry,
                     mid=full["mid"] + sign * carry,
                     delta=full["delta"] + np.where(is_call, -1.0, 1.0) * np.exp(-full["div_yield"] * full["T"]))
panel = (pd.concat([full.assign(real=True), mirror.assign(real=False)])
         .drop_duplicates(key + ["day_no"], keep="first")          # real quotes win over rebuilt ones
         .set_index(key + ["day_no"])[["bid", "ask", "mid", "spot", "delta"]])


def hold(entry, h):
    """Follow each entry for h trading days, re-hedging daily. Returns entry rows with P&L columns.
    Only trades that can be held the whole time: exit inside the test period, and the option
    still at least a week from expiry at exit (our data stops at 7 days to expiry)."""
    cal_days = int(np.ceil(h * 7 / 5)) + 3
    entry = entry[(entry["day_no"] + h <= len(days) - 1) & (entry["T"] * 365 >= 7 + cal_days)]
    cur = entry[key + ["day_no", "mid", "bid", "ask", "spot", "delta"]].copy()
    hedge = np.zeros(len(cur))
    alive = np.ones(len(cur), bool)
    for step in range(1, h + 1):
        nxt = cur[key].assign(day_no=cur["day_no"] + step)
        nd = panel.reindex(pd.MultiIndex.from_frame(nxt)).reset_index(drop=True)
        alive &= nd["mid"].notna().values
        prev_spot = cur["spot"].values if step == 1 else prev["spot"].values
        prev_delta = cur["delta"].values if step == 1 else prev["delta"].values
        hedge += np.nan_to_num(prev_delta * (nd["spot"].values - prev_spot))
        prev = nd.fillna({"spot": 0, "delta": 0})
    out = entry.copy()
    out["alive"] = alive
    hold.dropped.append(1 - alive.mean() if len(alive) else 0)
    out["exit_mid"], out["exit_bid"], out["exit_ask"] = nd["mid"].values, nd["bid"].values, nd["ask"].values
    out["hedge"] = hedge
    return out[out["alive"]]


hold.dropped = []


def summarize(name, side, t):
    sign = 1 if side == "buy" else -1
    mid = sign * ((t["exit_mid"] - t["mid"]) - t["hedge"]) * 100
    real = (((t["exit_bid"] - t["ask"]) - t["hedge"]) if side == "buy"
            else -((t["exit_ask"] - t["bid"]) - t["hedge"])) * 100
    daily = real.groupby(t["date"]).mean()
    return {"rule": name, "side": side, "trades": len(t),
            "$/trade mid-to-mid": mid.mean(), "$/trade real": real.mean(),
            "% days profitable": (daily > 0).mean(),
            "t-stat": daily.mean() / daily.std() * np.sqrt(len(daily)) if len(daily) > 2 else np.nan}


rules = {
    "gap > 3 vol pts": (p["gap"] < -0.03, p["gap"] > 0.03),
    "model price outside quote": (p["model_price"] > p["ask"], p["model_price"] < p["bid"]),
    "outside quote by 10%+ of spread": (p["model_price"] > p["ask"] + 0.1 * (p["ask"] - p["bid"]),
                                        p["model_price"] < p["bid"] - 0.1 * (p["ask"] - p["bid"])),
}
rng = np.random.default_rng(0)
rand = pd.Series(rng.random(len(p)) < 0.03, index=p.index)

for h in HOLD_DAYS:
    rows = []
    for name, (buy_sel, sell_sel) in rules.items():
        rows.append(summarize(name, "buy", hold(p[buy_sel], h)))
        rows.append(summarize(name, "sell", hold(p[sell_sel], h)))
    rows.append(summarize("random (no skill)", "buy", hold(p[rand], h)))
    rows.append(summarize("random (no skill)", "sell", hold(p[rand], h)))
    r = pd.DataFrame(rows)
    print(f"Held {h} trading day(s), delta-hedged daily, 1 contract per trade "
          f"(trades dropped for missing data: {np.mean(hold.dropped):.1%}):")
    hold.dropped.clear()
    print(r.to_string(index=False, formatters={
        "trades": "{:,}".format, "$/trade mid-to-mid": "${:,.2f}".format, "$/trade real": "${:,.2f}".format,
        "% days profitable": "{:.0%}".format, "t-stat": "{:.1f}".format}))
    print()
print("'real' = buy at the ask / sell at the bid, both ways. t-stat above ~2 = unlikely to be luck.")
print("Options that moved in-the-money are priced by put-call parity; only moves beyond +-20% are dropped.")
