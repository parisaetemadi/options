"""
Intraday option trades on SPY and QQQ (Databento, every 15 min, options 0-7 days out, strikes +-3%).
BUYING short-dated options -- many small losses, rare big wins -- at different speeds.

The stock price at each snapshot comes from the options themselves (put-call parity at the strike
where call and put are closest in price, nearest expiry): S = K + C - P + K*r*T.

A. MOVE rules: when the stock moved more than `thr` over the last `look` (15/30/60/120 min, or since
   9:45), buy the at-the-money option of the nearest expiry -- with the move (momentum: call after a
   rise) or against it (reversal) -- and sell it `hold` later (15/30/60/120 min, or at 16:00).
   100 rules: with that many, some look good by luck -- hence the separate confirmation months.
   At most one trade per ticker per snapshot; entries 9:45-15:30.
B. TIME-OF-DAY: buy (or sell) the at-the-money straddle of the nearest expiry at each time of day and
   close it at 16:00 -- is short-dated volatility cheap or expensive at certain hours?

Costs: buy at the ask, sell at the bid (fill 1.0; fill 0.5 shown too), IBKR $0.70 per contract each way.
Rules are picked on 2026-01-23 .. 2026-06-30 and checked ONCE on 2026-07-01 .. 2026-09-28.

Run:  python intraday_test.py
"""
import glob
import itertools

import numpy as np
import pandas as pd

import strategy_search as ss

SPLIT = "2026-07-01"
COMM = ss.COMMISSION
tbill = pd.read_csv("data/market/vix_rates.csv").set_index("date")["tbill"]

# ---------- load and clean ----------
cols = ["date", "snap", "ticker", "type", "strike", "expiration", "bid", "ask"]
q = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob("data/databento/intraday/2*.parquet"))],
              ignore_index=True)
q = q.astype({"date": str, "ticker": str, "type": str, "expiration": str})
q = q[(q["bid"] > 0) & (q["ask"] > q["bid"])].copy()
q["mid"] = (q["bid"] + q["ask"]) / 2
q["half"] = (q["ask"] - q["bid"]) / 2
q["snap"] = pd.to_datetime(q["snap"], utc=True).dt.tz_convert("America/New_York")
q["hhmm"] = q["snap"].dt.strftime("%H:%M")
q["dte"] = (pd.to_datetime(q["expiration"]) - pd.to_datetime(q["date"])).dt.days
print(f"{len(q):,} live quotes, {q['date'].nunique()} days, {q['date'].min()} to {q['date'].max()}")

# ---------- the stock price at every snapshot, from put-call parity on the nearest expiry ----------
near = q[q["dte"] == q.groupby(["date", "ticker"])["dte"].transform("min")]
pc = near.pivot_table(index=["date", "snap", "ticker", "strike", "dte"], columns="type", values="mid").dropna().reset_index()
pc["gap"] = (pc["call"] - pc["put"]).abs()
pc = pc.loc[pc.groupby(["date", "snap", "ticker"])["gap"].idxmin()]
pc["r"] = pc["date"].map(tbill).ffill().fillna(0.04)
pc["spot"] = pc["strike"] + pc["call"] - pc["put"] + pc["strike"] * pc["r"] * np.maximum(pc["dte"], 0.25) / 365
spot = pc.set_index(["date", "ticker", "snap"])["spot"].sort_index()
# sanity check against the official close on days we have it
closes = pd.read_csv("data/databento/underlying.csv").set_index(["ticker", "date"])["close"]
last = spot.groupby(["date", "ticker"]).last()
chk = (last / closes.reindex(list(zip(last.index.get_level_values(1), last.index.get_level_values(0)))).values - 1).abs()
print(f"Parity stock price at 16:00 vs official close: median gap {np.nanmedian(chk):.3%}, 99th pct {np.nanpercentile(chk, 99):.3%}")

snaps = spot.reset_index()
snaps["i"] = snaps.groupby(["date", "ticker"]).cumcount()
snaps["n"] = snaps.groupby(["date", "ticker"])["i"].transform("max")
snaps["open_spot"] = snaps.groupby(["date", "ticker"])["spot"].transform("first")
STEPS = {"15m": 1, "30m": 2, "60m": 4, "120m": 8}       # 15-minute snapshots
for look, k in STEPS.items():
    snaps[f"ret_{look}"] = snaps["spot"] / snaps.groupby(["date", "ticker"])["spot"].shift(k) - 1
snaps["ret_open"] = snaps["spot"] / snaps["open_spot"] - 1

# at-the-money option of the nearest expiry, per snapshot and type
atm = q[q["dte"] == q.groupby(["date", "ticker"])["dte"].transform("min")].merge(
    snaps[["date", "ticker", "snap", "spot"]], on=["date", "ticker", "snap"])
atm["dist"] = (atm["strike"] - atm["spot"]).abs()
atm = atm.loc[atm.groupby(["date", "ticker", "snap", "type"])["dist"].idxmin()]
book = q.set_index(["date", "ticker", "type", "strike", "expiration", "snap"])[["mid", "half"]].sort_index()


def exits(entries, hold):
    """Exit quote for each entry `hold` snapshots later (or at the day's last snapshot)."""
    e = entries.merge(snaps[["date", "ticker", "snap", "i", "n"]], on=["date", "ticker", "snap"]).reset_index(drop=True)
    target = e["n"] if hold == "close" else np.minimum(e["i"] + STEPS[hold], e["n"])
    xs = snaps.set_index(["date", "ticker", "i"])["snap"]
    e["exit_snap"] = xs.reindex(list(zip(e["date"], e["ticker"], target))).reset_index(drop=True)
    e = e[e["exit_snap"] > e["snap"]]
    x = book.reindex(pd.MultiIndex.from_arrays([e["date"], e["ticker"], e["type"], e["strike"], e["expiration"], e["exit_snap"]]))
    return e.assign(x_mid=x["mid"].values, x_half=x["half"].values).dropna(subset=["x_mid"])


def pnl(e, side, fill):
    if side == "buy":
        p = (e["x_mid"] - fill * e["x_half"]) - (e["mid"] + fill * e["half"])
    else:
        p = (e["mid"] - fill * e["half"]) - (e["x_mid"] + fill * e["x_half"])
    return e.assign(pnl=p * 100 - 2 * COMM, premium=e["mid"] * 100)


def stats(t):
    if len(t) < 20:
        return {"n": len(t)}
    by_day = t.groupby("date")["pnl"].sum()
    return {"n": len(t), "$/trade": t["pnl"].mean(), "% won": (t["pnl"] > 0).mean() * 100,
            "best 5% share of gains": t["pnl"].nlargest(max(1, len(t) // 20)).sum() / max(t.loc[t["pnl"] > 0, "pnl"].sum(), 1e-9) * 100,
            "t (by day)": by_day.mean() / by_day.std() * np.sqrt(len(by_day)) if by_day.std() > 0 else np.nan}


def report(name, rows):
    res = pd.DataFrame([{**r["rule"], **{f"sel {k}": v for k, v in stats(r["sel"]).items()}} for r in rows])
    print(f"\n===== {name}: picked on {q['date'].min()} .. {SPLIT} (full bid/ask + fees), top 12 =====")
    print(res.sort_values("sel t (by day)", ascending=False).head(12).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    best = rows[int(res["sel t (by day)"].idxmax())]
    print(f"Chosen: {best['rule']}  ({len(rows)} rules tried -- a selection t below ~3 is likely luck)")
    for label, t in [("fill 1.0", best["conf"]), ("fill 0.5", best["conf_half"])]:
        c = stats(t)
        print(f"  CONFIRMATION {SPLIT}..: {label}: {c['n']} trades, ${c.get('$/trade', np.nan):,.2f}/trade, "
              f"{c.get('% won', np.nan):.0f}% won, t {c.get('t (by day)', np.nan):.1f}")


# ---------- A. move rules ----------
GRID = {"style": ["momentum", "reversal"], "look": ["15m", "30m", "60m", "120m", "open"], "thr": [0.0015, 0.003],
        "hold": ["15m", "30m", "60m", "120m", "close"]}
rows = []
cand = snaps[(snaps["snap"].dt.strftime("%H:%M") >= "09:45") & (snaps["snap"].dt.strftime("%H:%M") <= "15:30")]
for r in (dict(zip(GRID, v)) for v in itertools.product(*GRID.values())):
    mv = cand[f"ret_{r['look']}"]
    up, down = cand[mv > r["thr"]], cand[mv < -r["thr"]]
    want_call = up if r["style"] == "momentum" else down
    want_put = down if r["style"] == "momentum" else up
    ent = pd.concat([atm.merge(want_call[["date", "ticker", "snap"]], on=["date", "ticker", "snap"]).query("type == 'call'"),
                     atm.merge(want_put[["date", "ticker", "snap"]], on=["date", "ticker", "snap"]).query("type == 'put'")])
    e = exits(ent, r["hold"])
    full, half = pnl(e, "buy", 1.0), pnl(e, "buy", 0.5)
    rows.append({"rule": r, "sel": full[full["date"] < SPLIT], "conf": full[full["date"] >= SPLIT],
                 "conf_half": half[half["date"] >= SPLIT]})
report("A. buy short-dated options on intraday moves", rows)

# ---------- B. time of day: straddle bought (or sold) at each time, closed at 16:00 ----------
rows = []
straddle = atm[atm["hhmm"] <= "15:30"]
for side, hhmm in itertools.product(["buy", "sell"], sorted(straddle["hhmm"].unique())):
    e = exits(straddle[straddle["hhmm"] == hhmm], "close")
    full = pnl(e, side, 1.0).groupby(["date", "ticker"]).agg(pnl=("pnl", "sum"), premium=("premium", "sum")).reset_index()
    half = pnl(e, side, 0.5).groupby(["date", "ticker"]).agg(pnl=("pnl", "sum"), premium=("premium", "sum")).reset_index()
    rows.append({"rule": {"side": side, "enter": hhmm}, "sel": full[full["date"] < SPLIT],
                 "conf": full[full["date"] >= SPLIT], "conf_half": half[half["date"] >= SPLIT]})
report("B. at-the-money straddle at a time of day, closed at 16:00", rows)
