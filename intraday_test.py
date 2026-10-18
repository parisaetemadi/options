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
C. SCHEDULED NEWS: the chosen move rule and the 9:45 straddle on Fed days (decision at 2pm, inside the
   session) vs CPI/jobs days (released 8:30am, before the open) vs normal days, all months together.
   Only ~20 news days -- a description, not a test.

Costs: buy at the ask, sell at the bid (fill 1.0; fill 0.5 shown too), IBKR $0.70 per contract each way.
Rules are picked on 2026-01-23 .. 2026-06-30 and checked ONCE on 2026-07-01 .. 2026-09-28. Days before
2026-01-23 were bought later and never used to pick anything, so they are a second, fresh check (D).
An option with no bid at the exit is sold for $0 (a total loss) -- not dropped, which flattered results.
Writes data/paper_options/validation.json for the website's Validation tab.

Run:  python intraday_test.py
      GROUP=stocks python intraday_test.py   (NVDA + TSLA, Sep 2025 ..; prints only, the website keeps the SPY/QQQ cards)
"""
import glob
import itertools
import os

import numpy as np
import pandas as pd

import strategy_search as ss

SPLIT = "2026-07-01"
FRESH_END = "2026-01-23"     # first day of the original download: everything before it is unseen by the picks
COMM = ss.COMMISSION
GROUP = os.environ.get("GROUP", "etf")
SRC = {"etf": "data/databento/intraday", "stocks": "data/databento/intraday_stocks"}[GROUP]
tbill = pd.read_csv("data/market/vix_rates.csv").set_index("date")["tbill"]

# ---------- load and clean ----------
cols = ["date", "snap", "ticker", "type", "strike", "expiration", "bid", "ask"]
q = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{SRC}/2*.parquet"))],
              ignore_index=True)
q = q.astype({"date": str, "ticker": str, "type": str, "expiration": str})
q["snap"] = pd.to_datetime(q["snap"], utc=True).dt.tz_convert("America/New_York")
q["bid"] = q["bid"].fillna(0)        # Databento leaves the bid blank when nobody is bidding: that's a $0 bid
q["mid"] = (q["bid"] + q["ask"]) / 2
q["half"] = (q["ask"] - q["bid"]) / 2
# exits may land on a quote with no bid (worthless by then): it sells for $0, so keep those for the exit book
exit_quotes = q[(q["ask"] > 0) & (q["bid"] >= 0) & (q["ask"] >= q["bid"])]
q = q[(q["bid"] > 0) & (q["ask"] > q["bid"])].copy()
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
book = exit_quotes.set_index(["date", "ticker", "type", "strike", "expiration", "snap"])[["mid", "half"]].sort_index()


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


def stats(t, min_n=20):
    if len(t) < min_n:
        return {"n": len(t)}
    by_day = t.groupby("date")["pnl"].sum()
    return {"n": len(t), "$/trade": t["pnl"].mean(), "% won": (t["pnl"] > 0).mean() * 100,
            "best 5% share of gains": t["pnl"].nlargest(max(1, len(t) // 20)).sum() / max(t.loc[t["pnl"] > 0, "pnl"].sum(), 1e-9) * 100,
            "t (by day)": by_day.mean() / by_day.std() * np.sqrt(len(by_day)) if by_day.std() > 0 else np.nan}


def brief(c):
    return {"n": int(c["n"]), "per_trade": round(float(c.get("$/trade", np.nan)), 2), "won": round(float(c.get("% won", np.nan)), 1),
            "t": round(float(c.get("t (by day)", np.nan)), 2)} if c["n"] >= 5 else {"n": int(c["n"])}


def periods(full, half):
    """Split one rule's trades: picked on (sel), checked once (conf), and the never-seen earlier days (fresh)."""
    d = full["date"]
    return {"sel": full[(d >= FRESH_END) & (d < SPLIT)], "conf": full[d >= SPLIT], "conf_half": half[half["date"] >= SPLIT],
            "fresh": full[d < FRESH_END]}


def report(name, rows):
    res = pd.DataFrame([{**r["rule"], **{f"sel {k}": v for k, v in stats(r["sel"]).items()}} for r in rows])
    print(f"\n===== {name}: picked on {FRESH_END} .. {SPLIT} (full bid/ask + fees), top 12 =====")
    print(res.sort_values("sel t (by day)", ascending=False).head(12).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    best = rows[int(res["sel t (by day)"].idxmax())]
    print(f"Chosen: {best['rule']}  ({len(rows)} rules tried -- a selection t below ~3 is likely luck)")
    for label, t in [(f"CONFIRMATION {SPLIT}.., fill 1.0", best["conf"]), (f"CONFIRMATION {SPLIT}.., fill 0.5", best["conf_half"]),
                     (f"FRESH (unseen) ..{FRESH_END}, fill 1.0", best["fresh"])]:
        c = stats(t)
        print(f"  {label}: {c['n']} trades, ${c.get('$/trade', np.nan):,.2f}/trade, "
              f"{c.get('% won', np.nan):.0f}% won, t {c.get('t (by day)', np.nan):.1f}")
    return {**best, "tried": len(rows)}


# ---------- A. move rules ----------
GRID = {"style": ["momentum", "reversal"], "look": ["15m", "30m", "60m", "120m", "open"], "thr": [0.0015, 0.003] if GROUP == "etf" else [0.003, 0.006],   # stocks move ~2-3x more
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
    rows.append({"rule": r, **periods(full, half)})
best_move = report("A. buy short-dated options on intraday moves", rows)
# the rule being paper-traded (picked before blank bids were counted as $0), shown as it stands now
PAPER_RULE = {"style": "reversal", "look": "15m", "thr": 0.003, "hold": "close"}
paper_move = {**next(r for r in rows if r["rule"] == PAPER_RULE), "tried": len(rows)}
for label, t in [("picked on", paper_move["sel"]), ("confirmation", paper_move["conf"]), ("fresh (unseen)", paper_move["fresh"])]:
    c = stats(t)
    print(f"  PAPER RULE {PAPER_RULE} {label}: {c['n']} trades, ${c.get('$/trade', np.nan):,.2f}/trade, "
          f"{c.get('% won', np.nan):.0f}% won, t {c.get('t (by day)', np.nan):.1f}")

# ---------- B. time of day: straddle bought (or sold) at each time, closed at 16:00 ----------
rows = []
straddle = atm[atm["hhmm"] <= "15:30"]
for side, hhmm in itertools.product(["buy", "sell"], sorted(straddle["hhmm"].unique())):
    e = exits(straddle[straddle["hhmm"] == hhmm], "close")
    full = pnl(e, side, 1.0).groupby(["date", "ticker"]).agg(pnl=("pnl", "sum"), premium=("premium", "sum")).reset_index()
    half = pnl(e, side, 0.5).groupby(["date", "ticker"]).agg(pnl=("pnl", "sum"), premium=("premium", "sum")).reset_index()
    rows.append({"rule": {"side": side, "enter": hhmm}, **periods(full, half)})
best_time = report("B. at-the-money straddle at a time of day, closed at 16:00", rows)


# ---------- C. scheduled news days vs normal days ----------
ev = pd.read_csv("data/events/macro.csv")
kind = {d: "CPI/jobs day" for d in ev.loc[ev["type"] != "fomc", "reaction_day"]} \
    | {d: "Fed day" for d in ev.loc[ev["type"] == "fomc", "reaction_day"]}           # Fed wins when both
day_kind = lambda t: t["date"].map(kind).fillna("normal day")
e = exits(atm[atm["hhmm"] == "09:45"], "close")
straddles = {side: pnl(e, side, 1.0).groupby(["date", "ticker"]).agg(pnl=("pnl", "sum")).reset_index() for side in ("buy", "sell")}
print(f"\n===== C. scheduled news days, {q['date'].min()} .. {q['date'].max()}, full bid/ask + fees =====")
fed_days = set(ev.loc[ev["type"] == "fomc", "reaction_day"])
fs = {f: pnl(e, "buy", f).groupby(["date", "ticker"]).agg(pnl=("pnl", "sum")).reset_index() for f in (1.0, 0.5)}
fed_straddle = {**periods(fs[1.0][fs[1.0]["date"].isin(fed_days)], fs[0.5][fs[0.5]["date"].isin(fed_days)]), "tried": 1}
news = []
r0 = best_move["rule"]
for name, t in [(f"{r0['style']} after {r0['thr']:.2%} over {r0['look']}", pd.concat([best_move["sel"], best_move["conf"], best_move["fresh"]])),
                ("buy 9:45 straddle, hold to 16:00", straddles["buy"]), ("sell 9:45 straddle, hold to 16:00", straddles["sell"])]:
    print(name)
    for k, g in t.groupby(day_kind(t)):
        c = stats(g, min_n=5)
        print(f"  {k:13s} {g['date'].nunique():3d} days, {c['n']:4d} trades, ${c.get('$/trade', np.nan):8,.2f}/trade, "
              f"{c.get('% won', np.nan):3.0f}% won, t {c.get('t (by day)', np.nan):.1f}")
        if c["n"] >= 5:
            news.append({"rule": name, "kind": k, "days": int(g["date"].nunique()), **brief(c)})


# ---------- publish for the website (data/paper_options/validation.json) ----------
if GROUP != "etf":
    raise SystemExit
import json
from pathlib import Path


def verdict(best):
    c, f = stats(best["conf"], min_n=5), stats(best["fresh"], min_n=5)
    strong = [x.get("t (by day)", 0) >= 2 and x.get("$/trade", 0) > 0 for x in (c, f)]
    positive = [x.get("$/trade", 0) > 0 for x in (c, f)]
    return ("holds up in both checks — still needs paper trading" if all(strong)
            else "positive in both checks but weak (t under 2) — not proven" if all(positive)
            else "positive in one check only — not proven" if any(positive)
            else "fails the checks")


def card(name, title, best):
    return {"name": name, "title": title, "tried": best["tried"], "selection": brief(stats(best["sel"], min_n=5)),
            "confirmation": brief(stats(best["conf"], min_n=5)), "confirmation_half": brief(stats(best["conf_half"], min_n=5)),
            "fresh": brief(stats(best["fresh"], min_n=5)), "verdict": verdict(best)}


r = best_move["rule"]
out = {"as_of": pd.Timestamp.now().strftime("%Y-%m-%d"),
       "data": f"Databento SPY/QQQ option quotes every 15 min, {q['date'].min()} .. {q['date'].max()}",
       "select": f"{FRESH_END} .. {SPLIT}", "confirm": f"{SPLIT} .. {q['date'].max()}", "fresh": f"{q['date'].min()} .. {FRESH_END}",
       "rules": [card("best time-of-day rule now", f"{best_time['rule']['side']} the straddle at {best_time['rule']['enter']}, held to the close", best_time),
                 card("best move rule now", f"{r['style']} after a {r['thr']:.2%} move over {r['look']}, held to {r['hold']}", best_move),
                 card("reversal-15m (dropped)", "buy against a 0.3%+ move over 15 min, held to the close", paper_move),
                 card("fed-straddle (dropped)", "buy the 9:45 straddle on Fed days, held to the close (one result per day and ticker)", fed_straddle)],
       "news_days": news}
Path("data/paper_options").mkdir(parents=True, exist_ok=True)
vf = Path("data/paper_options/validation.json")
if vf.exists():          # keep the cards iron_fly_test.py publishes (the paper-traded rule and its plain version)
    old = json.loads(vf.read_text()).get("rules", [])
    out["rules"] = [r for r in old if r["name"].startswith(("iron-fly", "plain 11:15", "put-spread"))] + out["rules"]
vf.write_text(json.dumps(out, indent=1))
print("\nWrote data/paper_options/validation.json")
