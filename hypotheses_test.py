"""
The five pre-registered ideas in HYPOTHESES.md (committed before this file was written), each tested
ONCE with exactly the rules written there. Nothing here is tuned; a failed idea is dropped.

  H1 pre-earnings volatility run-up      buy the near-the-money pair 5 days before, sell before the news
  H2 weekend decay                       sell an iron condor (wings 5% out) at the week's last close, buy back next close
  H3 overnight decay (intraday data)     sell the 1% iron fly at 15:45, buy back at 9:45 next day
  H4 index put spreads                   sell the 5%-out put, buy the 10%-out put, monthly, held to expiry
  H5 pre-Fed volatility run-up           as H1, on SPY/QQQ/IWM before Fed decisions

Pass = t >= 2.6 at full costs and average > 0 in both halves of the entry dates.

Run:  python hypotheses_test.py            (all)      python hypotheses_test.py H3   (one)
"""
import glob
import sys

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

import strategy_search as ss
from pricing import bs_price, implied_vol_many

COMM = ss.COMMISSION
DATA = ss.DATA
GROUPS = ("core", "extra", "more", "wide")
KEY = ["ticker", "type", "strike", "expiration"]
all_days = ss.all_days
day_no = ss.day_no
closes = ss.closes
T_PASS = 2.6


# ---------- daily quotes (all four ticker groups), with in-the-money quotes rebuilt by put-call parity ----------
def quotes(days, tickers):
    files = [DATA / "databento" / "quotes" / g / d[:4] / f"{d}.parquet" for d in sorted(set(days)) for g in GROUPS]
    parts = [pd.read_parquet(p, filters=[("ticker", "in", list(tickers))]) for p in files if p.exists()]
    f = pd.concat(parts, ignore_index=True).astype({"date": str, "ticker": str, "type": str, "expiration": str})
    f = f.merge(ss.rates, on="date", how="left").merge(ss.divs, on=["ticker", "date"], how="left")
    f["tbill"] = f["tbill"].ffill().fillna(0.0)
    f["div_yield"] = f["div_yield"].fillna(0.0)
    f["T"] = (pd.to_datetime(f["expiration"]) - pd.to_datetime(f["date"])).dt.days / 365
    f["mid"] = (f["bid"] + f["ask"]) / 2
    return f


def with_parity(f):
    """Add each quote's same-strike mirror (call <-> put) via put-call parity, so in-the-money options have prices."""
    carry = f["spot"] * np.exp(-f["div_yield"] * f["T"]) - f["strike"] * np.exp(-f["tbill"] * f["T"])
    is_call = f["type"] == "call"
    sign = np.where(is_call, -1.0, 1.0)
    mirror = f.assign(type=np.where(is_call, "put", "call"), bid=f["bid"] + sign * carry, ask=f["ask"] + sign * carry)
    return pd.concat([f, mirror]).drop_duplicates(KEY + ["date"], keep="first").set_index(KEY + ["date"])[["bid", "ask", "spot"]]


def near_pair(g):
    """OTM put with the highest strike <= stock and OTM call with the lowest strike >= stock, both bid > 0."""
    s = g["spot"].iloc[0]
    p = g[(g["type"] == "put") & (g["strike"] <= s) & (g["bid"] > 0)]
    c = g[(g["type"] == "call") & (g["strike"] >= s) & (g["bid"] > 0)]
    if p.empty or c.empty:
        return None
    return [p.loc[p["strike"].idxmax()], c.loc[c["strike"].idxmin()]]


def splits_between(ticker, start, end):
    sp = ss.SPLITS
    return bool(((sp["ticker"] == ticker) & (sp["date"] > start) & (sp["date"] <= end)).any())


def value_legs(legs, fill):
    """$ P&L per leg. legs: entry bid/ask, side (+1 bought, -1 sold), exit bid/ask (NaN = missing: Black-Scholes at
    entry IV from the exit close, with a full extra spread against us). fill = share of the half-spread paid."""
    e_mid, e_half = (legs["bid"] + legs["ask"]) / 2, (legs["ask"] - legs["bid"]) / 2
    miss = legs["x_bid"].isna() | legs["x_ask"].isna()
    bs = bs_price(legs["type"].values, legs["x_spot"].values, legs["strike"].values, legs["x_T"].clip(lower=1 / 365).values,
                  legs["tbill"].values, legs["iv"].values, legs["div_yield"].values)
    x_mid = np.where(miss, bs, (legs["x_bid"] + legs["x_ask"]) / 2)
    x_half = np.where(miss, 2 * e_half, (legs["x_ask"] - legs["x_bid"]) / 2)       # missing: a full extra spread
    long = legs["side"] > 0
    entry_px = np.where(long, e_mid + fill * e_half, e_mid - fill * e_half)
    exit_px = np.where(long, x_mid - fill * x_half, x_mid + fill * x_half)
    exit_px = np.maximum(exit_px, 0.0)                                               # a bought option never sells below $0
    return (exit_px - entry_px) * legs["side"].values * 100 - 2 * COMM, miss


def run_daily(trades):
    """trades: list of dicts {id, ticker, date, exit, legs: [(row, side)]}. Returns one row per trade (fill 1.0 and 0.5)."""
    if not trades:
        return pd.DataFrame()
    rows = []
    for t in trades:
        for r, side in t["legs"]:
            rows.append({"id": t["id"], "entry": t["date"], "exit": t["exit"], "side": side,
                         **{k: r[k] for k in ["ticker", "type", "strike", "expiration", "bid", "ask", "spot", "T", "tbill", "div_yield"]}})
    legs = pd.DataFrame(rows)
    legs["iv"] = implied_vol_many((legs["bid"] + legs["ask"]) / 2, legs["spot"], legs["strike"], legs["T"],
                                  legs["tbill"].values, legs["type"], q=legs["div_yield"].values)
    legs["iv"] = legs["iv"].fillna(legs.groupby("id")["iv"].transform("mean")).fillna(0.3)
    out = []
    for yr, ly in legs.groupby(legs["exit"].str[:4]):
        px = with_parity(quotes(ly["exit"].unique(), ly["ticker"].unique()))
        x = px.reindex(pd.MultiIndex.from_frame(ly[KEY + ["exit"]].rename(columns={"exit": "date"})))
        ly = ly.assign(x_bid=x["bid"].values, x_ask=x["ask"].values,
                       x_spot=closes.reindex(list(zip(ly["ticker"], ly["exit"]))).values,
                       x_T=(pd.to_datetime(ly["expiration"]) - pd.to_datetime(ly["exit"])).dt.days / 365)
        ly["x_spot"] = ly["x_spot"].fillna(ly["spot"])
        out.append(ly)
    legs = pd.concat(out, ignore_index=True)
    legs["pnl"], legs["missing"] = value_legs(legs, 1.0)
    legs["pnl_half"], _ = value_legs(legs, 0.5)
    return legs.groupby("id").agg(ticker=("ticker", "first"), date=("entry", "first"), pnl=("pnl", "sum"),
                                  pnl_half=("pnl_half", "sum"), missing=("missing", "max")).reset_index()


# ---------- statistics and the pass rule ----------
def verdict(name, t, lags=0):
    if t.empty:
        print(f"\n{name}: no trades")
        return {"name": name, "trades": 0, "pass": False}
    by = t.groupby("date")["pnl"].sum().sort_index()
    tstat = ss.nw_tstat(by.values, lags) if lags else by.mean() / by.std() * np.sqrt(len(by))
    dates = sorted(t["date"].unique())
    mid = dates[len(dates) // 2]                                   # median entry date
    h1, h2 = t[t["date"] < mid]["pnl"].mean(), t[t["date"] >= mid]["pnl"].mean()
    ok = bool(tstat >= T_PASS and h1 > 0 and h2 > 0)
    print(f"\n===== {name} =====")
    print(f"{len(t):,} trades on {len(by):,} entry dates, {t['date'].min()} .. {t['date'].max()}; "
          f"{t['missing'].mean():.1%} had a missing exit quote")
    print(f"full costs: ${t['pnl'].mean():,.2f}/trade, {(t['pnl'] > 0).mean():.0%} won, t {tstat:.2f}; "
          f"halves (split {mid}): ${h1:,.2f} / ${h2:,.2f}; worst entry date ${by.min():,.0f}")
    print(f"half-spread fills (info only): ${t['pnl_half'].mean():,.2f}/trade")
    if "ticker" in t:
        print("by ticker ($/trade, n): " + ", ".join(f"{k} {g['pnl'].mean():.1f} ({len(g)})" for k, g in t.groupby("ticker")))
    yr = t.groupby(t["date"].str[:4])["pnl"].mean().round(1).to_dict()
    print(f"by year ($/trade): {yr}")
    print(f"VERDICT: {'PASS' if ok else 'FAIL'}")
    return {"name": name, "trades": len(t), "per_trade": round(t["pnl"].mean(), 2), "t": round(tstat, 2),
            "half1": round(h1, 2), "half2": round(h2, 2), "pass": ok}


# ---------- H1 / H5: buy the near-the-money pair 5 trading days before the news, sell at the last close before it ----------
def run_up(events, name):
    """events: DataFrame ticker, news_day (first day the news can move the stock)."""
    trades, skipped = [], {"no data": 0, "no pair": 0, "split": 0}
    ev = events.copy()
    ev["x_no"] = np.searchsorted(all_days, ev["news_day"].values, side="left") - 1      # last close before the news
    ev = ev[(ev["x_no"] - 5 >= 0) & (ev["news_day"] <= all_days[-1])]
    ev["exit"], ev["date"] = all_days[ev["x_no"]], all_days[ev["x_no"] - 5]
    for yr, ey in ev.groupby(ev["date"].str[:4]):
        q = quotes(ey["date"].unique(), ey["ticker"].unique())
        q = q.set_index(["date", "ticker"]).sort_index()
        for i, e in ey.iterrows():
            if (e["date"], e["ticker"]) not in q.index:
                skipped["no data"] += 1
                continue
            if splits_between(e["ticker"], e["date"], e["exit"]):
                skipped["split"] += 1
                continue
            g = q.loc[(e["date"], e["ticker"])].reset_index()
            # nearest expiry on/after the news AND with >= 7 days left at the exit (the data has nothing shorter): amendment 1
            min_exp = max(e["news_day"], (pd.Timestamp(e["exit"]) + pd.Timedelta(days=7)).strftime("%Y-%m-%d"))
            g = g[g["expiration"] >= min_exp]
            if g.empty:
                skipped["no pair"] += 1
                continue
            pair = near_pair(g[g["expiration"] == g["expiration"].min()])
            if pair is None:
                skipped["no pair"] += 1
                continue
            trades.append({"id": f"{e['ticker']} {e['news_day']}", "date": e["date"], "exit": e["exit"],
                           "legs": [(pair[0], 1), (pair[1], 1)]})
    print(f"{name}: {len(ev)} events, {len(trades)} traded, skipped {skipped}")
    return run_daily(trades)


def h1():
    ev = pd.read_csv(DATA / "events" / "earnings.csv").query("type == 'earnings'")
    ev = ev.rename(columns={"reaction_day": "news_day"})[["ticker", "news_day"]].drop_duplicates()
    return verdict("H1 pre-earnings run-up (buy pair 5 days before, sell the close before the news)",
                   run_up(ev, "H1"), lags=5)


def h5():
    m = pd.read_csv(DATA / "events" / "macro.csv").query("type == 'fomc' and release_day <= '2026-09-30'")
    ev = pd.concat([pd.DataFrame({"ticker": t, "news_day": m["release_day"].values}) for t in ["SPY", "QQQ", "IWM"]])
    # the decision is at 2pm: the last close before it is the day BEFORE the decision day
    return verdict("H5 pre-Fed run-up (SPY/QQQ/IWM, buy pair 5 days before, sell the close before the decision)",
                   run_up(ev, "H5"), lags=5)


# ---------- H2: weekend iron condor ----------
def h2():
    tick = ["SPY", "QQQ", "IWM", "NVDA", "TSLA"]
    d = pd.Series(pd.to_datetime(all_days))
    last_of_week = d.groupby(d.dt.strftime("%G-%V")).transform("max") == d
    idx = np.where(last_of_week.values)[0]
    idx = idx[idx + 1 < len(all_days)]
    trades, skipped = [], {"no legs": 0, "split": 0}
    for yr in sorted({all_days[i][:4] for i in idx}):
        iy = [i for i in idx if all_days[i][:4] == yr]
        q = quotes([all_days[i] for i in iy], tick).set_index(["date", "ticker"]).sort_index()
        for i in iy:
            day, nxt = str(all_days[i]), str(all_days[i + 1])
            for tk in tick:
                if (day, tk) not in q.index:
                    continue
                if splits_between(tk, day, nxt):
                    skipped["split"] += 1
                    continue
                g = q.loc[(day, tk)].reset_index()
                g = g[g["expiration"] >= (pd.Timestamp(nxt) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")]   # amendment 1
                if g.empty:
                    skipped["no legs"] += 1
                    continue
                g = g[g["expiration"] == g["expiration"].min()]
                pair = near_pair(g)
                s = g["spot"].iloc[0]
                wp = g[(g["type"] == "put") & (g["strike"] <= 0.95 * s) & (g["ask"] > 0)]
                wc = g[(g["type"] == "call") & (g["strike"] >= 1.05 * s) & (g["ask"] > 0)]
                if pair is None or wp.empty or wc.empty:
                    skipped["no legs"] += 1
                    continue
                trades.append({"id": f"{tk} {day}", "date": day, "exit": nxt,
                               "legs": [(pair[0], -1), (pair[1], -1), (wp.loc[wp["strike"].idxmax()], 1),
                                        (wc.loc[wc["strike"].idxmin()], 1)]})
    print(f"H2: {len(trades)} condors, skipped {skipped}")
    return verdict("H2 weekend iron condor (SPY/QQQ/IWM/NVDA/TSLA, sell at the week's last close, buy back next close)",
                   run_daily(trades))


# ---------- H4: monthly index put spread held to expiry ----------
def h4():
    tick = ["SPY", "QQQ", "IWM"]
    months = pd.period_range("2013-04", "2026-09", freq="M")
    third_fri = [(m.start_time + pd.DateOffset(days=(4 - m.start_time.weekday()) % 7 + 14)).strftime("%Y-%m-%d") for m in months]
    rows, skipped = [], {"no legs": 0, "not expired yet": 0}
    for a, b in zip(third_fri[:-1], third_fri[1:]):
        e_no = np.searchsorted(all_days, a, side="right")          # first trading day after this month's expiry
        if e_no >= len(all_days):
            continue
        day = all_days[e_no]
        q = quotes([day], tick)
        for tk in tick:
            g = q[(q["ticker"] == tk) & (q["type"] == "put")]
            # next month's standard expiry: the 3rd Friday, or the day before it when that Friday is a holiday
            # (dated the Saturday after it before 2015, the day before it when that Friday is a holiday)
            cands = [(pd.Timestamp(b) + pd.Timedelta(days=k)).strftime("%Y-%m-%d") for k in (0, 1, -1)]
            exp = next((c for c in cands if c in set(g["expiration"])), b)
            g = g[g["expiration"] == exp]
            if g.empty:
                skipped["no legs"] += 1
                continue
            s = g["spot"].iloc[0]
            sell = g[g["bid"] > 0]
            if sell.empty:
                skipped["no legs"] += 1
                continue
            k1 = sell.loc[(sell["strike"] - 0.95 * s).abs().idxmin()]
            buy = g[(g["strike"] < k1["strike"]) & (g["ask"] > 0)]
            if buy.empty:
                skipped["no legs"] += 1
                continue
            k2 = buy.loc[(buy["strike"] - 0.90 * s).abs().idxmin()]
            settle_no = np.searchsorted(all_days, exp, side="right") - 1
            if exp > all_days[-1] or settle_no <= e_no:
                skipped["not expired yet"] += 1
                continue
            S = closes.get((tk, all_days[settle_no]), np.nan)
            if not np.isfinite(S) or splits_between(tk, day, exp):
                skipped["no legs"] += 1
                continue
            pay = max(k1["strike"] - S, 0) - max(k2["strike"] - S, 0)               # what the spread owes at expiry
            for fill, col in [(1.0, "pnl"), (0.5, "pnl_half")]:
                credit = ((k1["bid"] + k1["ask"]) / 2 - fill * (k1["ask"] - k1["bid"]) / 2) \
                    - ((k2["bid"] + k2["ask"]) / 2 + fill * (k2["ask"] - k2["bid"]) / 2)
                pnl = (credit - pay) * 100 - 2 * COMM - (2 * COMM if S < k1["strike"] else 0)
                if col == "pnl":
                    row = {"ticker": tk, "date": day, "pnl": pnl, "missing": False, "credit": credit * 100,
                           "width": k1["strike"] - k2["strike"]}
                else:
                    row["pnl_half"] = pnl
            rows.append(row)
    t = pd.DataFrame(rows)
    print(f"H4: {len(t)} put spreads, skipped {skipped}; avg credit ${t['credit'].mean():.0f}, "
          f"avg width ${t['width'].mean():.1f}, sold put ended in the money {(t['pnl'] < -2 * COMM).mean():.0%}")
    res = verdict("H4 index put spread (SPY/QQQ/IWM, sell 5%-out put, buy 10%-out put, monthly, held to expiry)", t)
    publish_h4(t, res)
    return res


def publish_h4(t, res):
    """The website's Validation card for the paper-traded put spread (merged into data/paper_options/validation.json)."""
    import json
    from pathlib import Path

    def line(g, col="pnl"):
        by = g.groupby("date")[col].sum()
        return {"n": int(len(g)), "per_trade": round(float(g[col].mean()), 2), "won": round(float((g[col] > 0).mean() * 100), 1),
                "t": round(float(by.mean() / by.std() * np.sqrt(len(by))), 2)}
    dates = sorted(t["date"].unique())
    mid = dates[len(dates) // 2]
    worst = t.groupby("date")["pnl"].sum()
    card = {"name": "put-spread-monthly (paper-traded)",
            "title": "sell the SPY/QQQ/IWM put ~5% below, buy the one ~10% below, next month's expiry, held to expiry "
                     f"(worst month ${worst.min():,.0f} on {worst.idxmin()}, all three together; loss capped)",
            "lines": [{"label": f"All months {t['date'].min()} .. {t['date'].max()} (written down first, tested once)", "stats": line(t)},
                      {"label": f"First half (.. {mid})", "stats": line(t[t["date"] < mid])},
                      {"label": f"Second half ({mid} ..)", "stats": line(t[t["date"] >= mid])},
                      {"label": "Same, paying half the spread", "stats": line(t, "pnl_half"), "meta": True}],
            "verdict": "positive in both halves but t under 2.6 (the bar set in advance) — not proven; paper-traded unchanged"
            if not res["pass"] else "passes the pre-set bar — still needs paper trading"}
    vf = Path("data/paper_options/validation.json")
    v = json.loads(vf.read_text()) if vf.exists() else {"rules": []}
    v["rules"] = [card] + [r for r in v.get("rules", []) if not r["name"].startswith("put-spread")]
    vf.write_text(json.dumps(v, indent=1))
    print(f"Updated {vf}")


# ---------- H3: overnight iron fly on the intraday data ----------
def parity_spot(g):
    p = g[g["bid"] > 0].pivot_table(index="strike", columns="type", values="mid")
    if not {"call", "put"} <= set(p.columns) or p.dropna().empty:
        return np.nan
    p = p.dropna()
    k = (p["call"] - p["put"]).abs().idxmin()
    return k + p.loc[k, "call"] - p.loc[k, "put"]


def h3():
    nyse = mcal.get_calendar("NYSE").schedule(start_date="2023-01-01", end_date="2026-12-31")
    trading = [d.strftime("%Y-%m-%d") for d in nyse.index]
    nxt_day = dict(zip(trading[:-1], trading[1:]))
    cols = ["date", "snap", "ticker", "type", "strike", "expiration", "bid", "ask"]
    rows = []
    for src in ["data/databento/intraday", "data/databento/intraday_stocks"]:
        files = {f.split("/")[-1][:10]: f for f in sorted(glob.glob(f"{src}/2*.parquet"))}
        for day, f in files.items():
            nd = nxt_day.get(day)
            if nd not in files:
                continue
            a = pd.read_parquet(f, columns=cols)
            b = pd.read_parquet(files[nd], columns=cols)
            for d in (a, b):
                d["snap"] = pd.to_datetime(d["snap"], utc=True).dt.tz_convert("America/New_York").dt.strftime("%H:%M")
                d["bid"] = d["bid"].fillna(0)
                d["mid"] = (d["bid"] + d["ask"]) / 2
            a, b = a[(a["snap"] == "15:45") & (a["ask"] > 0)], b[(b["snap"] == "09:45") & (b["ask"] > 0)]
            for tk, g in a.groupby("ticker"):
                g = g[g["expiration"].astype(str) > day]
                if g.empty:
                    continue
                g = g[g["expiration"] == g["expiration"].min()]
                s_in = parity_spot(g)
                bx = b[(b["ticker"] == tk) & (b["expiration"] == g["expiration"].iloc[0])]
                s_out = parity_spot(bx) if len(bx) else np.nan
                if not np.isfinite(s_in) or not np.isfinite(s_out):
                    continue
                live = g[(g["bid"] > 0) & (g["ask"] > g["bid"])]
                both = set(live.loc[live["type"] == "call", "strike"]) & set(live.loc[live["type"] == "put", "strike"])
                wc = g[(g["type"] == "call") & (g["strike"] >= s_in * 1.01)]
                wp = g[(g["type"] == "put") & (g["strike"] <= s_in * 0.99)]
                if not both or wc.empty or wp.empty:
                    continue
                k0 = min(both, key=lambda k: abs(k - s_in))
                kc, kp = wc["strike"].min(), wp["strike"].max()
                book_in, book_out = g.set_index(["type", "strike"]), bx.set_index(["type", "strike"])
                miss = False

                def out(kind, k, side, fill):
                    nonlocal miss
                    if (kind, k) in book_out.index:
                        r = book_out.loc[(kind, k)]
                        mid, half = r["mid"], (r["ask"] - r["bid"]) / 2
                        return max(mid + fill * half if side < 0 else mid - fill * half, 0.0)
                    miss = True
                    return max(0.0, (s_out - k) if kind == "call" else (k - s_out))

                res = {}
                for fill in (1.0, 0.5):
                    legs = [("call", k0, -1), ("put", k0, -1), ("call", kc, 1), ("put", kp, 1)]
                    pnl = 0.0
                    for kind, k, side in legs:
                        r = book_in.loc[(kind, k)]
                        mid, half = r["mid"], (r["ask"] - r["bid"]) / 2
                        px_in = mid + fill * half if side > 0 else mid - fill * half
                        pnl += (out(kind, k, side, fill) - px_in) * side * 100 - 2 * COMM
                    res[fill] = pnl
                rows.append({"ticker": tk, "date": day, "pnl": res[1.0], "pnl_half": res[0.5], "missing": miss})
    t = pd.DataFrame(rows)
    return verdict("H3 overnight iron fly (sell 15:45, buy back 9:45 next day; SPY/QQQ/NVDA/TSLA)", t)


if __name__ == "__main__":
    want = sys.argv[1:] or ["H1", "H2", "H3", "H4", "H5"]
    fns = {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5}
    results = [fns[h]() for h in want]
    print("\n===== SUMMARY =====")
    print(pd.DataFrame(results).to_string(index=False))
