"""
Event trades: do options over- (or under-) price the move on earnings and Fed / CPI / jobs days?

For every event, buy or sell the near-the-money pair (OTM put just below + OTM call just above the
stock, the expiry 12-45 days out closest to 20 days) at the close BEFORE the news, delta-hedge,
and close at the end of the reaction day. Real bid/ask both ways.
  earnings: AAPL MSFT NVDA AMZN TSLA          fomc / cpi / jobs: SPY QQQ IWM

Signal (rough, known before the news): the move options price in for the event day vs the average
actual move over the ticker's previous 8 events of that type (ratio > 1: options expect more than usual).
  implied event move = sqrt( atm_iv^2 * T  -  rv21^2 * (T - 1/252) )   (total minus normal days;
  overstates the event, because options also carry a normal premium over realized vol)

Rules are chosen on 2014-2021 (best per event type) and checked ONCE on 2022-today.

Run:  python event_trades.py
"""
import itertools

import numpy as np
import pandas as pd

import strategy_search as ss
import train as tr

SELECT = range(2014, 2022)
EVENT_TICKERS = {"earnings": ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"],
                 "fomc": ["SPY", "QQQ", "IWM"], "cpi": ["SPY", "QQQ", "IWM"], "jobs": ["SPY", "QQQ", "IWM"]}
FILTERS = {"all": (None, None), "ratio > 1.0": (1.0, None), "ratio > 1.2": (1.2, None),
           "ratio < 1.0": (None, 1.0), "ratio < 0.8": (None, 0.8)}
LOOKBACK = 8

# ---------- events and what the stock actually did ----------
ev = pd.read_csv(tr.DATA / "events" / "earnings.csv")
ev = ev[ev["type"] == "earnings"][["ticker", "reaction_day"]].assign(event="earnings")
mac = pd.read_csv(tr.DATA / "events" / "macro.csv")
ev = pd.concat([ev] + [mac.assign(ticker=t, event=mac["type"])[["ticker", "reaction_day", "event"]]
                       for t in ["SPY", "QQQ", "IWM"]], ignore_index=True)
ev = ev[[t in EVENT_TICKERS[e] for t, e in zip(ev["ticker"], ev["event"])]]
ev = ev[ev["reaction_day"].isin(ss.day_no)].drop_duplicates(["ticker", "reaction_day"])  # FOMC+CPI same day: once
ev["exit_no"] = ev["reaction_day"].map(ss.day_no)
ev = ev[ev["exit_no"] > 0]
ev["date"] = ss.all_days[ev["exit_no"] - 1]                       # entry: the close before the news

s = pd.read_csv(tr.DATA / "databento" / "underlying.csv").sort_values(["ticker", "date"])
s["ret"] = s.groupby("ticker")["close"].transform(lambda c: np.log(c / c.shift(1)))
s = s.merge(pd.read_csv(tr.DATA / "events" / "splits.csv"), on=["ticker", "date"], how="left")
s["ret"] += np.log(s["ratio"].fillna(1.0))
ev = ev.merge(s[["ticker", "date", "ret"]].rename(columns={"date": "reaction_day", "ret": "move"}),
              on=["ticker", "reaction_day"])
ev["move"] = ev["move"].abs()
ev = ev.sort_values("reaction_day")
# average actual move over the PREVIOUS 8 events of this type (never including this one)
ev["past_move"] = ev.groupby(["ticker", "event"])["move"].transform(
    lambda m: m.shift(1).rolling(LOOKBACK, min_periods=4).mean())

# ---------- the option pair at entry ----------
cols = ["date", "ticker", "type", "strike", "expiration", "spot", "bid", "ask", "mid", "T", "tbill", "div_yield",
        "iv", "atm_iv", "atm_used", "rv21"]
q = pd.read_parquet(tr.Path(tr.__file__).parent / "models" / "dataset_core.parquet", columns=cols,
                    filters=[("atm_used", "==", True)])
q = q.astype({"date": str, "ticker": str, "type": str, "expiration": str})
q = q.merge(ev[["ticker", "date"]], on=["ticker", "date"])
q["dte"] = q["T"] * 365
q = q[q["dte"].between(12, 45)]
q["dist"] = (q["dte"] - 20).abs()
q = q[q["dist"] == q.groupby(["date", "ticker"])["dist"].transform("min")]
legs = q.merge(ev, on=["ticker", "date"])
event_var = legs["atm_iv"] ** 2 * legs["T"] - legs["rv21"] ** 2 * (legs["T"] - 1 / 252)
legs["implied_move"] = np.sqrt(event_var.clip(lower=0))
legs["ratio"] = legs["implied_move"] / legs["past_move"]
legs["delta"] = ss.delta(legs)
legs["day_no"] = legs["exit_no"] - 1
legs["year"] = legs["date"].str[:4].astype(int)
legs = legs[legs["year"] >= 2014]

# what the options priced in, measured the standard way (after the fact, for description only):
# the same expiry's variance before minus after the news = the variance the event "used up"
post = pd.read_parquet(tr.Path(tr.__file__).parent / "models" / "dataset_core.parquet",
                       columns=["date", "ticker", "expiration", "atm_iv", "T"], filters=[("atm_used", "==", True)])
post = post.astype({"date": str, "ticker": str, "expiration": str}).drop_duplicates(["date", "ticker", "expiration"])
post = post.rename(columns={"date": "reaction_day", "atm_iv": "iv_after", "T": "T_after"})
pairs = legs.drop_duplicates(["ticker", "date"]).merge(post, on=["ticker", "reaction_day", "expiration"])
pairs["crush_move"] = np.sqrt((pairs["atm_iv"] ** 2 * pairs["T"] - pairs["iv_after"] ** 2 * pairs["T_after"]).clip(lower=0))
# both as standard deviations (root-mean-square), so they're comparable
print("Reaction-day move, % of stock price: priced in by options (IV crush) vs actual, both as RMS:")
rms = lambda x: np.sqrt(np.mean(x ** 2)) * 100
print(pairs.assign(y=np.where(pairs["date"] < "2022", "2014-21", "2022+"))
      .groupby(["event", "y"]).agg(events=("move", "size"), priced_in=("crush_move", rms), actual=("move", rms))
      .round(2).to_string())

# ---------- P&L of each pair, both sides ----------
res = []
for year in sorted(legs["year"].unique()):
    ly = legs[legs["year"] == year]
    days = sorted(set(ly["date"]) | set(ly["reaction_day"]))
    panel = ss.panel_for(days)
    for side in ["sell", "buy"]:
        t = ss.trade_pnl(ly, 1, panel, side)
        t = t.assign(premium=t["ask" if side == "buy" else "bid"] * 100)
        res.append(t.groupby(["ticker", "date"]).agg(pnl=("pnl", "sum"), premium=("premium", "sum"), event=("event", "first"),
                                                    ratio=("ratio", "first"), dropped=("dropped", "max"))
                   .reset_index().assign(side=side))
    print(f"{year} done", flush=True)
res = pd.concat(res, ignore_index=True)
res["ret"] = res["pnl"] / res["premium"]                           # P&L as a share of the premium
res["year"] = res["date"].str[:4].astype(int)


def stats(t):
    if len(t) < 5:
        return {"n": len(t)}
    daily = t.groupby("date")["ret"].mean()
    return {"n": len(t), "$/pair": t["pnl"].mean(), "% of premium": t["ret"].mean() * 100,
            "% won": (t["pnl"] > 0).mean() * 100, "t": ss.nw_tstat(daily.values, 1)}


rows = []
for event, side, (f, (lo, hi)) in itertools.product(EVENT_TICKERS, ["sell", "buy"], FILTERS.items()):
    t = res[(res["event"] == event) & (res["side"] == side)]
    if lo is not None:
        t = t[t["ratio"] > lo]
    if hi is not None:
        t = t[t["ratio"] < hi]
    rows.append({"event": event, "side": side, "filter": f, "_t": t,
                 **{f"sel {k}": v for k, v in stats(t[t["year"].isin(SELECT)]).items()}})
table = pd.DataFrame(rows)
print("\nSELECTION 2014-2021, delta-hedged pair held from the close before the news to the close after:")
print(table.drop(columns="_t").to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

print("\nBest rule per event type, checked ONCE on 2022-today (4 checks: only t > ~2.5 is convincing):")
for event, g in table.groupby("event", sort=False):
    b = g.loc[g["sel t"].idxmax()]
    c = stats(b["_t"][~b["_t"]["year"].isin(SELECT)])
    print(f"  {event:8s} {b['side']} ({b['filter']}): sel t {b['sel t']:.1f} -> CONFIRM n {c['n']}, "
          f"${c.get('$/pair', np.nan):,.0f}/pair, {c.get('% of premium', np.nan):+.1f}% of premium, "
          f"{c.get('% won', np.nan):.0f}% won, t {c.get('t', np.nan):.1f}")
