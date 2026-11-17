"""
Paper trading OPTIONS with fake money, from the free Yahoo intraday snapshots (snapshot_intraday.py).

Every run replays all saved snapshots from the start, so the trade log is always rebuilt the same way
from the same quotes, and nothing depends on what an earlier run remembered. A rule only ever sees
snapshots up to the moment it decides.

Yahoo's option quotes run ~15 minutes behind (their last trades are ~15 min old at snapshot time), so
the stock price used for signals comes from the options themselves (put-call parity), keeping the
signal and the prices it trades on from the same moment -- the same method as intraday_test.py.

Fills: buy at the ask, sell at the bid (the full spread), plus IBKR's $0.70 per contract each way.
Positions held "to the close" are sold at the day's last snapshot from 15:45 on; if the day's runs
stopped earlier, at its last snapshot (flagged "late close"). The monthly put spread is held to its
expiry and settled at its value then, from the stock's last price on expiry day.

Writes data/paper_options/trades.csv and dashboard.json (read by the website's Options page).
Run:  python paper_options.py              (after snapshot_intraday.py; on GitHub: intraday.yml)
      python paper_options.py --coverage   (laptop only: refresh the data-coverage summary)
"""
import argparse
import functools
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
SNAPS = ROOT / "data" / "yahoo_intraday"
OUT = ROOT / "data" / "paper_options"
COMM = 0.70
NY = "America/New_York"

# Rule being paper-traded (see validation.json). History of choices:
# - reversal and Fed-day rules dropped 2026-10-04: failed once worthless options counted as $0;
# - the plain 11:15 sold straddle dropped 2026-10-04: ~ -$1/trade over Aug 2024-Sep 2026 and -$7.6k
#   on 2025-04-09 (iron_fly_test.py);
# - now the same trade with wings (iron fly), which caps the loss. It has NO proven edge either
#   (about -$1/trade overall): it runs to test the live system with a loss limit, not as a bet.
# - 2026-10-18: the monthly index put spread (HYPOTHESES.md H4, the closest any idea came: +$42/trade,
#   91% won, t 2.2 on 2013-2026, short of the 2.6 bar), traded UNCHANGED so new months can judge it.
RULES = [
    {"name": "iron-fly-1115", "kind": "fly", "enter": "11:15", "width": 0.01,
     "tickers": ["SPY", "QQQ"], "what": "Every day at 11:15: sell the at-the-money call and put of the nearest "
     "expiry, and buy a call 1% above and a put 1% below the stock price as a safety net; close all four at "
     "the close. Wins when the stock stays calm; the worst possible loss is fixed when it opens (about $400-550). "
     "No proven edge: running to test the live system with a capped loss."},
    {"name": "put-spread-monthly", "kind": "put_spread", "enter": "15:45", "sell_at": 0.95, "buy_at": 0.90,
     "tickers": ["SPY", "QQQ", "IWM"], "what": "Once a month, on the first trading day after the monthly options "
     "expire (3rd Friday), at the close: sell the put about 5% below the stock and buy the put about 10% below, both "
     "expiring next month, and hold them to expiry. Wins when the stock doesn't fall more than ~5% in the month; the "
     "worst loss is the gap between the strikes minus the credit. Backtest 2013-2026: +$42/trade, 91% won, t 2.2 "
     "(short of the 2.6 needed): tested unchanged on new months (HYPOTHESES.md, H4)."},
]
HOLD_TO_EXPIRY = {r["name"] for r in RULES if r["kind"] == "put_spread"}
ENTRY_FIRST, ENTRY_LAST, CLOSE_FROM = "09:45", "15:30", "15:45"
MAX_PREMIUM_PER_DAY = 1000.0    # user's cap (2026-10-05): net premium per rule per day, whole positions only


def load_snapshots():
    parts = []
    for f in sorted(glob.glob(str(SNAPS / "*" / "*.csv.gz"))):
        d = pd.read_csv(f, usecols=["snapshot_time", "ticker", "spot", "expiration", "type", "strike", "bid", "ask"])
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    q = pd.concat(parts, ignore_index=True)
    t = q["snapshot_time"].str.slice(0, 19)
    q["snap"] = pd.to_datetime(t).dt.tz_localize(NY, ambiguous="NaT", nonexistent="NaT")
    q = q.dropna(subset=["snap"])
    q["date"] = q["snap"].dt.strftime("%Y-%m-%d")
    q["hhmm"] = q["snap"].dt.strftime("%H:%M")
    q["bid"] = q["bid"].fillna(0)
    q["mid"] = (q["bid"] + q["ask"]) / 2
    return q


def parity_spots(q):
    """Stock price at each snapshot from the nearest expiry: S = K + C - P at the strike where C and P are closest."""
    live = q[(q["bid"] > 0) & (q["ask"] > q["bid"])]
    near = live[live["expiration"] == live.groupby(["snap", "ticker"])["expiration"].transform("min")]
    pc = near.pivot_table(index=["snap", "ticker", "strike"], columns="type", values="mid").dropna().reset_index()
    pc["gap"] = (pc["call"] - pc["put"]).abs()
    pc = pc.loc[pc.groupby(["snap", "ticker"])["gap"].idxmin()]
    pc["pspot"] = pc["strike"] + pc["call"] - pc["put"]
    return pc.set_index(["snap", "ticker"])["pspot"]


def atm(day_q, snap, ticker, kind, spot):
    """The at-the-money contract of the nearest expiry with a live two-sided quote, at one snapshot."""
    g = day_q[(day_q["snap"] == snap) & (day_q["ticker"] == ticker) & (day_q["type"] == kind)
              & (day_q["bid"] > 0) & (day_q["ask"] > day_q["bid"])]
    if g.empty:
        return None
    g = g[g["expiration"] == g["expiration"].min()]
    return g.loc[(g["strike"] - spot).abs().idxmin()]


def entries(q, spots):
    """Every entry the rules would have made, decided only from snapshots up to that moment."""
    fed = set(pd.read_csv(ROOT / "data" / "events" / "macro.csv").query("type == 'fomc'")["reaction_day"])
    rows = []
    sp_all = spots.reset_index()
    sp_all["date"] = sp_all["snap"].dt.strftime("%Y-%m-%d")
    for (day, ticker), sp in sp_all.groupby(["date", "ticker"]):
        sp = sp.sort_values("snap").reset_index(drop=True)
        day_q = q[(q["date"] == day) & (q["ticker"] == ticker)]
        for rule in RULES:
            if ticker not in rule["tickers"]:
                continue
            if rule["kind"] == "move":
                for i in range(1, len(sp)):
                    now = sp.loc[i]
                    # compare with the latest snapshot at least ~look minutes back (extra runs a few minutes apart are skipped)
                    back = sp.loc[:i - 1][sp.loc[:i - 1, "snap"] <= now["snap"] - pd.Timedelta(minutes=0.6 * rule["look_min"])]
                    if back.empty:
                        continue
                    before = back.iloc[-1]
                    gap = (now["snap"] - before["snap"]).total_seconds() / 60
                    hhmm = now["snap"].strftime("%H:%M")
                    if not (0.6 * rule["look_min"] <= gap <= 1.7 * rule["look_min"]) or not (ENTRY_FIRST <= hhmm <= ENTRY_LAST):
                        continue
                    move = now["pspot"] / before["pspot"] - 1
                    if abs(move) <= rule["thr"]:
                        continue
                    up = move > 0
                    kind = ("call" if up else "put") if rule["style"] == "momentum" else ("put" if up else "call")
                    c = atm(day_q, now["snap"], ticker, kind, now["pspot"])
                    if c is not None:
                        rows.append({"rule": rule["name"], "why": f"{ticker} {move:+.2%} in {gap:.0f} min", **_entry(c, now, "buy")})
            elif rule["kind"] == "time" or (rule["kind"] == "fed" and day in fed):
                start = pd.Timestamp(f"{day} {rule['enter']}", tz=NY)
                first = sp[(sp["snap"] >= start) & (sp["snap"] <= start + pd.Timedelta(minutes=30))].head(1)
                if first.empty:
                    continue
                now = first.iloc[0]
                why = "Fed decision day" if rule["kind"] == "fed" else f"daily at {rule['enter']}"
                for kind in ("call", "put"):
                    c = atm(day_q, now["snap"], ticker, kind, now["pspot"])
                    if c is not None:
                        rows.append({"rule": rule["name"], "why": why, **_entry(c, now, rule.get("side", "buy"))})
            elif rule["kind"] in ("fly", "put_spread"):
                if rule["kind"] == "put_spread" and day != spread_day(day):
                    continue
                start = pd.Timestamp(f"{day} {rule['enter']}", tz=NY)
                first = sp[(sp["snap"] >= start) & (sp["snap"] <= start + pd.Timedelta(minutes=30))].head(1)
                if not first.empty:
                    rows += (iron_fly if rule["kind"] == "fly" else put_spread)(day_q, first.iloc[0], rule)
    return premium_cap(pd.DataFrame(rows))


def iron_fly(day_q, now, rule):
    """Four legs at one snapshot, nearest expiry: short call + put at the strike nearest the stock (both live),
    long call at the first strike >= stock x (1 + width), long put at the last strike <= stock x (1 - width)."""
    g = day_q[(day_q["snap"] == now["snap"]) & (day_q["ask"] > 0) & (day_q["ask"] >= day_q["bid"])]
    live = g[(g["bid"] > 0) & (g["ask"] > g["bid"])]
    exp = live["expiration"].min()
    g, live = g[g["expiration"] == exp], live[live["expiration"] == exp]
    # the options we sell need a bid; the wings we buy only need an ask (far-away ones often have no bid)
    calls, puts = live[live["type"] == "call"].set_index("strike"), live[live["type"] == "put"].set_index("strike")
    both = calls.index.intersection(puts.index)
    s, w = now["pspot"], rule["width"]
    buy_c, buy_p = g[g["type"] == "call"].set_index("strike"), g[g["type"] == "put"].set_index("strike")
    wing_c, wing_p = buy_c[buy_c.index >= s * (1 + w)], buy_p[buy_p.index <= s * (1 - w)]
    if both.empty or wing_c.empty or wing_p.empty:
        return []
    k0 = min(both, key=lambda k: abs(k - s))
    kc, kp = wing_c.index.min(), wing_p.index.max()
    legs = [(calls.loc[k0], "sell"), (puts.loc[k0], "sell"), (buy_c.loc[kc], "buy"), (buy_p.loc[kp], "buy")]
    credit = calls.loc[k0, "bid"] + puts.loc[k0, "bid"] - buy_c.loc[kc, "ask"] - buy_p.loc[kp, "ask"]
    max_loss = (max(kc - k0, k0 - kp) - credit) * 100 + 8 * COMM
    why = f"daily at {rule['enter']} · net credit ${credit * 100:,.0f} · max loss ${max_loss:,.0f}"
    return [{"rule": rule["name"], "why": why, **_entry(pd.Series({**c.to_dict(), "strike": k}), now, side)}
            for (c, side), k in zip(legs, [k0, k0, kc, kp])]


def third_friday(year, month):
    first = pd.Timestamp(year=year, month=month, day=1)
    return first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)


@functools.lru_cache(maxsize=None)
def _spread_day_for_month(year, month):
    """Building the NYSE schedule costs about a fifth of a second and the answer
    depends only on the month, but this was called once per trading day — about
    twenty seconds of a four-minute run spent recomputing two distinct values."""
    import pandas_market_calendars as mcal
    fri = third_friday(year, month)
    sched = mcal.get_calendar("NYSE").schedule(start_date=fri + pd.Timedelta(days=1),
                                               end_date=fri + pd.Timedelta(days=10))
    return sched.index[0].strftime("%Y-%m-%d")


def spread_day(day):
    """The put spread's entry day in day's month: the first NYSE trading day after the 3rd Friday."""
    return _spread_day_for_month(int(day[:4]), int(day[5:7]))


def put_spread(day_q, now, rule):
    """Sell the put nearest stock x sell_at, buy the put nearest stock x buy_at (strictly lower), next month's
    standard expiry (3rd Friday, or the day before when that Friday is a holiday). Held to expiry."""
    d = pd.Timestamp(now["snap"].strftime("%Y-%m-%d"))
    nxt = third_friday(d.year + d.month // 12, d.month % 12 + 1)
    g = day_q[(day_q["snap"] == now["snap"]) & (day_q["type"] == "put")]
    exp = next((e for e in [nxt.strftime("%Y-%m-%d"), (nxt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")]
                if e in set(g["expiration"])), None)
    if exp is None:
        return []
    g, s = g[g["expiration"] == exp], now["pspot"]
    sell = g[g["bid"] > 0]
    if sell.empty:
        return []
    k1 = sell.loc[(sell["strike"] - rule["sell_at"] * s).abs().idxmin()]
    buy = g[(g["strike"] < k1["strike"]) & (g["ask"] > 0)]
    if buy.empty:
        return []
    k2 = buy.loc[(buy["strike"] - rule["buy_at"] * s).abs().idxmin()]
    credit = k1["bid"] - k2["ask"]
    max_loss = (k1["strike"] - k2["strike"] - credit) * 100 + 4 * COMM
    why = f"monthly at the close · net credit ${credit * 100:,.0f} · max loss ${max_loss:,.0f} · expires {exp}"
    return [{"rule": rule["name"], "why": why, **_entry(c, now, side)} for c, side in [(k1, "sell"), (k2, "buy")]]


def premium_cap(e):
    """Open positions in time order (SPY before QQQ at the same moment) while each rule's net premium that
    day stays within MAX_PREMIUM_PER_DAY; a whole position that would go over is skipped."""
    if e.empty:
        return e
    order = {t: i for i, t in enumerate(dict.fromkeys(t for r in RULES for t in r["tickers"]))}
    e = e.assign(_o=e["ticker"].map(order)).sort_values(["entry_time", "_o", "type"])
    keep, used = [], {}
    for (day, _, ticker, rule), pos in e.groupby(["date", "entry_time", "ticker", "rule"], sort=False):
        sold = pos[pos["side"] == "sell"]
        # user's rule (2026-10-07): losses are always capped -- every option sold needs one bought of the same type
        bought = pos[pos["side"] == "buy"]
        if any((sold["type"] == t).sum() > (bought["type"] == t).sum() for t in ("call", "put")):
            print(f"{day} {ticker} {rule}: skipped, sold options without bought protection (loss not capped)")
            continue
        # premium = the net credit for covered positions (sold minus bought; the bought legs cap the loss),
        # what is paid for buy-only ones
        prem = float(abs((sold["entry_price"].sum() - bought["entry_price"].sum()) if len(sold) else bought["entry_price"].sum()) * 100)
        if used.get((day, rule), 0.0) + prem <= MAX_PREMIUM_PER_DAY:
            used[(day, rule)] = used.get((day, rule), 0.0) + prem
            keep.append(pos)
        else:
            print(f"{day} {ticker} {rule}: skipped, ${prem:,.0f} premium would pass the ${MAX_PREMIUM_PER_DAY:,.0f} daily cap")
    return pd.concat(keep).drop(columns="_o") if keep else e.iloc[:0].drop(columns="_o")


def _entry(c, now, side):
    """Buying pays the ask; selling gets the bid."""
    return {"date": c["date"], "ticker": c["ticker"], "type": c["type"], "strike": float(c["strike"]),
            "expiration": c["expiration"], "side": side, "entry_time": now["snap"],
            "entry_price": float(c["ask"] if side == "buy" else c["bid"]), "entry_mid": float(c["mid"]),
            "spot": round(float(now["pspot"]), 2)}


def settle(trades, q, now=None):
    """Exit each trade at the close (or value it at the latest quote while the day is still running).
    A bought option is sold at the bid (no bid = $0); a sold one is bought back at the ask."""
    if trades.empty:
        return trades
    key = ["date", "ticker", "type", "strike", "expiration"]
    book = q[key + ["snap", "bid", "ask"]].sort_values("snap")
    # One lookup instead of a merge per trade. Each of those merges factorized
    # all of `book` — millions of rows — to pull out a single contract, which
    # was about eighty seconds of a four-minute run.
    #
    # `.indices` maps each key to row positions without building a frame per
    # group: there are ~170k contracts in the book and only a couple of hundred
    # trades, so the rows are sliced out on demand. Positions keep `book`'s
    # order, which is sorted by snap, so `.iloc[-1]` still means the latest
    # quote.
    by_contract = book.groupby(key[1:], sort=False).indices
    by_trade = book.groupby(key, sort=False).indices
    no_quotes = book.iloc[0:0]
    rows = lambda idx: book.take(idx) if idx is not None else no_quotes
    trades = trades.reset_index(drop=True)
    now = now if now is not None else pd.Timestamp.now(tz=NY)
    # a day is over once it's past 4:05pm New York (a missed closing run then closes at the day's last quote)
    today = now.strftime("%Y-%m-%d") if now.strftime("%H:%M") < "16:05" else (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    pos = ["date", "entry_time", "ticker", "rule"]
    short_k = trades[trades["side"] == "sell"].groupby(pos)["strike"].max()
    spot_at = q.sort_values("snap").groupby(["ticker", "date"]).tail(1).set_index(["ticker", "date"])
    out = []
    for _, t in trades.iterrows():
        if t["rule"] in HOLD_TO_EXPIRY:          # held to expiry: marked at the latest quote until then
            if now >= pd.Timestamp(f"{t['expiration']} 16:05", tz=NY):
                out.append(expire(t, spot_at, short_k.get(tuple(t[pos]), 0.0)))
                continue
            quotes = rows(by_contract.get(tuple(t[key[1:]])))
            later = quotes[quotes["snap"] > t["entry_time"]]
            x, status = (later.iloc[-1] if not later.empty else None), "open"
        else:
            quotes = rows(by_trade.get(tuple(t[key])))
            later = quotes[quotes["snap"] > t["entry_time"]]
            closing = later[later["snap"].dt.strftime("%H:%M") >= CLOSE_FROM]
            if not closing.empty:
                x, status = closing.iloc[-1], "closed"
            elif t["date"] < today and not later.empty:
                x, status = later.iloc[-1], "late close"
            elif t["date"] < today:
                x, status = None, "no exit quote"
            else:
                x, status = (later.iloc[-1] if not later.empty else None), "open"
        if t["side"] == "buy":
            price = float(x["bid"]) if x is not None and pd.notna(x["bid"]) else 0.0     # no bid = worth $0
            pnl = (price - t["entry_price"]) * 100 - 2 * COMM
        else:   # no quote to buy back at: assume the worst seen so far (the last ask, else the entry ask)
            asks = later["ask"].dropna() if not later.empty else pd.Series(dtype=float)
            price = float(x["ask"]) if x is not None and pd.notna(x["ask"]) else float(asks.iloc[-1]) if len(asks) else t["entry_price"]
            pnl = (t["entry_price"] - price) * 100 - 2 * COMM
        out.append({**t.to_dict(), "exit_time": x["snap"] if x is not None else pd.NaT, "exit_price": price,
                    "status": status, "pnl": round(pnl, 2), "premium": round(t["entry_price"] * 100, 2)})
    return pd.DataFrame(out)


def expire(t, spot_at, short_strike):
    """Settle a held-to-expiry put at its value on expiry day, from the stock's last price that day.
    If the sold put ends in the money, both legs also pay a commission to close (as in the backtest)."""
    last = spot_at[(spot_at.index.get_level_values(0) == t["ticker"]) & (spot_at.index.get_level_values(1) <= t["expiration"])]
    if last.empty:
        return {**t.to_dict(), "exit_time": pd.NaT, "exit_price": np.nan, "status": "no exit quote", "pnl": np.nan,
                "premium": round(t["entry_price"] * 100, 2)}
    x = last.iloc[-1]
    S = float(x["spot"])
    value = max(t["strike"] - S, 0.0)
    extra = COMM if S < short_strike else 0.0
    sign = 1 if t["side"] == "buy" else -1
    return {**t.to_dict(), "exit_time": x["snap"], "exit_price": value,
            "status": "expired" if last.index[-1][1] == t["expiration"] else "expired (late)",
            "pnl": round(sign * (value - t["entry_price"]) * 100 - COMM - extra, 2), "premium": round(t["entry_price"] * 100, 2)}


def health(q):
    if q.empty:
        return {}
    last = q["snap"].max()
    lq = q[q["snap"] == last]
    per_day = q.groupby("date")["snap"].nunique().tail(15)
    return {"last_snapshot": last.isoformat(), "quotes": int(len(lq)),
            "live_share": round(float(((lq["bid"] > 0) & (lq["ask"] > 0)).mean()), 3),
            "tickers": sorted(lq["ticker"].unique().tolist()),
            "snapshots_by_day": {d: int(n) for d, n in per_day.items()},
            "expected_per_day": 26}


def plan(ticker, day, now, trades):
    """What the rules will do next for this ticker, in words (positions being held first)."""
    rules = [r for r in RULES if ticker in r["tickers"]]
    if not rules:
        return "watch only"
    held, rest = [], []
    for r in rules:
        mine = trades[(trades["ticker"] == ticker) & (trades["rule"] == r["name"])] if not trades.empty else trades
        if r["kind"] == "put_spread":
            open_ = mine[mine["status"] == "open"] if len(mine) else mine
            if len(open_):
                held.append(f"holding the put spread to {open_['expiration'].iloc[0]}")
                continue
            nxt = spread_day(day)
            if nxt < day or (nxt == day and now.strftime("%H:%M") > r["enter"]):
                m = pd.Timestamp(day) + pd.offsets.MonthBegin(1)
                nxt = spread_day(m.strftime("%Y-%m-%d"))
            rest.append(f"put spread {'today' if nxt == day else pd.Timestamp(nxt).strftime('%b %-d')} at {r['enter']}")
            continue
        today = mine[mine["date"] == day] if len(mine) else mine
        if len(today) and (today["status"] == "open").any():
            held.append("holding: closes at 3:45-4pm")
        elif len(today):
            rest.append("done for today")
        elif r["kind"] in ("time", "fly") and now.strftime("%H:%M") < r["enter"]:
            rest.append(f"opens the iron fly at {r['enter']}" if r["kind"] == "fly" else f"{r['side']}s the straddle at {r['enter']}")
        else:
            rest.append("no trade today")
    return " · ".join(held + rest)


def signals(spots, trades):
    """Latest move per ticker between two checks, and what the rules will do next."""
    if spots.empty:
        return []
    s = spots.reset_index()
    s["date"] = s["snap"].dt.strftime("%Y-%m-%d")
    s = s[s["date"] == s["date"].max()].sort_values("snap")
    rows = []
    for ticker, g in s.groupby("ticker"):
        g = g.reset_index(drop=True)
        now = g.iloc[-1]
        move, gap = None, None
        if len(g) > 1:
            gap = (now["snap"] - g.iloc[-2]["snap"]).total_seconds() / 60
            move = float(now["pspot"] / g.iloc[-2]["pspot"] - 1)
        rows.append({"ticker": ticker, "time": now["snap"].isoformat(), "spot": round(float(now["pspot"]), 2),
                     "move": move, "gap_min": gap, "day_move": float(now["pspot"] / g.iloc[0]["pspot"] - 1),
                     "traded": any(ticker in r["tickers"] for r in RULES),
                     "plan": plan(ticker, now["snap"].strftime("%Y-%m-%d"), now["snap"], trades)})
    return sorted(rows, key=lambda r: -abs(r["move"] or 0))


def iso(x):
    return x.isoformat() if isinstance(x, pd.Timestamp) and not pd.isna(x) else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    q = load_snapshots()
    spots = parity_spots(q) if not q.empty else pd.Series(dtype=float)
    trades = settle(entries(q, spots), q) if not q.empty else pd.DataFrame()
    if not trades.empty:
        trades = trades.sort_values(["entry_time", "ticker", "type"]).reset_index(drop=True)
        trades.to_csv(OUT / "trades.csv", index=False)
    read = lambda name: json.loads((OUT / name).read_text()) if (OUT / name).exists() else None
    dash = {
        "updated": pd.Timestamp.now(tz=NY).isoformat(),
        "mode": "paper",
        "fees": f"buy at ask, sell at bid, IBKR ${COMM:.2f}/contract each way; at most ${MAX_PREMIUM_PER_DAY:,.0f} of net premium per rule per day; every loss capped by bought options",
        "rules": RULES,
        "trades": [{k: (iso(v) if isinstance(v, pd.Timestamp) else v) for k, v in r.items()}
                   for r in trades.to_dict("records")] if not trades.empty else [],
        "signals": signals(spots, trades),
        "health": health(q),
        "validation": read("validation.json"),
        "coverage": read("coverage.json"),
    }
    (OUT / "dashboard.json").write_text(json.dumps(dash, default=lambda v: None if pd.isna(v) else str(v)))
    n_open = int((trades["status"] == "open").sum()) if not trades.empty else 0
    print(f"{len(trades)} paper trades ({n_open} open), "
          f"realized ${trades.loc[trades['status'] != 'open', 'pnl'].sum() if not trades.empty else 0:,.2f}")


def coverage():
    """What data we hold (laptop only: the Databento files aren't on GitHub)."""
    db = ROOT / "data" / "databento"
    out = {"as_of": pd.Timestamp.now(tz=NY).strftime("%Y-%m-%d"), "daily": [], "intraday": None}
    import download_databento as dd
    for g, tickers in dd.GROUPS.items():
        days = sorted(p.stem for p in (db / "quotes" / g).rglob("*.parquet"))
        if days:
            out["daily"].append({"group": g, "tickers": tickers, "first": days[0], "last": days[-1], "days": len(days)})
    intra = sorted(Path(p).stem for p in glob.glob(str(db / "intraday" / "2*.parquet")))
    if intra:
        qc = pd.read_csv(db / "intraday" / "quality.csv")
        out["intraday"] = {"tickers": ["SPY", "QQQ"], "first": intra[0], "last": intra[-1], "days": len(intra),
                           "every": "15 min", "live_share": round(float(qc["live"].mean()), 3)}
    out["spent"] = round(dd.spent(), 2)
    (OUT / "coverage.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--coverage", action="store_true")
    coverage() if p.parse_args().coverage else main()
