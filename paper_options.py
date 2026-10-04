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
stopped earlier, at its last snapshot (flagged "late close").

Writes data/paper_options/trades.csv and dashboard.json (read by the website's Options page).
Run:  python paper_options.py              (after snapshot_intraday.py; on GitHub: intraday.yml)
      python paper_options.py --coverage   (laptop only: refresh the data-coverage summary)
"""
import argparse
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

# Rules being paper-traded, picked by intraday_test.py on Databento history (see validation.json).
# The reversal and Fed-day rules were dropped on 2026-10-04: once options that end worthless were
# counted as $0 (the old test skipped them), both failed. The code for those kinds stays below.
RULES = [
    {"name": "sell-straddle-1115", "kind": "time", "side": "sell", "enter": "11:15",
     "tickers": ["SPY", "QQQ"], "what": "Every day at 11:15, sell the at-the-money call and put of the nearest "
     "expiry (a bet that the stock stays calm for the rest of the day); buy both back at the close. "
     "The only rule positive in all three checks so far, and only weakly."},
]
ENTRY_FIRST, ENTRY_LAST, CLOSE_FROM = "09:45", "15:30", "15:45"


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
    return pd.DataFrame(rows)


def _entry(c, now, side):
    """Buying pays the ask; selling gets the bid."""
    return {"date": c["date"], "ticker": c["ticker"], "type": c["type"], "strike": float(c["strike"]),
            "expiration": c["expiration"], "side": side, "entry_time": now["snap"],
            "entry_price": float(c["ask"] if side == "buy" else c["bid"]), "entry_mid": float(c["mid"]),
            "spot": round(float(now["pspot"]), 2)}


def settle(trades, q):
    """Exit each trade at the close (or value it at the latest quote while the day is still running).
    A bought option is sold at the bid (no bid = $0); a sold one is bought back at the ask."""
    if trades.empty:
        return trades
    key = ["date", "ticker", "type", "strike", "expiration"]
    book = q[key + ["snap", "bid", "ask"]].sort_values("snap")
    today = pd.Timestamp.now(tz=NY).strftime("%Y-%m-%d")
    out = []
    for _, t in trades.iterrows():
        quotes = book.merge(pd.DataFrame([t[key]]), on=key)
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
    """What the rules will do next for this ticker today, in words."""
    rules = [r for r in RULES if ticker in r["tickers"]]
    if not rules:
        return "watch only"
    mine = trades[(trades["ticker"] == ticker) & (trades["date"] == day)] if not trades.empty else trades
    if len(mine) and (mine["status"] == "open").any():
        return "holding: closes at 3:45-4pm"
    if len(mine):
        return "done for today"
    timed = [r for r in rules if r["kind"] == "time"]
    if timed and now.strftime("%H:%M") < timed[0]["enter"]:
        return f"{timed[0]['side']}s the straddle at {timed[0]['enter']}"
    return "no trade today"


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
        "fees": f"buy at ask, sell at bid, IBKR ${COMM:.2f}/contract each way",
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
