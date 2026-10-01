"""
Intraday option quotes from Databento (official OPRA feed, cbbo-1m): SPY and QQQ options expiring
within 0-7 days, strikes within +-3% of the previous close, one snapshot every 15 minutes: the 1-minute quote ending at 9:45, 10:00, ... 16:00
New York time (early-close days handled). Last year only: 2025-09-29 .. 2026-09-28.

Approved by the user 2026-10-01 (~$11; the 1-day test priced the year at ~$12). Shares the spending ledger with download_databento.py
(counted x1.15); this downloader stops at BUDGET below.
Saved per day: data/databento/intraday/raw/<day>.parquet (the purchase, first) and
data/databento/intraday/<day>.parquet (tidy). Quality: data/databento/intraday/quality.csv

Run:  python download_intraday.py --days 1     (one day first, to check the cost)
      python download_intraday.py
"""
import argparse
import fcntl
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pandas_market_calendars as mcal

import download_databento as dd

TICKERS = ["SPY", "QQQ"]
FIRST, LAST = "2025-09-29", dd.LAST_DAY
STEP_MIN, MAX_DTE, STRIKE_RANGE = 15, 7, 0.03
dd.BUDGET = 106.0          # the other downloads stop at $95; this one may add the approved ~$11 on top
CAP = 12.0                 # this downloader's own limit (what the user approved), counted x1.15 like the ledger
OUT = dd.OUT / "intraday"


def snapshot_times(open_utc, close_utc):
    t = open_utc + pd.Timedelta(minutes=STEP_MIN)
    times = []
    while t <= close_utc:
        times.append(t)
        t += pd.Timedelta(minutes=STEP_MIN)
    return times


def one_day(day, open_utc, close_utc, prev_close):
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    defs = dd.buy_definitions(TICKERS, day, nxt, f"intraday definitions {day}")      # same-day list: new daily expiries
    defs["dte"] = (pd.to_datetime(defs["expiration"]) - pd.Timestamp(day)).dt.days
    defs["m"] = defs["strike"] / defs["ticker"].map(prev_close)
    want = defs[defs["dte"].between(0, MAX_DTE) & defs["m"].between(1 - STRIKE_RANGE, 1 + STRIKE_RANGE)]
    syms = want["raw_symbol"].tolist()
    chunks = [syms[i:i + dd.CHUNK] for i in range(0, len(syms), dd.CHUNK)]
    jobs = [(t, c) for t in snapshot_times(open_utc, close_utc) for c in chunks]
    bill = []

    def fetch(job):
        t, c = job
        q = dd.get("cbbo-1m", c, "raw_symbol", t - pd.Timedelta(minutes=1), t, f"intraday {day} {t:%H:%M}", bill)
        return q.reset_index().assign(snap=t) if len(q) else None

    with ThreadPoolExecutor(8) as pool:
        parts = [p for p in pool.map(fetch, jobs) if p is not None]
    raw = pd.concat(parts, ignore_index=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    raw.to_parquet(OUT / "raw" / f"{day}.parquet", compression="zstd")              # keep the purchase first
    raw = raw.sort_values("ts_event").drop_duplicates(["snap", "symbol"], keep="last")
    out = want.merge(pd.DataFrame({"raw_symbol": raw["symbol"].values, "snap": raw["snap"].values,
                                   "bid": raw["bid_px_00"].values, "ask": raw["ask_px_00"].values,
                                   "bid_size": raw["bid_sz_00"].values, "ask_size": raw["ask_sz_00"].values,
                                   "last": raw["price"].values, "ts_event": raw["ts_event"].values}), on="raw_symbol")
    out = out.assign(date=day)[["date", "snap", "ticker", "type", "strike", "expiration", "bid", "ask",
                                "bid_size", "ask_size", "last", "ts_event"]]
    out.to_parquet(OUT / f"{day}.parquet", index=False, compression="zstd")
    live = out[(out["bid"] > 0) & (out["ask"] > 0)]
    qc = {"date": day, "snapshots": out["snap"].nunique(), "expected_snapshots": len(snapshot_times(open_utc, close_utc)),
          "contracts": len(syms), "rows": len(out), "live": round(len(live) / max(len(out), 1), 3),
          "crossed": int((live["bid"] > live["ask"]).sum()), "cost": round(sum(bill), 4)}
    pd.DataFrame([qc]).to_csv(OUT / "quality.csv", mode="a", index=False, header=not (OUT / "quality.csv").exists())
    return qc


def main(max_days):
    OUT.mkdir(parents=True, exist_ok=True)
    lock = open(OUT / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another intraday download is running -- exiting.")
    closes = pd.read_csv(dd.OUT / "underlying.csv").set_index(["date", "ticker"])["close"]
    sched = mcal.get_calendar("NYSE").schedule(start_date=FIRST, end_date=LAST)
    days = [d.strftime("%Y-%m-%d") for d in sched.index][::-1]                         # newest first
    all_days = sorted(closes.index.get_level_values("date").unique())
    done = 0
    for day in days:
        if (OUT / f"{day}.parquet").exists():
            continue
        if done >= max_days:
            break
        qf = OUT / "quality.csv"
        used = pd.read_csv(qf)["cost"].sum() * dd.SAFETY if qf.exists() else 0.0
        if used + 0.08 > CAP:
            print(f"Stopping at the approved limit: ${used:.2f} of ${CAP:.0f} used (newest days first).")
            break
        prev = all_days[all_days.index(day) - 1]
        t0 = time.time()
        row = sched.loc[day]
        qc = one_day(day, row["market_open"], row["market_close"], closes.xs(prev, level="date")[TICKERS])
        done += 1
        print(f"{day}: {qc['snapshots']}/{qc['expected_snapshots']} snapshots, {qc['contracts']} contracts, "
              f"{qc['live']:.0%} live, ${qc['cost']:.4f} (ledger ${dd.spent():.2f}), {time.time() - t0:.0f}s", flush=True)
    print(f"Done. Ledger ${dd.spent():.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=10**6)
    main(p.parse_args().days)
