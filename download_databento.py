"""
Download closing option quotes (bid/ask) from Databento's official OPRA feed,
using ONLY the free $125 credit, as efficiently as possible.

For each trading day (newest first, going back to 2013):
  - the last one-minute quote before the close (handles early-close days and daylight saving)
  - only the contracts we train on: out-of-the-money, strikes within +-20% of the stock,
    7 days to 1 year to expiry. That's ~5x cheaper than downloading every contract.
Contract lists ("definitions") are bought once a week and reused for that week.

Tickers come in groups, each stored separately (data/databento/quotes/<group>/...):
  core:  the original 8           extra: 6 more liquid names added later     more: XOM (energy)     wide: 25 more of the top 100
Meta's options traded as "FB" until 2022-06-09; that's handled automatically.

Every day is quality-checked (coverage, bid <= ask, spreads) into data/databento/quality_<group>.csv.
Spending (shared by all groups) is tracked in data/databento/spent.json and it stops before BUDGET.
Safe to stop and re-run: finished days are skipped.

Run:  python download_databento.py --group core     (add --days 5 to try a few days first)
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import databento as db
import requests
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

GROUPS = {"core": ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA"],
          "extra": ["META", "GOOGL", "AMD", "GLD", "TLT", "XLF"],
          "more": ["XOM"],
          # 25 long-history names from the top 100 option underlyings (snapshot_yahoo.py), added 2026-09-30
          "wide": ["INTC", "MU", "NFLX", "ORCL", "AVGO", "NOK", "TSM", "NOW", "BAC", "WMT", "NKE", "F", "PFE",
                   "AAL", "UNH", "QCOM", "CRM", "BA", "T", "MRVL", "PYPL", "BABA", "GME", "MSTR", "SMCI"]}
ALL_TICKERS = [t for g in GROUPS.values() for t in g]
OLD_ROOTS = {"META": ("FB", "2022-06-09")}   # options root before that date (Facebook -> Meta rename)
TICKERS = GROUPS["core"]                     # set by --group
GROUP = "core"
DATASET = "OPRA.PILLAR"
FIRST_DAY = "2013-04-01"      # OPRA history on Databento starts here
LAST_DAY = "2026-09-28"       # history ends here: newer days need a paid live license (403); Yahoo covers them
BUDGET = 106.0                # all downloads together stop at $106 of usage = keep >= ~$19 of the $125 credit
                              # (user OK'd: resume 'wide' keeping $30, plus ~$11-12 intraday from that reserve)
SAFETY = 1.25                 # the real bill runs ~15% above get_cost quotes; count 25% higher to stay on the safe side
# Databento's bill (portal, 2026-09-30: $55.76) matches our quote purchases only -- contract lists
# ("definition" schema) are quoted by get_cost but not billed, so they're tracked separately.
ALREADY_SPENT = 0.20          # the tests we ran before this script existed
STRIKE_RANGE = 0.20
MIN_DAYS, MAX_DAYS = 7, 365
CHUNK = 500                   # symbols per request (max 2,000; big requests time out at the gateway -- 504s)
DAYS_AT_ONCE = 8              # download several days in parallel
OUT = Path(__file__).parent / "data" / "databento"


def load_key():
    for line in (Path(__file__).parent / ".env").read_text().splitlines():
        if line.startswith("DATABENTO_API_KEY="):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit("No DATABENTO_API_KEY in .env")

KEY = load_key()
_local = threading.local()

def client():
    """One connection per worker thread (sharing one across threads breaks SSL)."""
    if not hasattr(_local, "client"):
        _local.client = db.Historical(KEY)
    return _local.client


# ---------- budget: estimate the cost BEFORE every purchase, never go over ----------
_lock = threading.Lock()
LEDGER = OUT / "spent.json"

def spent():
    return json.loads(LEDGER.read_text())["spent"] if LEDGER.exists() else ALREADY_SPENT

def charge(cost, what, billed=True):
    import fcntl
    with _lock, open(LEDGER.with_suffix(".lock"), "w") as fl:
        fcntl.flock(fl, fcntl.LOCK_EX)              # several downloaders may share the ledger
        led = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"spent": ALREADY_SPENT}
        if not billed:
            led["definitions_quoted_not_billed"] = round(led.get("definitions_quoted_not_billed", 0) + cost, 4)
        elif led["spent"] + cost * SAFETY > BUDGET:
            raise SystemExit(f"Budget reached: ${led['spent']:.2f} spent, next item ({what}) costs ${cost:.4f}. Stopping.")
        else:
            led["spent"] = round(led["spent"] + cost * SAFETY, 4)
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        LEDGER.write_text(json.dumps(led))

def get(schema, symbols, stype, start, end, what, bill=None):
    """Cost check, then download. Retries if Databento's gateway times out."""
    kw = dict(dataset=DATASET, symbols=symbols, stype_in=stype, schema=schema, start=start, end=end)
    cost = retry(lambda **k: client().metadata.get_cost(**k), **kw)
    charge(cost, what, billed=schema != "definition")
    if bill is not None:
        bill.append(cost)
    return retry(lambda **k: client().timeseries.get_range(**k).to_df(), **kw)

def retry(f, **kw):
    """Databento's gateway sometimes times out; wait and try again."""
    for attempt in range(5):
        try:
            return f(**kw)
        except (db.BentoServerError, db.BentoClientError, db.BentoError, requests.exceptions.RequestException) as e:
            if isinstance(e, db.BentoClientError) and getattr(e, "http_status", None) != 429:
                raise                                   # a real mistake in our request: don't retry
            print(f"  network/server error ({str(e)[:120]}), retrying...", flush=True)
            _local.__dict__.pop("client", None)          # fresh connection next time
            time.sleep(15 * (attempt + 1))
    raise RuntimeError("Databento kept failing")


# ---------- stock closing prices (free, from Yahoo), un-adjusted for splits ----------
def stock_closes():
    """Real traded closes (option strikes are in these units, so undo split adjustments)."""
    rows = []
    for t in ALL_TICKERS:
        tk = yf.Ticker(t)
        h = tk.history(start=FIRST_DAY, auto_adjust=False)["Close"]
        h.index = h.index.tz_localize(None).normalize()
        factor = pd.Series(1.0, index=h.index)
        for day, ratio in tk.splits.items():
            factor[h.index < day.tz_localize(None).normalize()] *= ratio   # split day itself already trades split
        rows.append(pd.DataFrame({"ticker": t, "date": h.index.strftime("%Y-%m-%d"),
                                  "close": (h * factor).round(4)}))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(OUT / "underlying.csv", index=False)
    return df.set_index(["ticker", "date"])["close"]


# ---------- contract lists, bought once a week ----------
_def_lock = threading.Lock()

def definitions(day):
    with _def_lock:
        return _definitions(day)

def root(ticker, day):
    """The options' trading symbol on that day (e.g. Meta's were "FB" before 2022-06-09)."""
    old = OLD_ROOTS.get(ticker)
    return old[0] if old and day < old[1] else ticker


def buy_definitions(tickers, day, nxt, what):
    """Contract list for these tickers on this day, standard contracts only, with our ticker names."""
    roots = {root(t, day): t for t in tickers}
    # one request per ticker: many tickers in one request exceed Databento's 60-second gateway limit (504)
    def one(r):
        try:
            return get("definition", [f"{r}.OPT"], "parent", day, nxt, f"{what} {r}")
        except db.BentoClientError as e:                  # not listed yet on that day (e.g. PYPL before 2015)
            if "symbology" not in str(e):
                raise
            return pd.DataFrame()
    with ThreadPoolExecutor(5) as pool:
        parts = [p for p in pool.map(one, roots) if len(p)]
    if not parts:
        return pd.DataFrame(columns=["raw_symbol", "ticker", "type", "strike", "expiration"])
    d = pd.concat(parts).drop_duplicates("raw_symbol")
    d = pd.DataFrame({"raw_symbol": d["raw_symbol"].values,
                      "ticker": d["underlying"].values,
                      "type": np.where(d["instrument_class"].values == "C", "call", "put"),
                      "strike": d["strike_price"].values,
                      "expiration": pd.to_datetime(d["expiration"]).dt.strftime("%Y-%m-%d").values})
    # keep only standard contracts (adjusted ones after corporate actions have odd roots like "AAPL1")
    d = d[d["raw_symbol"].str[:6].str.strip() == d["ticker"]]
    d["ticker"] = d["ticker"].map(roots)             # "FB" -> "META"
    return d.dropna(subset=["ticker"])


def _definitions(day):
    path = OUT / "definitions" / GROUP / f"{day}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    d = buy_definitions(TICKERS, day, nxt, f"definitions {GROUP} {day}")
    path.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(path, index=False, compression="zstd")
    return d


# ---------- turn Databento's raw rows into our clean table ----------
COLS = ["date", "ticker", "type", "strike", "expiration", "spot",
        "bid", "ask", "bid_size", "ask_size", "last", "last_trade_age_s"]

def tidy(day, want, q, end):
    if q.empty:                                   # no quotes at all in that minute (e.g. a few tiny contracts)
        return pd.DataFrame(columns=COLS)
    q = q.reset_index().drop_duplicates("symbol", keep="last")
    # seconds between the last trade and the snapshot (blank = no trade that day)
    age = (end - q["ts_event"]).dt.total_seconds()
    out = want.merge(pd.DataFrame({"raw_symbol": q["symbol"].values,
                                   "bid": q["bid_px_00"].values, "ask": q["ask_px_00"].values,
                                   "bid_size": q["bid_sz_00"].values, "ask_size": q["ask_sz_00"].values,
                                   "last": q["price"].values,
                                   "last_trade_age_s": age.round().astype("Int32").values}),
                     on="raw_symbol")
    return out.assign(date=day)[COLS]


# ---------- one trading day ----------
def download_day(day, close_utc, defs, spot):
    d = defs[defs["ticker"].isin(spot.index)].copy()
    d["spot"] = d["ticker"].map(spot)
    d["m"] = d["strike"] / d["spot"]
    d["days"] = (pd.to_datetime(d["expiration"]) - pd.Timestamp(day)).dt.days
    otm = ((d["type"] == "call") & (d["m"] > 1)) | ((d["type"] == "put") & (d["m"] < 1))
    want = d[otm & d["m"].between(1 - STRIKE_RANGE, 1 + STRIKE_RANGE) & d["days"].between(MIN_DAYS, MAX_DAYS)]
    syms = want["raw_symbol"].tolist()
    start, end = close_utc - pd.Timedelta(minutes=1), close_utc

    bill = []
    chunks = [syms[i:i + CHUNK] for i in range(0, len(syms), CHUNK)]
    with ThreadPoolExecutor(4) as pool:
        parts = list(pool.map(lambda s: get("cbbo-1m", s, "raw_symbol", start, end, f"quotes {day}", bill), chunks))
    q = pd.concat(parts) if parts else pd.DataFrame()
    # keep the raw purchase first, so a bug below never means paying twice
    raw = OUT / "raw" / GROUP / f"{day}.parquet"
    raw.parent.mkdir(parents=True, exist_ok=True)
    q.to_parquet(raw, compression="zstd")
    out = tidy(day, want, q, end)
    path = quote_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    for col in ("date", "ticker", "type", "expiration"):
        out[col] = out[col].astype("category")
    out.to_parquet(path, index=False, compression="zstd")

    # ----- quality check for this day -----
    live = out[(out["bid"] > 0) & (out["ask"] > 0)]
    qc = {"date": day, "requested": len(syms), "received": len(out),
          "coverage": round(len(out) / max(len(syms), 1), 3),
          "live_quotes": round(len(live) / max(len(out), 1), 3),
          "crossed": int((live["bid"] > live["ask"]).sum()),
          "median_spread": round(((live["ask"] - live["bid"]) / ((live["ask"] + live["bid"]) / 2)).median(), 4),
          "tickers": out["ticker"].nunique(), "cost": round(sum(bill), 4)}
    flags = [msg for bad, msg in [(qc["coverage"] < 0.9, "low coverage"),
                                  (qc["live_quotes"] < 0.8, "many empty quotes"),
                                  (qc["crossed"] > 0, "crossed quotes"),
                                  (qc["tickers"] < spot.size, "missing tickers")] if bad]
    qc["flags"] = "; ".join(flags)
    qfile = OUT / f"quality_{GROUP}.csv"
    with _lock:
        pd.DataFrame([qc]).to_csv(qfile, mode="a", index=False, header=not qfile.exists())
    return qc


def quote_path(day, group=None):
    return OUT / "quotes" / (group or GROUP) / day[:4] / f"{day}.parquet"


def migrate_old_layout():
    """Files from before ticker groups existed belong to "core": move them into the new folders."""
    moves = [(p, OUT / "quotes" / "core" / p.parent.name / p.name) for p in (OUT / "quotes").glob("[0-9]*/*.parquet")]
    moves += [(p, OUT / "definitions" / "core" / p.name) for p in (OUT / "definitions").glob("*.parquet")]
    moves += [(p, OUT / "raw" / "core" / p.name) for p in (OUT / "raw").glob("*.parquet")]
    for name in ("quality.csv", "COMPLETE", "REPAIRED"):
        if (OUT / name).exists():
            moves.append((OUT / name, OUT / name.replace(".csv", "_core.csv") if name.endswith(".csv") else OUT / f"{name}_core"))
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
    for d in (OUT / "quotes").glob("[0-9]*"):
        d.rmdir()
    if moves:
        print(f"Moved {len(moves):,} files from the old layout into the 'core' group.")


def main(max_days, group):
    global GROUP, TICKERS
    GROUP, TICKERS = group, GROUPS[group]
    OUT.mkdir(parents=True, exist_ok=True)
    # never run two copies at once (they would buy the same days twice)
    import fcntl
    main.lock = open(OUT / ".lock", "w")
    try:
        fcntl.flock(main.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another download is already running -- exiting.")
    migrate_old_layout()
    print(f"Group '{GROUP}': {', '.join(TICKERS)}. Stock closes from Yahoo...")
    closes = stock_closes()

    yesterday = (pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    sched = mcal.get_calendar("NYSE").schedule(start_date=FIRST_DAY, end_date=min(yesterday, LAST_DAY))
    sched["day"] = sched.index.strftime("%Y-%m-%d")
    sched["week"] = sched.index.to_period("W").astype(str)
    first_of_week = sched.groupby("week")["day"].transform("min")

    todo = []
    for (_, row), week_start in zip(sched[::-1].iterrows(), first_of_week[::-1]):   # newest first
        day = row["day"]
        if quote_path(day).exists():
            continue
        if len(todo) >= max_days:
            break
        try:
            todo.append((day, row["market_close"], week_start, closes.xs(day, level="date")[TICKERS]))
        except KeyError:
            print(f"{day}: no stock prices, skipping")
    print(f"{len(todo):,} days to download")
    if not todo:
        (OUT / f"COMPLETE_{GROUP}").write_text(f"All days downloaded ({pd.Timestamp.now():%Y-%m-%d %H:%M})\n")

    def one(job):
        day, close_utc, week_start, spot = job
        t0 = time.time()
        try:
            qc = download_day(day, close_utc, definitions(week_start), spot)
        except SystemExit:
            raise
        except Exception as e:
            return print(f"{day}: FAILED ({str(e)[:150]}) -- will retry next run", flush=True)
        print(f"{day}: {qc['received']:,}/{qc['requested']:,} contracts, spread {qc['median_spread']:.1%}, "
              f"${qc['cost']:.4f} (total ${spent():.2f}), {time.time() - t0:.0f}s"
              + (f"  WARNING: {qc['flags']}" if qc["flags"] else ""), flush=True)

    with ThreadPoolExecutor(DAYS_AT_ONCE) as pool:
        list(pool.map(one, todo))
    print(f"Done. Total spent: ${spent():.2f} of ${BUDGET:.0f} budget.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=10**6, help="stop after this many days (for testing)")
    p.add_argument("--group", choices=list(GROUPS), default="core")
    a = p.parse_args()
    main(a.days, a.group)
