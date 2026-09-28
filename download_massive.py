"""
Download ~2 years of daily option prices from Massive (free plan) into data/.
Free plan = 5 API calls per minute, so this is slow. It saves as it goes and
picks up where it left off, so you can stop it and run it again any time.

What it downloads, for each ticker and each monthly expiration (3rd Friday):
  - up to 20 out-of-the-money puts and 20 out-of-the-money calls,
    chosen around the stock price ~30 days before expiration
  - each contract's daily open/high/low/close/volume (1 API call per contract)

Run:  python download_massive.py --minutes 60
(On GitHub this runs by itself every 6 hours -- see .github/workflows/download.yml.)
Needs MASSIVE_API_KEY in the environment or in a .env file.
"""
import argparse
import csv
import datetime as dt
import os
import time
from pathlib import Path

import requests

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]
STRIKES_PER_SIDE = 20        # puts below the price + calls above it, per expiration
STRIKE_RANGE = 0.20          # only strikes within +-20% of the price
CALL_GAP = 12.5              # seconds between API calls (5 per minute)
API = "https://api.massive.com"
DATA = Path(__file__).parent / "data"

TODAY = dt.date.today()
START = TODAY - dt.timedelta(days=2 * 365 - 5)   # free plan: last 2 years


# ---------- API key + a polite, rate-limited GET ----------
def load_key():
    if os.environ.get("MASSIVE_API_KEY"):
        return os.environ["MASSIVE_API_KEY"].strip()
    env = Path(__file__).parent / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("MASSIVE_API_KEY="):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit("No MASSIVE_API_KEY found (environment or .env).")

KEY = load_key()
_last_call = 0.0

def get(url, **params):
    """One API call. Waits so we never go over 5 calls/minute; retries if throttled."""
    global _last_call
    for attempt in range(5):
        wait = CALL_GAP - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        r = requests.get(url if url.startswith("http") else API + url,
                         params={**params, "apiKey": KEY}, timeout=60)
        if r.status_code == 429:           # too many calls: wait a minute and retry
            time.sleep(60)
            continue
        if r.status_code >= 500:           # server hiccup: retry
            time.sleep(30)
            continue
        if not r.ok:
            print(f"  API error {r.status_code}: {r.text[:200]}")
            return None
        return r.json()
    return None


# ---------- small CSV helpers (append as we go, so progress is never lost) ----------
def append_rows(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)

def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def read_set(path):
    return set(path.read_text().split("\n")) - {""} if path.exists() else set()

def mark(path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(item + "\n")


def monthly_expirations():
    """Third Friday of every month between START + 1 month and today."""
    out = []
    y, m = START.year, START.month + 1
    while True:
        if m > 12:
            y, m = y + 1, 1
        first = dt.date(y, m, 1)
        third_friday = first + dt.timedelta(days=(4 - first.weekday()) % 7 + 14)
        if third_friday >= TODAY:
            return out
        out.append(third_friday)
        m += 1


def ms_to_date(ms):
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).date().isoformat()


# ---------- the three download steps ----------
def download_underlying():
    """Daily closes for each stock (1 call per ticker)."""
    path = DATA / "underlying.csv"
    have = {r["ticker"] for r in read_rows(path)}
    for t in TICKERS:
        if t in have:
            continue
        print(f"Stock prices: {t}")
        d = get(f"/v2/aggs/ticker/{t}/range/1/day/{START}/{TODAY}", adjusted="false", limit=50000)
        rows = [(t, ms_to_date(b["t"]), b["c"]) for b in (d or {}).get("results", [])]
        if rows:
            append_rows(path, ["ticker", "date", "close"], rows)


def pick_contracts(ticker, expiry, closes):
    """List contracts for one expiration and keep a spread of OTM strikes (1+ calls)."""
    ref_day = (expiry - dt.timedelta(days=30)).isoformat()
    past = [c for d, c in closes if d <= ref_day]
    if not past:
        return []
    ref = past[-1]                                   # stock price ~30 days before expiry
    found, url = [], "/v3/reference/options/contracts"
    params = {"underlying_ticker": ticker, "expired": "true", "limit": 1000,
              # a day's slack in case the 3rd Friday was a holiday (then it's Thursday)
              "expiration_date.gte": (expiry - dt.timedelta(days=1)).isoformat(),
              "expiration_date.lte": expiry.isoformat(),
              "strike_price.gte": round(ref * (1 - STRIKE_RANGE), 2),
              "strike_price.lte": round(ref * (1 + STRIKE_RANGE), 2)}
    while url:
        d = get(url, **params)
        if not d:
            break
        found += d.get("results", [])
        url, params = d.get("next_url"), {}          # next page, if any
    if found:                                        # keep only the monthly date (not Thursday dailies)
        last = max(c["expiration_date"] for c in found)
        found = [c for c in found if c["expiration_date"] == last]

    def spread(cs):                                   # evenly spaced pick of N strikes
        cs = sorted(cs, key=lambda c: c["strike_price"])
        if len(cs) <= STRIKES_PER_SIDE:
            return cs
        step = (len(cs) - 1) / (STRIKES_PER_SIDE - 1)
        return [cs[round(i * step)] for i in range(STRIKES_PER_SIDE)]

    puts = spread([c for c in found if c["contract_type"] == "put" and c["strike_price"] < ref])
    calls = spread([c for c in found if c["contract_type"] == "call" and c["strike_price"] > ref])
    return [(ticker, c["ticker"], c["contract_type"], c["strike_price"], c["expiration_date"])
            for c in puts + calls]


def main(minutes):
    deadline = time.time() + minutes * 60
    download_underlying()

    closes = {}
    for r in read_rows(DATA / "underlying.csv"):
        closes.setdefault(r["ticker"], []).append((r["date"], float(r["close"])))

    # Step 2: which contracts to download (1 call per ticker per expiration)
    done_exp = read_set(DATA / "expirations_done.txt")
    for expiry in monthly_expirations():
        for t in TICKERS:
            key = f"{t} {expiry}"
            if key in done_exp or t not in closes:
                continue
            if time.time() > deadline:
                return print("Time's up -- run again to continue.")
            rows = pick_contracts(t, expiry, sorted(closes[t]))
            print(f"Contracts: {key} -> {len(rows)}")
            append_rows(DATA / "contracts.csv",
                        ["ticker", "contract", "type", "strike", "expiration"], rows)
            mark(DATA / "expirations_done.txt", key)

    # Step 3: daily prices for every chosen contract (1 call per contract)
    contracts = read_rows(DATA / "contracts.csv")
    done = read_set(DATA / "contracts_done.txt")
    todo = [c for c in contracts if c["contract"] not in done]
    print(f"Contracts: {len(contracts)} total, {len(todo)} left "
          f"(~{len(todo) * CALL_GAP / 3600:.1f} hours)")
    for i, c in enumerate(todo, 1):
        if time.time() > deadline:
            return print("Time's up -- run again to continue.")
        expiry = dt.date.fromisoformat(c["expiration"])
        start = max(START, expiry - dt.timedelta(days=365))  # at most 1 year before expiry
        d = get(f"/v2/aggs/ticker/{c['contract']}/range/1/day/{start}/{expiry}",
                adjusted="false", limit=50000)
        if d is None:
            continue                                   # error: try again next run
        rows = [(c["contract"], ms_to_date(b["t"]), b["o"], b["h"], b["l"], b["c"],
                 b["v"], b.get("vw", ""), b.get("n", ""))
                for b in d.get("results", [])]
        append_rows(DATA / "bars" / f"{c['ticker']}.csv",
                    ["contract", "date", "open", "high", "low", "close", "volume", "vwap", "trades"],
                    rows)
        mark(DATA / "contracts_done.txt", c["contract"])
        print(f"[{i}/{len(todo)}] {c['contract']}: {len(rows)} days")

    (DATA / "COMPLETE").write_text(f"Finished {dt.datetime.now():%Y-%m-%d %H:%M}\n")
    print("All done!")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=60, help="stop after this many minutes")
    main(p.parse_args().minutes)
