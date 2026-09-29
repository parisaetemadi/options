"""
News headlines per ticker from Massive's news feed (free plan: 5 calls/minute), turned into
daily model inputs. Saved to data/news/.

For each trading day we only count news published BEFORE our 3:59pm option snapshot
(from the previous day's snapshot up to 3:59pm), so the model never sees the future:
  news_count        articles about the ticker in that window
  news_pos/neg      articles Massive's sentiment tags as positive / negative for that ticker
  news_count_5d     articles over the last 5 trading days (a "busy news week" signal)

Safe to stop and re-run: tickers already downloaded are skipped (use --refresh to update).

Run:  python fetch_news.py
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
import requests

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA",
           "META", "GOOGL", "AMD", "GLD", "TLT", "XLF"]
OUT = Path(__file__).parent / "data" / "news"
CALL_GAP = 12.5          # 5 calls per minute


def load_key():
    for line in (Path(__file__).parent / ".env").read_text().splitlines():
        if line.startswith("MASSIVE_API_KEY="):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit("No MASSIVE_API_KEY in .env")


KEY = load_key()
_last = 0.0


def get(url, params=None):
    global _last
    for _ in range(6):
        wait = CALL_GAP - (time.time() - _last)
        if wait > 0:
            time.sleep(wait)
        _last = time.time()
        r = requests.get(url, params={**(params or {}), "apiKey": KEY}, timeout=60)
        if r.status_code == 429:
            time.sleep(60)
            continue
        if r.status_code >= 500:
            time.sleep(30)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Massive kept failing")


def download(ticker):
    """Every article tagged with this ticker, oldest to newest (1,000 per call)."""
    rows, url = [], "https://api.massive.com/v2/reference/news"
    params = {"ticker": ticker, "limit": 1000, "order": "asc", "sort": "published_utc"}
    while url:
        j = get(url, params)
        for a in j.get("results", []):
            ins = {i.get("ticker"): i.get("sentiment") for i in a.get("insights") or []}
            rows.append({"id": a.get("id"), "published_utc": a.get("published_utc"),
                         "publisher": (a.get("publisher") or {}).get("name"),
                         "title": a.get("title"), "tickers": ",".join(a.get("tickers") or []),
                         "sentiment": ins.get(ticker)})
        url, params = j.get("next_url"), None            # next page, if any
        print(f"  {ticker}: {len(rows):,} articles so far", flush=True)
    return pd.DataFrame(rows).drop_duplicates("id")


def daily_features(news):
    """Count articles in each trading day's window: previous 3:59pm -> this day's 3:59pm (New York)."""
    sched = mcal.get_calendar("NYSE").schedule(start_date="2013-01-01", end_date=pd.Timestamp.now())
    cut = sched["market_close"] - pd.Timedelta(minutes=1)            # our option snapshot time (UTC)
    news = news.assign(t=pd.to_datetime(news["published_utc"], utc=True)).sort_values("t")
    # each article belongs to the first snapshot at or after it was published
    idx = cut.searchsorted(news["t"])
    news = news[idx < len(cut)].assign(date=cut.index[idx[idx < len(cut)]].strftime("%Y-%m-%d"))
    g = news.groupby(["ticker", "date"])
    f = pd.DataFrame({"news_count": g.size(),
                      "news_pos": g["sentiment"].apply(lambda s: (s == "positive").sum()),
                      "news_neg": g["sentiment"].apply(lambda s: (s == "negative").sum())}).reset_index()
    # fill days with no news with zeros, then add a 5-day count
    days = sched.index.strftime("%Y-%m-%d")
    full = pd.MultiIndex.from_product([sorted(news["ticker"].unique()), days], names=["ticker", "date"])
    f = f.set_index(["ticker", "date"]).reindex(full, fill_value=0).reset_index()
    f["news_count_5d"] = f.groupby("ticker")["news_count"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    return f


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true", help="download every ticker again")
    p.add_argument("--tickers", default=",".join(TICKERS))
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for t in a.tickers.split(","):
        path = OUT / f"{t}.csv.gz"
        if path.exists() and not a.refresh:
            continue
        print(f"Downloading news for {t}...")
        download(t).assign(ticker=t).to_csv(path, index=False)
    allnews = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("*.csv.gz"))], ignore_index=True)
    f = daily_features(allnews)
    f.to_csv(OUT / "daily_features.csv", index=False)
    # quick quality summary
    first = allnews.groupby("ticker")["published_utc"].min().str[:10]
    tagged = allnews["sentiment"].notna().groupby(allnews["ticker"]).mean()
    print(pd.DataFrame({"articles": allnews.groupby("ticker").size(), "first": first,
                        "with sentiment": tagged.map("{:.0%}".format)}).to_string())
