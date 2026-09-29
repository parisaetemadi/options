"""
"Attention" data, free and official, as possible model inputs. Saved to data/attention/.

1. Wikipedia daily page views (Wikimedia's official API, from July 2015).
   A page's views for a day are only complete at midnight UTC -- after our 3:59pm snapshot --
   so the model gets YESTERDAY's views (never today's).
   Pages that were renamed are added up (e.g. "Facebook,_Inc." + "Meta_Platforms").
   ETFs use the page of the index / asset they track.
2. SEC 8-K filings other than earnings (item 2.02) and exhibits-only (9.01): deals, executive
   changes, guidance, lawsuits... Timestamped to the second, so each filing counts toward the
   first snapshot after it was filed.

Features per ticker and trading day (data/attention/daily_features.csv):
  wiki_views_log     log of yesterday's page views
  wiki_spike         yesterday's views / median of the 30 days before (attention spike)
  sec_8k             number of non-earnings 8-K filings since the previous snapshot
  sec_8k_20d         ... over the last 20 trading days

Run:  python fetch_attention.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import requests

OUT = Path(__file__).parent / "data" / "attention"
HEADERS = {"User-Agent": "options-research et1378parisa@gmail.com"}
WIKI_PAGES = {
    "AAPL": ["Apple_Inc."], "MSFT": ["Microsoft"], "NVDA": ["Nvidia"],
    "AMZN": ["Amazon_(company)", "Amazon.com"], "TSLA": ["Tesla,_Inc.", "Tesla_Motors"],
    "META": ["Meta_Platforms", "Facebook,_Inc."], "GOOGL": ["Alphabet_Inc."],
    "AMD": ["AMD", "Advanced_Micro_Devices"],
    "SPY": ["S&P_500"], "QQQ": ["Nasdaq-100", "NASDAQ-100"], "IWM": ["Russell_2000_Index"],
    "GLD": ["Gold_as_an_investment"], "TLT": ["United_States_Treasury_security"],
    "XLF": ["JPMorgan_Chase"],                        # biggest financial stock (the ETF page is new)
}
SEC_CIKS = {"AAPL": [320193], "MSFT": [789019], "NVDA": [1045810], "AMZN": [1018724], "TSLA": [1318605],
            "META": [1326801], "GOOGL": [1652044, 1288776], "AMD": [2488]}


def wiki_views(page):
    url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
           f"{requests.utils.quote(page, safe='')}/daily/20150701/{pd.Timestamp.now():%Y%m%d}")
    r = requests.get(url, headers=HEADERS, timeout=60)
    if r.status_code == 404:
        return pd.Series(dtype=float)                    # page has no views under this name
    r.raise_for_status()
    time.sleep(0.2)
    items = r.json()["items"]
    return pd.Series({pd.Timestamp(i["timestamp"][:8]): i["views"] for i in items}, dtype=float)


def sec_8k(ciks):
    base = "https://data.sec.gov/submissions/"
    frames = []
    for cik in ciks:
        j = requests.get(f"{base}CIK{cik:010d}.json", headers=HEADERS, timeout=30).json()
        frames.append(pd.DataFrame(j["filings"]["recent"]))
        for f in j["filings"]["files"]:
            frames.append(pd.DataFrame(requests.get(base + f["name"], headers=HEADERS, timeout=30).json()))
        time.sleep(0.2)
    f = pd.concat(frames, ignore_index=True)
    f = f[f["form"] == "8-K"]
    items = f["items"].fillna("").str.split(",")
    news = items.apply(lambda i: any(x not in ("2.02", "9.01", "") for x in i))   # not just earnings/exhibits
    return pd.to_datetime(f.loc[news, "acceptanceDateTime"], utc=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sched = mcal.get_calendar("NYSE").schedule(start_date="2015-07-01", end_date=pd.Timestamp.now())
    days = sched.index                                     # trading days (dates)
    snap = sched["market_close"] - pd.Timedelta(minutes=1)  # our option snapshot (UTC)
    rows, report = [], []
    for t, pages in WIKI_PAGES.items():
        views = pd.concat([wiki_views(p) for p in pages], axis=1).sum(axis=1, min_count=1).sort_index()
        # yesterday's full UTC day, as known at today's snapshot
        yday = views.reindex(days - pd.Timedelta(days=1))
        yday.index = days
        base = views.rolling(30, min_periods=10).median().shift(1).reindex(days - pd.Timedelta(days=1))
        base.index = days
        f = pd.DataFrame({"ticker": t, "date": days.strftime("%Y-%m-%d"),
                          "wiki_views_log": np.log1p(yday.values), "wiki_spike": (yday / base).values})
        if t in SEC_CIKS:
            filed = sec_8k(SEC_CIKS[t])
            idx = snap.searchsorted(filed)
            counts = pd.Series(idx[idx < len(snap)]).value_counts()
            f["sec_8k"] = pd.Series(counts, index=range(len(days))).fillna(0).values
        else:
            f["sec_8k"] = 0.0                               # ETFs don't file 8-Ks
        f["sec_8k_20d"] = f["sec_8k"].rolling(20, min_periods=1).sum()
        rows.append(f)
        report.append({"ticker": t, "wiki days": int(views.notna().sum()), "wiki first": views.first_valid_index().date(),
                       "median daily views": int(views.median()), "missing trading days": int(yday.isna().sum()),
                       "8-K news filings": int(f["sec_8k"].sum())})
        print(f"{t}: done", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT / "daily_features.csv", index=False)
    print(pd.DataFrame(report).to_string(index=False))
