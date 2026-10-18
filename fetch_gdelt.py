"""
Free daily NEWS data from GDELT (worldwide online news, monitored every 15 minutes), 2017 -> today:
for each stock (searched by company name) and for one market-stress topic, every day's
  count  articles mentioning it            norm   all articles GDELT saw that day
  share  count / norm (corrects for GDELT's own coverage growing or dropping)
  tone   average tone of those articles (GDELT's word-based score, < 0 = negative)
GDELT days are UTC: a day is complete at 8pm New York (7pm in winter), i.e. after the 4pm close, so
a model may use day D's numbers from the close of the next trading day on -- never at D's own close.

GDELT asks for at most one request every 5 seconds and often answers 429; we wait and retry.
Saved to data/news/gdelt_daily.csv (one row per name and day). Safe to re-run: finished names are skipped.

Run:  python fetch_gdelt.py
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).parent / "data" / "news" / "gdelt_daily.csv"
START, END = "20170101000000", pd.Timestamp.now("UTC").strftime("%Y%m%d000000")
API = "https://api.gdeltproject.org/api/v2/doc/doc"
# company names, not tickers (ticker letters like "F" or "T" match everything); ambiguous names get qualifiers
QUERIES = {
    "AAL": '"American Airlines"', "AAPL": '("Apple Inc" OR iPhone OR "Tim Cook")', "AMD": '("Advanced Micro Devices" OR "AMD chip")',
    "AMZN": "Amazon", "AVGO": "Broadcom", "BA": "Boeing", "BABA": "Alibaba", "BAC": '"Bank of America"',
    "CRM": "Salesforce", "F": '"Ford Motor"', "GME": "GameStop", "GOOGL": "(Google OR Alphabet)", "INTC": "Intel",
    "META": '(Facebook OR "Meta Platforms")', "MRVL": '"Marvell Technology"', "MSFT": "Microsoft",
    "MSTR": '(MicroStrategy OR "Michael Saylor")', "MU": '"Micron Technology"', "NFLX": "Netflix", "NKE": "Nike",
    "NOK": "Nokia", "NOW": "ServiceNow", "NVDA": "Nvidia", "ORCL": "Oracle", "PFE": "Pfizer", "PYPL": "PayPal",
    "QCOM": "Qualcomm", "SMCI": '("Super Micro Computer" OR Supermicro)', "T": '"AT&T"', "TSLA": "Tesla",
    "TSM": '(TSMC OR "Taiwan Semiconductor")', "UNH": "UnitedHealth", "WMT": "Walmart", "XOM": "Exxon",
    # market-wide stress news (fixed before any test, HYPOTHESES.md N3)
    "_STRESS": '(recession OR tariffs OR "trade war" OR selloff OR "market crash")',
}


def get(query, mode):
    params = {"query": query, "mode": mode, "format": "json", "startdatetime": START, "enddatetime": END}
    for attempt in range(12):
        time.sleep(6 if attempt == 0 else 20)
        try:
            r = requests.get(API, params=params, timeout=120, headers={"User-Agent": "Mozilla/5.0 (research)"})
        except requests.RequestException as e:
            print(f"  {mode}: {e.__class__.__name__}, retrying", flush=True)
            continue
        if r.status_code == 200:
            try:
                data = json.loads(r.text)["timeline"]
            except (ValueError, KeyError):
                print(f"  {mode}: not JSON ({r.text[:80]!r}), retrying", flush=True)
                continue
            return data
        if r.status_code != 429:
            print(f"  {mode}: HTTP {r.status_code}, retrying", flush=True)
    raise RuntimeError(f"GDELT kept refusing {mode} for {query}")


def one(name, query):
    vol = get(query, "timelinevolraw")[0]["data"]
    tone = get(query, "timelinetone")[0]["data"]
    d = pd.DataFrame({"date": [x["date"][:8] for x in vol], "count": [x["value"] for x in vol], "norm": [x["norm"] for x in vol]})
    t = pd.DataFrame({"date": [x["date"][:8] for x in tone], "tone": [x["value"] for x in tone]})
    d = d.merge(t, on="date", how="left")
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    d["share"] = d["count"] / d["norm"].where(d["norm"] > 0)
    return d.assign(name=name, query=query)[["name", "date", "count", "norm", "share", "tone", "query"]]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(OUT)["name"]) if OUT.exists() else set()
    for name, query in QUERIES.items():
        if name in done:
            continue
        d = one(name, query)
        d.to_csv(OUT, mode="a", index=False, header=not OUT.exists())
        print(f"{name}: {len(d)} days {d['date'].min()}..{d['date'].max()}, median {d['count'].median():.0f} articles/day, "
              f"{(d['count'] == 0).mean():.1%} zero days", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
