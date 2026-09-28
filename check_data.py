"""
Double-check the downloaded data. Safe to run any time (reads files only, costs nothing).

1. Databento daily quality report (coverage, empty quotes, bid > ask)
2. Databento vs Massive, same contract and same day:
     - Databento's last trade vs Massive's closing trade (both come from OPRA, should match)
     - Is Massive's close inside Databento's closing bid/ask?
3. Stock prices: Databento's (from Yahoo) vs Massive's

Run:  python check_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
DB, MS = DATA / "databento", DATA / "massive"


def pct(x):
    return f"{x:.1%}"


# ---------- 1. Databento daily quality ----------
q = pd.read_csv(DB / "quality.csv").drop_duplicates("date", keep="last")
print(f"1. Databento: {len(q):,} days  ({q['date'].min()} to {q['date'].max()}), "
      f"${q['cost'].sum():.2f} on quotes")
print(f"   coverage: median {pct(q['coverage'].median())}, worst {pct(q['coverage'].min())}")
print(f"   live quotes: median {pct(q['live_quotes'].median())}, worst {pct(q['live_quotes'].min())}")
print(f"   crossed quotes (bid > ask): {q['crossed'].sum()} in total on {(q['crossed'] > 0).sum()} days")
flagged = q[q["flags"].notna() & (q["flags"] != "")]
print(f"   flagged days: {len(flagged)}")
if len(flagged):
    print(flagged[["date", "coverage", "live_quotes", "crossed", "flags"]].tail(10).to_string(index=False))


# ---------- 2. Databento vs Massive, contract by contract ----------
files = sorted((DB / "quotes").glob("*/*.parquet"))
bars = pd.concat([pd.read_csv(p) for p in (MS / "bars").glob("*.csv")], ignore_index=True)
bars = bars[bars["date"].isin({f.stem for f in files})]
if len(bars):
    d = pd.concat([pd.read_parquet(f) for f in files if f.stem in set(bars["date"])], ignore_index=True)
    d = d.astype({"date": str, "ticker": str, "type": str, "expiration": str})
    d["contract"] = ("O:" + d["ticker"] + pd.to_datetime(d["expiration"]).dt.strftime("%y%m%d")
                     + d["type"].str[0].str.upper() + (d["strike"] * 1000).round().astype(int).astype(str).str.zfill(8))
    j = d.merge(bars, on=["contract", "date"])
    j = j[j["volume"] >= 50]                          # contracts that really traded that day
    same_trade = (j["last"] - j["close"]).abs() <= 0.011
    inside = (j["close"] >= j["bid"] - 0.011) & (j["close"] <= j["ask"] + 0.011)
    mid = (j["bid"] + j["ask"]) / 2
    print(f"\n2. Databento vs Massive: {len(j):,} matching contract-days on {j['date'].nunique()} days")
    print(f"   same last trade price (within 1 cent): {pct(same_trade.mean())}"
          f"   (Databento snapshot is 1 minute before close, so a few late trades differ)")
    print(f"   Massive close inside Databento bid/ask: {pct(inside.mean())}")
    print(f"   median gap between Massive close and Databento mid: {pct(((j['close'] - mid).abs() / mid).median())}")
    worst = j.assign(gap=((j["close"] - mid).abs() / mid)).nlargest(5, "gap")
    print("   biggest gaps (check these look like real late moves, not bad data):")
    print(worst[["date", "contract", "bid", "ask", "last", "close", "volume"]].to_string(index=False))
else:
    print("\n2. No overlapping days with Massive yet.")


# ---------- 3. Stock prices ----------
u = pd.read_csv(DB / "underlying.csv").merge(pd.read_csv(MS / "underlying.csv"), on=["ticker", "date"],
                                                 suffixes=("_databento", "_massive"))
diff = (u["close_databento"] / u["close_massive"] - 1).abs()
print(f"\n3. Stock closes: {len(u):,} matching days, median diff {diff.median():.4%}, worst {diff.max():.4%}")
