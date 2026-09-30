"""
Do Yahoo's daily snapshots give the same numbers as Databento's official OPRA quotes?
(Paper trading will run on Yahoo, the model was built on Databento -- they must agree.)

For a day both have: same filter as the Databento download (OTM, strikes +-20%, 7-365 days),
same cleaning and IV code (train.prepare), then compare option by option and the stock signals.

Run:  python check_yahoo.py 2026-09-28
"""
import sys

import numpy as np
import pandas as pd

import download_databento as dd
import train as tr

day = sys.argv[1] if len(sys.argv) > 1 else "2026-09-28"

# Databento: the last minute before the close
db = pd.concat([pd.read_parquet(dd.quote_path(day, g)) for g in dd.GROUPS if dd.quote_path(day, g).exists()],
               ignore_index=True)

# Yahoo: same shape and same filter
y = pd.read_csv(tr.DATA / "yahoo" / f"{day}.csv.gz")
print(f"Yahoo snapshot taken {y['snapshot_time'].iloc[0]}")
y = y.assign(date=day)
days = (pd.to_datetime(y["expiration"]) - pd.Timestamp(day)).dt.days
m = y["strike"] / y["spot"]
otm = ((y["type"] == "call") & (m > 1)) | ((y["type"] == "put") & (m < 1))
y = y[otm & m.between(1 - dd.STRIKE_RANGE, 1 + dd.STRIKE_RANGE) & days.between(dd.MIN_DAYS, dd.MAX_DAYS)]
y = y[y["ticker"].isin(db["ticker"].astype(str).unique())]

# the same stock price for both (Databento rows use our official close; Yahoo's may differ slightly)
spot_db = db.drop_duplicates("ticker").set_index(db.drop_duplicates("ticker")["ticker"].astype(str))["spot"]
spot_y = y.drop_duplicates("ticker").set_index("ticker")["spot"]
print("Stock price, Yahoo vs our close (%):", ((spot_y / spot_db - 1) * 100).round(3).to_dict())

cols = ["date", "ticker", "type", "strike", "expiration", "spot", "bid", "ask"]
P = {"databento": tr.prepare(db[cols]), "yahoo": tr.prepare(y[cols])}

# ---------- option by option ----------
key = ["ticker", "type", "strike", "expiration"]
both = P["databento"].merge(P["yahoo"], on=key, suffixes=("_db", "_y"))
print(f"\nOptions in both after cleaning: {len(both):,} "
      f"(Databento {len(P['databento']):,}, Yahoo {len(P['yahoo']):,})")
both["mid_gap"] = (both["mid_y"] / both["mid_db"] - 1).abs()
both["iv_gap"] = (both["iv_y"] - both["iv_db"]).abs()
print(f"Mid price gap: median {both['mid_gap'].median():.2%}, 90th pct {both['mid_gap'].quantile(0.9):.2%}")
print(f"IV gap (vol points): median {both['iv_gap'].median() * 100:.2f}, 90th pct {both['iv_gap'].quantile(0.9) * 100:.2f}")
print("Median IV gap by ticker:", (both.groupby("ticker")["iv_gap"].median() * 100).round(2).to_dict())

# ---------- the signals the strategy uses ----------
S = {k: tr.smile_signals(v).set_index("ticker") for k, v in P.items()}
cmp = pd.concat({k: v[["atm", "put_skew", "call_skew", "term", "vrp"]] for k, v in S.items()}, axis=1)
print("\nSignals, Databento vs Yahoo:")
for c in ["atm", "put_skew", "call_skew", "term", "vrp"]:
    a, b = cmp[("databento", c)], cmp[("yahoo", c)]
    print(f"  {c:9s} rank agreement across tickers {a.rank().corr(b.rank()):.2f}, "
          f"median abs diff {np.nanmedian(np.abs(a - b)):.4f} (typical size {np.nanmedian(np.abs(a)):.4f})")
