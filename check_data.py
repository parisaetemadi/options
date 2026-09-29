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
q = pd.concat([pd.read_csv(f).assign(group=f.stem.split("_", 1)[1]) for f in DB.glob("quality_*.csv")]).drop_duplicates(["group", "date"], keep="last")
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
files = sorted((DB / "quotes").glob("*/*/*.parquet"))   # quotes/<group>/<year>/<day>.parquet
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


# ---------- 4. Field-by-field checks on every Databento quote ----------
import pandas_market_calendars as mcal
from pricing import implied_vol_many

q = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
q = q.astype({"date": str, "ticker": str, "type": str, "expiration": str})
n = len(q)
print(f"\n4. Field checks on {n:,} quotes, {q['date'].nunique():,} days ({q['date'].min()} to {q['date'].max()})")

def report(name, bad):
    print(f"   {'OK ' if bad == 0 else 'NOTE'} {name}: {bad:,} ({bad / n:.3%})")

# dates
sessions = set(mcal.get_calendar("NYSE").schedule(start_date=q["date"].min(), end_date=q["date"].max()).index.strftime("%Y-%m-%d"))
report("dates that are not NYSE trading days", int((~q["date"].isin(sessions)).sum()))
missing_days = sorted(sessions - set(q["date"]))
print(f"   trading days not downloaded (yet) in this range: {len(missing_days)}")
days_to_exp = (pd.to_datetime(q["expiration"]) - pd.to_datetime(q["date"])).dt.days
report("expiry not 7-365 days after the date", int((~days_to_exp.between(7, 365)).sum()))
report("duplicate rows (same contract, same day)",
       int(q.duplicated(["date", "ticker", "type", "strike", "expiration"]).sum()))
# strikes
m = q["strike"] / q["spot"]
report("strike outside +-20% of the stock", int((~m.between(0.8, 1.2)).sum()))
report("not out-of-the-money (call below / put above the stock)",
       int((((q["type"] == "call") & (m <= 1)) | ((q["type"] == "put") & (m >= 1))).sum()))
report("strike <= 0 or stock price <= 0", int(((q["strike"] <= 0) | (q["spot"] <= 0)).sum()))
# premiums
live = (q["bid"] > 0) & (q["ask"] > 0)
report("no live quote (bid or ask is 0)", int((~live).sum()))
report("bid above ask (crossed)", int((live & (q["bid"] > q["ask"])).sum()))
report("negative prices", int(((q["bid"] < 0) | (q["ask"] < 0)).sum()))
mid = (q["bid"] + q["ask"]) / 2
report("OTM option priced above the stock (call) or strike (put)",
       int((((q["type"] == "call") & (mid > q["spot"])) | ((q["type"] == "put") & (mid > q["strike"]))).sum()))
# no-arbitrage across strikes: for the same expiry, a call can't cost more than a lower-strike call
# (and a put can't cost more than a higher-strike put); allow 1 cent for rounding
s = q[live].sort_values(["date", "ticker", "expiration", "type", "strike"])
s_mid = (s["bid"] + s["ask"]) / 2
nxt = s.groupby(["date", "ticker", "expiration", "type"])
step = s_mid.groupby([s["date"], s["ticker"], s["expiration"], s["type"]]).diff()
wrong = ((s["type"] == "call") & (step > 0.01)) | ((s["type"] == "put") & (step < -0.01))
print(f"   {'OK ' if wrong.mean() < 0.005 else 'NOTE'} price goes the wrong way across strikes (arbitrage): "
      f"{int(wrong.sum()):,} of {len(s):,} ({wrong.mean():.3%})")

# IV: SPY 30-day at-the-money implied vol vs the VIX (VIX is S&P 500 30-day implied vol, usually a bit higher)
spy = q[(q["ticker"] == "SPY") & live].copy()
spy["T"] = days_to_exp[spy.index] / 365
spy = spy[spy["T"].between(20 / 365, 45 / 365) & (np.abs(np.log(spy["strike"] / spy["spot"])) < 0.02)]
spy["iv"] = implied_vol_many((spy["bid"] + spy["ask"]) / 2, spy["spot"], spy["strike"], spy["T"], 0.03, spy["type"])
atm = spy.groupby("date")["iv"].median() * 100
vix = pd.read_csv(DATA / "market" / "vix_rates.csv").set_index("date")["vix"]
both = pd.concat([atm.rename("spy_atm_iv"), vix], axis=1, join="inner").dropna()
print(f"   SPY 30-day ATM IV vs VIX on {len(both):,} days: correlation {both.corr().iloc[0, 1]:.3f}, "
      f"median VIX minus SPY IV {(both['vix'] - both['spy_atm_iv']).median():.1f} pts")
yearly = both.assign(y=both.index.str[:4]).groupby("y").median().round(1)
print("   by year (median):", yearly.to_dict("index"))
