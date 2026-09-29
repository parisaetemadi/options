"""
Fix days that failed the quality checks in download_databento.py (run AFTER it finishes).

Why days fail: when a stock pays an unusual dividend or has a corporate action, the exchange
adjusts its options (e.g. QQQ on 2023-12-27: every strike lowered by $0.22, so $130 became
$129.78). Our weekly contract list then has names that no longer exist, and that ticker's
quotes come back empty or missing until next week's list.

Fix: for each flagged day, buy that exact day's contract list for just the problem tickers,
re-download their quotes, and replace them in that day's file. Costs a few cents per day.

Run:  python repair_databento.py
"""
import pandas as pd
import pandas_market_calendars as mcal

import download_databento as dd

MIN_LIVE = 0.8      # a ticker is "broken" on a day if under 80% of its quotes are live...
MIN_SHARE = 0.5     # ...or it has under half the contracts it had the previous trading day


def broken_tickers(day, prev):
    q = pd.read_parquet(dd.OUT / "quotes" / day[:4] / f"{day}.parquet").astype({"ticker": str})
    live = ((q["bid"] > 0) & (q["ask"] > 0)).groupby(q["ticker"]).mean()
    count = q.groupby("ticker").size()
    prev_count = prev.groupby("ticker").size() if prev is not None else count
    bad = set(live[live < MIN_LIVE].index)
    bad |= {t for t in dd.TICKERS if count.get(t, 0) < MIN_SHARE * prev_count.get(t, 0)}
    return sorted(bad), q


def main():
    qc = pd.read_csv(dd.OUT / "quality.csv").drop_duplicates("date", keep="last")
    flags = qc["flags"].fillna("")
    flagged = qc[(flags != "") & (flags != "crossed quotes") & ~flags.str.startswith("repaired")]
    closes = pd.read_csv(dd.OUT / "underlying.csv").set_index(["ticker", "date"])["close"]
    sched = mcal.get_calendar("NYSE").schedule(start_date=dd.FIRST_DAY, end_date=qc["date"].max())
    sched.index = sched.index.strftime("%Y-%m-%d")
    days = list(sched.index)
    print(f"{len(flagged)} flagged days to check (crossed-quote-only days are real market events, left as is)")

    for day in sorted(flagged["date"]):
        i = days.index(day)
        prev_path = dd.OUT / "quotes" / days[i - 1][:4] / f"{days[i - 1]}.parquet"
        prev = pd.read_parquet(prev_path).astype({"ticker": str}) if prev_path.exists() else None
        bad, q = broken_tickers(day, prev)
        if not bad:
            print(f"{day}: nothing broken per ticker -- leaving it")
            continue
        # fresh contract list for exactly this day, only the broken tickers
        nxt = days[i + 1] if i + 1 < len(days) else day
        defs = dd.get("definition", [f"{t}.OPT" for t in bad], "parent", day, nxt, f"repair definitions {day}")
        defs = defs.drop_duplicates("raw_symbol")
        defs = pd.DataFrame({"raw_symbol": defs["raw_symbol"].values, "ticker": defs["underlying"].values,
                             "type": ["call" if c == "C" else "put" for c in defs["instrument_class"].values],
                             "strike": defs["strike_price"].values,
                             "expiration": pd.to_datetime(defs["expiration"]).dt.strftime("%Y-%m-%d").values})
        defs = defs[defs["raw_symbol"].str[:6].str.strip() == defs["ticker"]]
        spot = closes.xs(day, level="date")[bad]
        fixed_path = dd.OUT / "quotes" / day[:4] / f"{day}.parquet"
        backup = q.copy()
        # download_day writes the day file (and a quality row) for just these tickers...
        fixed_qc = dd.download_day(day, pd.Timestamp(sched.loc[day, "market_close"]), defs, spot)
        fixed = pd.read_parquet(fixed_path).astype({"ticker": str})
        # ...so put the good tickers back in
        merged = pd.concat([backup[~backup["ticker"].isin(bad)], fixed], ignore_index=True)
        for col in ("date", "ticker", "type", "expiration"):
            merged[col] = merged[col].astype(str).astype("category")
        merged.to_parquet(fixed_path, index=False, compression="zstd")
        # record the whole repaired day in the quality report (the last row per day wins)
        live_all = merged[(merged["bid"] > 0) & (merged["ask"] > 0)]
        pd.DataFrame([{"date": day, "requested": fixed_qc["requested"] + int((~backup["ticker"].isin(bad)).sum()),
                       "received": len(merged), "coverage": None,
                       "live_quotes": round(len(live_all) / len(merged), 3),
                       "crossed": int((live_all["bid"] > live_all["ask"]).sum()),
                       "median_spread": round(((live_all["ask"] - live_all["bid"]) / ((live_all["ask"] + live_all["bid"]) / 2)).median(), 4),
                       "tickers": merged["ticker"].nunique(), "cost": fixed_qc["cost"],
                       "flags": f"repaired {','.join(bad)}"}]).to_csv(dd.OUT / "quality.csv", mode="a", header=False, index=False)
        live = ((fixed["bid"] > 0) & (fixed["ask"] > 0)).mean()
        print(f"{day}: repaired {bad}: {len(backup[backup['ticker'].isin(bad)]):,} -> {len(fixed):,} contracts, "
              f"{live:.0%} live, ${fixed_qc['cost']:.4f}")
    print(f"Total spent: ${dd.spent():.2f}")


if __name__ == "__main__":
    main()
