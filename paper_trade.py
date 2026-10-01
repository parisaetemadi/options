"""
PAPER trading (fake money, no broker) of the stock-signal strategy from stock_signals.py.

Every trading day, after the close:
  1. today's option signals from the Yahoo snapshot (same cleaning/IV code as the backtest;
     check_yahoo.py showed Yahoo and Databento agree)
  2. a model trained ONLY on what was known before today ranks the tickers
     (signals only -- it doesn't know which ticker is which)
  3. target book: long the top-ranked, short the bottom-ranked, positions sum to zero,
     $100,000 gross, each day's ranking held 10 days (1/10 of the book changes per day)
  4. trades happen at the NEXT day's close (conservative: no timing luck), 0.02% cost per trade

The decisions (data/paper/weights.csv) are only ever APPENDED, once per day, and committed to git
-- so the record can't be tuned after the fact. P&L (data/paper/pnl.csv) is recomputed from those
decisions and real closing prices, next to simply holding SPY for comparison.

Run:  python paper_trade.py --init     (once, locally: signal history from the Databento data)
      python paper_trade.py            (daily; on GitHub by itself)
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf

import train as tr

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META", "GOOGL", "AMD", "GLD", "TLT", "XLF"]
CAPITAL, HOLD, COST = 100_000, 10, 0.0002
SIG = ["put_skew", "call_skew", "vrp", "term", "d_put_skew", "d_atm"]
HIST = tr.DATA / "signals" / "history.csv"
PAPER = tr.DATA / "paper"
WEIGHTS = PAPER / "weights.csv"
NY = "America/New_York"


def init_history():
    """Signals for every day we have Databento quotes (the backtest's data)."""
    cols = ["date", "ticker", "type", "expiration", "T", "x", "iv", "atm_iv", "atm30", "rv21"]
    groups = [g for g in ("core", "extra") if (tr.Path(tr.__file__).parent / "models" / f"dataset_{g}.parquet").exists()]
    q = pd.concat([pd.read_parquet(tr.Path(tr.__file__).parent / "models" / f"dataset_{g}.parquet", columns=cols)
                   .astype({"date": str, "ticker": str, "type": str, "expiration": str}) for g in groups])
    sig = tr.smile_signals(q[q["ticker"].isin(TICKERS)]).assign(source="databento")
    HIST.parent.mkdir(parents=True, exist_ok=True)
    sig.to_csv(HIST, index=False, float_format="%.6g")
    print(f"Signal history: {len(sig):,} ticker-days, {sig['date'].min()} to {sig['date'].max()} -> {HIST}")


def closes():
    """Split-adjusted daily closes (Yahoo), finished days only."""
    raw = yf.download(TICKERS, start="2012-06-01", auto_adjust=False, actions=False, progress=False)["Close"]
    c = raw.rename_axis("date").reset_index().melt("date", var_name="ticker", value_name="close").dropna()
    c["date"] = c["date"].dt.strftime("%Y-%m-%d")
    # the bulk download sometimes lacks the newest day (seen on GitHub): add the last few days per ticker
    recent = []
    for t in TICKERS:
        h = yf.Ticker(t).history(period="5d", auto_adjust=False)["Close"]
        recent.append(pd.DataFrame({"date": h.index.strftime("%Y-%m-%d"), "ticker": t, "close": h.values}))
    c = pd.concat([c] + recent).drop_duplicates(["date", "ticker"], keep="first").dropna()
    now = pd.Timestamp.now(tz=NY)
    last_final = now.strftime("%Y-%m-%d") if now.hour * 60 + now.minute >= 16 * 60 + 30 else \
        (now - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    c = c[c["date"] <= last_final]
    print(f"Closing prices up to {c['date'].max()} ({c[c['date'] == c['date'].max()]['ticker'].nunique()} tickers)")
    return c


def add_yahoo_days(hist, px):
    """Signals from each new, final Yahoo snapshot (its day's close must be known)."""
    have = set(hist["date"])
    new = []
    for f in sorted((tr.DATA / "yahoo").glob("*.csv.gz")):
        day = f.name[:10]
        if day in have or day <= hist["date"].max():
            continue
        if day not in set(px["date"]):
            print(f"{day}: snapshot waiting for that day's closing prices")
            continue
        y = pd.read_csv(f)
        y = y[y["ticker"].isin(TICKERS)].assign(date=day)
        days = (pd.to_datetime(y["expiration"]) - pd.Timestamp(day)).dt.days
        m = y["strike"] / y["spot"]
        otm = ((y["type"] == "call") & (m > 1)) | ((y["type"] == "put") & (m < 1))
        y = y[otm & m.between(0.8, 1.2) & days.between(7, 365)]
        q = tr.prepare(y[["date", "ticker", "type", "strike", "expiration", "spot", "bid", "ask"]], adjusted_closes=px)
        s = tr.smile_signals(q).assign(source="yahoo")
        print(f"{day}: signals for {len(s)} tickers from Yahoo (snapshot {y['snapshot_time'].iloc[0]})")
        new.append(s)
    if new:
        hist = pd.concat([hist] + new, ignore_index=True)
        hist.to_csv(HIST, index=False, float_format="%.6g")
    return hist


def with_features(hist):
    d = hist.sort_values(["ticker", "date"]).copy()
    g = d.groupby("ticker")
    d["d_put_skew"] = d["put_skew"] - g["put_skew"].shift(5)
    d["d_atm"] = np.log(d["atm"] / g["atm"].shift(5))
    return d


def decide(d, px, day):
    """Weights for signal day `day`, from a model trained only on outcomes known by then."""
    trading_days = sorted(px["date"].unique())
    r = px.pivot(index="date", columns="ticker", values="close").sort_index()
    fwd5 = (r.shift(-5) / r - 1).stack().rename("next5").reset_index()
    train = d.merge(fwd5, on=["date", "ticker"]).dropna(subset=SIG + ["next5"])
    known = trading_days[trading_days.index(day) - 5]              # next5 of this date ends on `day`
    train = train[train["date"] <= known]
    y = train["next5"] - train.groupby("date")["next5"].transform("mean")
    m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.02, num_leaves=7, min_child_samples=200,
                          subsample=0.8, subsample_freq=1, verbose=-1, random_state=0).fit(train[SIG].values, y)
    today = d[(d["date"] == day)].dropna(subset=SIG)
    fc = pd.Series(m.predict(today[SIG].values), index=today["ticker"].values)
    w = fc.rank() - fc.rank().mean()
    w = w / w.abs().sum()
    return pd.DataFrame({"signal_date": day, "ticker": fc.index, "forecast": fc.values, "weight": w.values,
                         "decided_at": pd.Timestamp.now(tz=NY).strftime("%Y-%m-%d %H:%M %Z"),
                         "trained_until": known})


def pnl(W, px):
    """Book from the decisions (entered at the next close), daily P&L from real closes."""
    days = sorted(px["date"].unique())
    w = W.pivot(index="signal_date", columns="ticker", values="weight").fillna(0).rolling(HOLD, min_periods=1).mean()
    nxt = {d: days[i + 1] for i, d in enumerate(days[:-1])}
    w.index = [nxt.get(d) for d in w.index]                       # trade at the next day's close
    w = w[w.index.notna()]
    if w.empty:
        return None, None                                          # first trade is at a close still to come
    r = px.pivot(index="date", columns="ticker", values="close").sort_index().pct_change()
    start = w.index.min()
    book = w.reindex([d for d in days if d >= start]).ffill().fillna(0)     # holdings after each close
    ret = r.reindex(book.index).fillna(0)
    held = book.shift(1).fillna(0)                                 # what we held going INTO each day
    gross = (held * ret[held.columns]).sum(axis=1) * CAPITAL
    cost = (book - book.shift(1).fillna(0)).abs().sum(axis=1) * COST * CAPITAL   # incl. the first day's entry
    out = pd.DataFrame({"gross": gross, "cost": cost})
    out["net"] = out["gross"] - out["cost"]
    out["equity"] = CAPITAL + out["net"].cumsum()
    out["spy_hold"] = CAPITAL * (1 + r["SPY"].reindex(book.index).fillna(0)).cumprod()
    return out.rename_axis("date").round(2), (book * CAPITAL).round(0)


def report(out, book, W):
    """data/paper/report.md: one row per trading day, plus today's positions."""
    first = W["signal_date"].min()
    eq, spy = out["equity"].iloc[-1], out["spy_hold"].iloc[-1]
    lines = ["# Paper trading -- FAKE money", "",
             f"Stock-signal strategy (stock_signals.py): {len(TICKERS)} tickers, ${CAPITAL:,} gross, long and short "
             f"in equal dollars, each day's ranking held {HOLD} days, trades at the next close, {COST:.2%} cost per trade.",
             f"First decision {first}; first trades {out.index[0]}. Updated {pd.Timestamp.now(tz=NY):%Y-%m-%d %H:%M %Z}.", "",
             f"**Account: ${eq:,.0f} ({eq / CAPITAL - 1:+.2%})** -- $100k simply held in SPY: ${spy:,.0f} "
             f"({spy / CAPITAL - 1:+.2%}). {len(out)} trading days: far too few to judge (needs 6-12 months).", "",
             "## Every day", "",
             "| Date | Day P&L | of which costs | Account | Total | SPY instead | Biggest longs | Biggest shorts |",
             "|---|---:|---:|---:|---:|---:|---|---|"]
    for d, r in out.iloc[::-1].iterrows():
        b = book.loc[d]
        longs = ", ".join(f"{t} ${v / 1000:.1f}k" for t, v in b[b > 0].nlargest(3).items())
        shorts = ", ".join(f"{t} ${-v / 1000:.1f}k" for t, v in b[b < 0].nsmallest(3).items())
        lines.append(f"| {d} | ${r['net']:+,.0f} | ${r['cost']:,.0f} | ${r['equity']:,.0f} | {r['equity'] / CAPITAL - 1:+.2%} "
                     f"| ${r['spy_hold']:,.0f} | {longs} | {shorts} |")
    b = book.iloc[-1]
    lines += ["", f"## Positions after the {book.index[-1]} close", "", "| Ticker | Position |", "|---|---:|"]
    lines += [f"| {t} | {'long' if v > 0 else 'short'} ${abs(v):,.0f} |" for t, v in b[b != 0].sort_values(ascending=False).items()]
    (PAPER / "report.md").write_text("\n".join(lines) + "\n")


def main():
    if "--init" in sys.argv:
        return init_history()
    px = closes()
    hist = add_yahoo_days(pd.read_csv(HIST), px)
    d = with_features(hist)
    W = pd.read_csv(WEIGHTS) if WEIGHTS.exists() else pd.DataFrame(columns=["signal_date"])
    todo = sorted(set(hist.loc[hist["source"] == "yahoo", "date"]) - set(W["signal_date"]))
    for day in todo:                                               # normally just today
        new = decide(d, px, day)
        W = pd.concat([W, new], ignore_index=True) if len(W) else new
        PAPER.mkdir(parents=True, exist_ok=True)
        new.to_csv(WEIGHTS, mode="a", header=not WEIGHTS.exists(), index=False, float_format="%.6g")
        print(f"{day}: decided -- long {', '.join(new.nlargest(3, 'weight')['ticker'])}; "
              f"short {', '.join(new.nsmallest(3, 'weight')['ticker'])}")
    if not len(W):
        return print("No paper decisions yet.")
    out, book = pnl(W, px)
    if out is None:
        return print("First paper trades happen at the next close -- nothing to score yet.")
    out.to_csv(PAPER / "pnl.csv")
    book.iloc[-1].rename("position_usd").rename_axis("ticker").to_csv(PAPER / "book.csv")
    report(out, book, W)
    last = out.iloc[-1]
    print(f"Paper account {out.index[-1]}: ${last['equity']:,.0f} (net {last['equity'] / CAPITAL - 1:+.2%}); "
          f"$100k in SPY instead: ${last['spy_hold']:,.0f} ({last['spy_hold'] / CAPITAL - 1:+.2%}); "
          f"{len(out)} trading days")


if __name__ == "__main__":
    main()
