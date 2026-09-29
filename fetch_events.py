"""
Earnings dates and dividends for our stocks (free), saved to data/events/.

Earnings: from SEC EDGAR, the official record. When a company releases earnings it must
file an 8-K form with "Item 2.02" -- we use the exact time it was filed:
  filed after the 4pm close  -> the market reacts the NEXT trading day
  filed before the 9:30 open -> the market reacts that same day
Cross-checked against Yahoo's earnings calendar. Upcoming (not yet reported) earnings come
from Yahoo's calendar, since the SEC only has past ones.

Event types: "earnings", "deliveries" (Tesla's quarterly delivery report) and "other"
(unscheduled news like profit warnings -- kept for reference, not used as a model input).

Dividends: from Yahoo (ex-dividend date and amount), for stocks and ETFs, plus each day's
trailing 12-month dividend yield (data/events/div_yield.csv) for Black-Scholes.

Run:  python fetch_events.py
"""
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal
import requests
import yfinance as yf

OUT = Path(__file__).parent / "data" / "events"
STOCKS = {"AAPL": 320193, "MSFT": 789019, "NVDA": 1045810, "AMZN": 1018724, "TSLA": 1318605}   # SEC CIK numbers
ALL_TICKERS = ["SPY", "QQQ", "IWM", *STOCKS]
SEC_HEADERS = {"User-Agent": "options-research et1378parisa@gmail.com"}   # SEC asks for a contact
FIRST = "2013-01-01"


def sec_filings(cik):
    """Every filing a company has made (recent list + older archive files)."""
    base = "https://data.sec.gov/submissions/"
    j = requests.get(f"{base}CIK{cik:010d}.json", headers=SEC_HEADERS, timeout=30).json()
    frames = [pd.DataFrame(j["filings"]["recent"])]
    for f in j["filings"]["files"]:
        frames.append(pd.DataFrame(requests.get(base + f["name"], headers=SEC_HEADERS, timeout=30).json()))
    return pd.concat(frames, ignore_index=True)


def earnings_from_sec():
    sessions = mcal.get_calendar("NYSE").schedule(start_date=FIRST, end_date="2027-12-31").index
    rows = []
    for ticker, cik in STOCKS.items():
        f = sec_filings(cik)
        f = f[(f["form"] == "8-K") & f["items"].fillna("").str.split(",").apply(lambda i: "2.02" in i)]
        for _, r in f.iterrows():
            t = pd.Timestamp(r["acceptanceDateTime"]).tz_convert("America/New_York")
            if t.strftime("%Y-%m-%d") < FIRST:
                continue
            minute = t.hour * 60 + t.minute
            day = pd.Timestamp(t.date())
            # first trading session whose prices include the news
            reaction = sessions[sessions > day][0] if minute >= 16 * 60 else sessions[sessions >= day][0]
            rows.append({"ticker": ticker, "filed_ny": t.strftime("%Y-%m-%d %H:%M"),
                         "when": "after close" if minute >= 16 * 60 else ("before open" if minute < 9 * 60 + 30 else "during market"),
                         "reaction_day": reaction.strftime("%Y-%m-%d")})
    e = pd.DataFrame(rows).sort_values(["ticker", "reaction_day"])
    e["type"] = [event_type(t, f) for t, f in zip(e["ticker"], e["filed_ny"])]
    # one event per quarter and type: drop repeat filings within 20 days
    gap = e.groupby(["ticker", "type"])["reaction_day"].transform(lambda d: pd.to_datetime(d).diff().dt.days)
    return e[gap.isna() | (gap > 20)].reset_index(drop=True)


_yahoo = {}

def event_type(ticker, filed):
    """earnings (confirmed on Yahoo's calendar), Tesla's quarterly delivery report, or other
    (unscheduled news such as a profit warning -- can't be known in advance, so not a model input)."""
    if ticker not in _yahoo:
        y = yf.Ticker(ticker).get_earnings_dates(limit=100)
        _yahoo[ticker] = set(y.index.tz_convert("America/New_York").strftime("%Y-%m-%d"))
    day = pd.Timestamp(filed[:10])
    if filed[:10] in _yahoo[ticker] or (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d") in _yahoo[ticker]:
        return "earnings"
    if ticker == "TSLA" and day.month in (1, 4, 7, 10) and day.day <= 5:
        return "deliveries"
    return "other"


def upcoming_from_yahoo():
    """Earnings that haven't happened yet (the SEC only has past ones). All five of our stocks
    report after the close, so when Yahoo doesn't give a time we assume after the close."""
    sessions = mcal.get_calendar("NYSE").schedule(start_date=FIRST, end_date="2027-12-31").index
    now = pd.Timestamp.now(tz="America/New_York")
    rows = []
    for ticker in STOCKS:
        y = yf.Ticker(ticker).get_earnings_dates(limit=12)
        for t in y.index[y.index > now].tz_convert("America/New_York"):
            minute = t.hour * 60 + t.minute
            day = pd.Timestamp(t.date())
            before_open = 0 < minute < 9 * 60 + 30
            reaction = sessions[sessions >= day][0] if before_open else sessions[sessions > day][0]
            rows.append({"ticker": ticker, "filed_ny": t.strftime("%Y-%m-%d %H:%M"),
                         "when": "before open" if before_open else "after close",
                         "reaction_day": reaction.strftime("%Y-%m-%d"), "type": "earnings",
                         "source": "yahoo upcoming"})
    # Tesla's delivery report: always released in the first days of each quarter
    last = pd.Timestamp(now.date())
    for k in range(1, 3):
        q_start = (last + pd.offsets.QuarterBegin(k, startingMonth=1)).normalize()
        reaction = sessions[sessions >= q_start + pd.Timedelta(days=1)][0]   # usually the 2nd day
        rows.append({"ticker": "TSLA", "filed_ny": reaction.strftime("%Y-%m-%d") + " (estimated)",
                     "when": "before open", "reaction_day": reaction.strftime("%Y-%m-%d"),
                     "type": "deliveries", "source": "estimated"})
    return pd.DataFrame(rows)


def check_against_yahoo(e):
    print("\nCross-check with Yahoo's earnings calendar:")
    for ticker in STOCKS:
        try:
            y = yf.Ticker(ticker).get_earnings_dates(limit=100)
        except Exception as ex:
            print(f"  {ticker}: Yahoo unavailable ({ex})")
            continue
        y = y[y.index <= pd.Timestamp.now(tz="America/New_York")]
        ydays = set(y.index.tz_convert("America/New_York").strftime("%Y-%m-%d"))
        sec = e[(e["ticker"] == ticker) & (e["type"] == "earnings") & (e["source"] == "sec 8-k")]
        sec = sec[sec["filed_ny"].str[:10] >= min(ydays)]
        match = sec["filed_ny"].str[:10].isin(ydays)
        print(f"  {ticker}: {match.sum()}/{len(sec)} SEC earnings dates also on Yahoo"
              + ("" if match.all() else f"  -- not on Yahoo: {sec.loc[~match, 'filed_ny'].tolist()}"))


def dividends():
    rows, yields = [], []
    for t in ALL_TICKERS:
        h = yf.Ticker(t).history(start="2012-01-01", auto_adjust=False)   # split-adjusted close + dividends
        h.index = h.index.tz_localize(None)
        d = h.loc[h["Dividends"] > 0, "Dividends"]
        rows += [{"ticker": t, "ex_date": i.strftime("%Y-%m-%d"), "amount": round(v, 6)}
                 for i, v in d.items() if i >= pd.Timestamp(FIRST)]
        trailing = h["Dividends"].rolling("365D").sum()
        y = (trailing / h["Close"]).loc[FIRST:]
        yields.append(pd.DataFrame({"ticker": t, "date": y.index.strftime("%Y-%m-%d"), "div_yield": y.round(6).values}))
    y = pd.concat(yields, ignore_index=True)
    assert y["div_yield"].between(0, 0.05).all(), "dividend yield out of range -- check the data"
    y.to_csv(OUT / "div_yield.csv", index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    e = earnings_from_sec().assign(source="sec 8-k")
    up = upcoming_from_yahoo()
    print("Upcoming events:", up[["ticker", "type", "reaction_day"]].values.tolist())
    e = pd.concat([e, up], ignore_index=True).sort_values(["ticker", "reaction_day"])
    e.to_csv(OUT / "earnings.csv", index=False)
    print(f"Events: {len(e)}  ({e['reaction_day'].min()} to {e['reaction_day'].max()})")
    print(pd.crosstab(e["ticker"], e["type"]).to_string())
    print("Unscheduled ('other') events:", e.loc[e["type"] == "other", ["ticker", "filed_ny"]].values.tolist())
    print("When filed:", e["when"].value_counts().to_dict())
    check_against_yahoo(e)

    d = dividends()
    d.to_csv(OUT / "dividends.csv", index=False)
    print(f"\nDividends: {len(d)} payments (tickers with none: "
          f"{sorted(set(ALL_TICKERS) - set(d['ticker']))})")
    print(d.groupby("ticker").agg(payments=("amount", "size"), last=("ex_date", "max"),
                                  last_amount=("amount", "last")).to_string())
