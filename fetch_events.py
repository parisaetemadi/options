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

Economic announcements (data/events/macro.csv), from the official sources:
  fomc: Fed rate decisions (statement at 2pm) -- regular meetings only, from federalreserve.gov
  cpi:  inflation report (8:30am), from bls.gov      jobs: jobs report (8:30am), from bls.gov
Past dates come from the archives, upcoming ones from the published schedules.

Dividends: from Yahoo (ex-dividend date and amount), for stocks and ETFs, plus each day's
trailing 12-month dividend yield (data/events/div_yield.csv) for Black-Scholes.

Run:  python fetch_events.py
"""
from pathlib import Path

import re

import pandas as pd
import pandas_market_calendars as mcal
import requests
import yfinance as yf

OUT = Path(__file__).parent / "data" / "events"
# SEC CIK numbers (a company can have two: Google Inc. before 2015, Alphabet after)
STOCKS = {"AAPL": [320193], "MSFT": [789019], "NVDA": [1045810], "AMZN": [1018724], "TSLA": [1318605],
          "META": [1326801], "GOOGL": [1652044, 1288776], "AMD": [2488]}
ALL_TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "XLF", *STOCKS]
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
    for ticker, ciks in STOCKS.items():
        f = pd.concat([sec_filings(c) for c in ciks], ignore_index=True)
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


BROWSER = {"User-Agent": "Mozilla/5.0 (options-research et1378parisa@gmail.com)"}
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


def page_text(url):
    html = requests.get(url, headers=BROWSER, timeout=30).text
    return html, re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def fomc_meetings():
    """End day of every regular FOMC meeting (the statement comes out at 2pm that day)."""
    abbr = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
    # "January 28-29", "April 30-May 1", or across months in short form: "Jan/Feb 31-1"
    date = rf"(?:({MONTHS}) (\d{{1,2}})(?:-(?:({MONTHS}) )?(\d{{1,2}}))?|(?:{abbr})/({abbr}) \d{{1,2}}-(\d{{1,2}}))"
    skip = r"\((?:unscheduled|cancelled|notation vote)\)"

    def end_days(text, year, tail):
        out = []
        for m in re.finditer(date + rf"\*?(?![\d,-])( {skip})?{tail}", text):   # not "January 25-26, 2028" notes
            if m.group(7):
                continue                                  # emergency / cancelled: not known in advance
            if m.group(5):
                out.append(pd.Timestamp(f"{m.group(5)} {m.group(6)} {year}"))
            else:
                out.append(pd.Timestamp(f"{m.group(3) or m.group(1)} {m.group(4) or m.group(2)} {year}"))
        return out

    days = []
    _, t = page_text("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
    on_calendar = []
    for sec in re.finditer(r"(\d{4}) FOMC Meetings(.*?)(?=\d{4} FOMC Meetings|Last Update)", t):
        on_calendar.append(int(sec.group(1)))
        days += end_days(sec.group(2), int(sec.group(1)), "")
    for year in range(2013, min(on_calendar)):             # older years: history pages
        _, t = page_text(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm")
        days += end_days(t, year, " Meeting")
    return sorted({d.strftime("%Y-%m-%d") for d in days})


def bls_dates(name):
    """Release dates of a BLS report: past ones from its archive, upcoming from its schedule."""
    html, _ = page_text(f"https://www.bls.gov/bls/news-release/{name}.htm")
    past = {f"{y}-{m}-{d}" for m, d, y in re.findall(rf"archives/{name}_(\d{{2}})(\d{{2}})(\d{{4}})\.htm", html)}
    _, t = page_text(f"https://www.bls.gov/schedule/news_release/{name}.htm")
    upcoming = {pd.Timestamp(f"{m} {d} {y}").strftime("%Y-%m-%d")
                for m, d, y in re.findall(r"\b([A-Z][a-z]{2})\.? (\d{1,2}), (\d{4})\s+08:30", t)}
    return sorted(d for d in past | upcoming if d >= FIRST)


def macro_events():
    sessions = mcal.get_calendar("NYSE").schedule(start_date=FIRST, end_date="2027-12-31").index
    rows = []
    for kind, days in (("fomc", fomc_meetings()), ("cpi", bls_dates("cpi")), ("jobs", bls_dates("empsit"))):
        for d in days:
            if d < FIRST:
                continue
            # released while the market is closed (e.g. a jobs report on Good Friday) -> next session
            reaction = sessions[sessions >= pd.Timestamp(d)][0].strftime("%Y-%m-%d")
            rows.append({"type": kind, "release_day": d, "reaction_day": reaction})
    m = pd.DataFrame(rows).sort_values("reaction_day").reset_index(drop=True)
    # sanity checks: regular FOMC meetings are 8 a year, CPI and jobs reports about 12 a year
    per_year = m.assign(y=m["release_day"].str[:4]).groupby(["type", "y"]).size().unstack(0)
    full_years = per_year.loc[[y for y in per_year.index if FIRST[:4] < y < str(pd.Timestamp.now().year)]]
    expected = pd.Series(8, index=full_years.index).where(full_years.index != "2020", 7)   # March 2020 was cancelled
    assert full_years["fomc"].eq(expected).all(), f"FOMC meetings per year off:\n{full_years['fomc']}"
    assert full_years[["cpi", "jobs"]].stack().between(11, 13).all(), f"CPI/jobs per year off:\n{full_years}"
    return m


def splits():
    """Stock split dates and ratios (e.g. NVDA 10-for-1 on 2024-06-10), from Yahoo."""
    rows = []
    for t in ALL_TICKERS:
        for day, ratio in yf.Ticker(t).splits.items():
            if day.strftime("%Y-%m-%d") >= FIRST:
                rows.append({"ticker": t, "date": day.strftime("%Y-%m-%d"), "ratio": ratio})
    return pd.DataFrame(rows, columns=["ticker", "date", "ratio"])


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
    assert y["div_yield"].between(0, 0.08).all(), "dividend yield out of range -- check the data"
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

    m = macro_events()
    m.to_csv(OUT / "macro.csv", index=False)
    print(f"\nEconomic announcements: {len(m)}")
    print(m.assign(y=m["release_day"].str[:4]).groupby(["y", "type"]).size().unstack().to_string())
    moved = m[m["release_day"] != m["reaction_day"]]
    print("Released on a market holiday (moved to next session):", moved[["type", "release_day", "reaction_day"]].values.tolist())

    sp = splits()
    sp.to_csv(OUT / "splits.csv", index=False)
    print("\nSplits:", sp.values.tolist())

    d = dividends()
    d.to_csv(OUT / "dividends.csv", index=False)
    print(f"\nDividends: {len(d)} payments (tickers with none: "
          f"{sorted(set(ALL_TICKERS) - set(d['ticker']))})")
    print(d.groupby("ticker").agg(payments=("amount", "size"), last=("ex_date", "max"),
                                  last_amount=("amount", "last")).to_string())
