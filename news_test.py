"""
Round 2 of HYPOTHESES.md: three news ideas from GDELT (fetch_gdelt.py), each tested ONCE as written there.

  N1 attention spike     share of news >= 3x its 60-day median (and >= 20 articles) -> buy the near-the-money pair
  N2 very negative tone  tone <= 5th percentile of the past 250 days (and normal-or-more coverage) -> buy the near-the-money put
  N3 market-stress news  stress-topic share >= 2x its 60-day median -> buy the SPY/QQQ/IWM near-the-money pair

Day D's news is used at the close of the first trading day after D (GDELT days end after the close);
sold at the next close. Pass = t >= 2.4 and average > 0 in both halves. Same machinery as hypotheses_test.py.

Run:  python news_test.py
"""
import numpy as np
import pandas as pd

import hypotheses_test as h

h.T_PASS = 2.4                      # 3 tests (Bonferroni)
all_days = h.all_days
STOCKS = ["AAL", "AAPL", "AMD", "AMZN", "AVGO", "BA", "BABA", "BAC", "CRM", "F", "GME", "GOOGL", "INTC", "META", "MRVL",
          "MSFT", "MSTR", "MU", "NFLX", "NKE", "NOK", "NOW", "NVDA", "ORCL", "PFE", "PYPL", "QCOM", "SMCI", "T", "TSLA",
          "TSM", "UNH", "WMT", "XOM"]

g = pd.read_csv(h.DATA / "news" / "gdelt_daily.csv").sort_values(["name", "date"])
g["med_share"] = g.groupby("name")["share"].transform(lambda s: s.shift(1).rolling(60, min_periods=30).median())
g["med_count"] = g.groupby("name")["count"].transform(lambda s: s.shift(1).rolling(60, min_periods=30).median())
g["tone_q05"] = g.groupby("name")["tone"].transform(lambda s: s.shift(1).rolling(250, min_periods=100).quantile(0.05))
print(f"GDELT: {g['name'].nunique()} names, {g['date'].min()} .. {g['date'].max()}; "
      f"days with no GDELT total (outages): {(g['norm'].fillna(0) == 0).groupby(g['name']).mean().max():.1%} at most")


def signal_days(rows, tickers):
    """GDELT signal days -> (ticker, entry = first trading day after D, exit = the next trading day)."""
    i = np.searchsorted(all_days, rows["date"].values, side="right")
    s = pd.DataFrame({"ticker": rows["ticker"].values, "news_day": rows["date"].values, "i": i})
    s = s[(s["i"] + 1 < len(all_days)) & s["ticker"].isin(tickers)]
    s["entry"], s["exit"] = all_days[s["i"]], all_days[s["i"] + 1]
    return s.drop_duplicates(["ticker", "entry"])          # a weekend's signals share one entry


def trades_for(sig, legs_of):
    trades, skipped = [], {"no data": 0, "no legs": 0, "split": 0}
    for yr, sy in sig.groupby(sig["entry"].str[:4]):
        q = h.quotes(sy["entry"].unique(), sy["ticker"].unique()).set_index(["date", "ticker"]).sort_index()
        for _, e in sy.iterrows():
            if (e["entry"], e["ticker"]) not in q.index:
                skipped["no data"] += 1
                continue
            if h.splits_between(e["ticker"], e["entry"], e["exit"]):
                skipped["split"] += 1
                continue
            x = q.loc[(e["entry"], e["ticker"])].reset_index()
            x = x[x["expiration"] >= (pd.Timestamp(e["exit"]) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")]
            legs = legs_of(x[x["expiration"] == x["expiration"].min()]) if len(x) else None
            if not legs:
                skipped["no legs"] += 1
                continue
            trades.append({"id": f"{e['ticker']} {e['entry']}", "date": e["entry"], "exit": e["exit"], "legs": legs})
    print(f"  {len(sig)} signals, {len(trades)} traded, skipped {skipped}")
    return h.run_daily(trades)


def pair(x):
    p = h.near_pair(x)
    return [(p[0], 1), (p[1], 1)] if p else None


def put_only(x):
    p = h.near_pair(x)
    return [(p[0], 1)] if p else None


stocks = g[g["name"].isin(STOCKS)].rename(columns={"name": "ticker"})
res = []

n1 = stocks[(stocks["share"] >= 3 * stocks["med_share"]) & (stocks["count"] >= 20)]
print(f"\nN1: {len(n1)} spike days ({len(n1) / stocks['med_share'].notna().sum():.1%} of stock-days)")
res.append(h.verdict("N1 news-attention spike (34 stocks, buy pair next close, sell the close after)",
                     trades_for(signal_days(n1, STOCKS), pair)))

n2 = stocks[(stocks["tone"] <= stocks["tone_q05"]) & (stocks["count"] >= stocks["med_count"])]
print(f"\nN2: {len(n2)} very-negative-tone days")
res.append(h.verdict("N2 very negative news tone (34 stocks, buy the near-the-money put next close, sell the close after)",
                     trades_for(signal_days(n2, STOCKS), put_only)))

st = g[g["name"] == "_STRESS"]
st = st[st["share"] >= 2 * st["med_share"]]
n3 = pd.concat([st.assign(ticker=t) for t in ["SPY", "QQQ", "IWM"]])
print(f"\nN3: {len(st)} market-stress spike days: {', '.join(st['date'].tolist()[:12])}{' ...' if len(st) > 12 else ''}")
res.append(h.verdict("N3 market-stress news spike (SPY/QQQ/IWM, buy pair next close, sell the close after)",
                     trades_for(signal_days(n3, ["SPY", "QQQ", "IWM"]), pair)))

print("\n===== SUMMARY =====")
print(pd.DataFrame(res).to_string(index=False))
