"""
Do range-based volatilities (data/stocks/range_vol.csv, from daily open/high/low/close) forecast
how much a stock will REALLY move over the next 21 trading days better than close-only ones?

Same year-by-year forecast as vrp.py (each year trained only on earlier years), run twice:
without and with the new inputs. Errors are over 2016-today.

Run:  python test_vol_forecast.py
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

import train as tr

H = 21
NEW = ["rs5", "rs21", "rs63", "on_share21"]
BASE = ["rv5", "rv21", "rv63", "atm30", "vix_prev", "days_to_event", "days_to_fomc", "days_to_cpi", "days_to_jobs"]

q = pd.read_parquet(tr.Path(tr.__file__).parent / "models" / "dataset_core.parquet", columns=["date", "ticker"] + BASE)
day = q.astype({"date": str, "ticker": str}).drop_duplicates(["date", "ticker"])
del q

s = pd.read_csv(tr.DATA / "databento" / "underlying.csv").sort_values(["ticker", "date"])
s["ret"] = s.groupby("ticker")["close"].transform(lambda c: np.log(c / c.shift(1)))
s = s.merge(pd.read_csv(tr.DATA / "events" / "splits.csv"), on=["ticker", "date"], how="left")
s["ret"] += np.log(s["ratio"].fillna(1.0))
s["rv_fwd"] = s.groupby("ticker")["ret"].transform(lambda r: np.sqrt((r**2).rolling(H).sum().shift(-H) * 252 / H))
day = (day.merge(s[["ticker", "date", "rv_fwd"]], on=["ticker", "date"])
          .merge(pd.read_csv(tr.DATA / "stocks" / "range_vol.csv"), on=["ticker", "date"], how="left")
          .dropna(subset=["rv_fwd", "rv63", "atm30", "rs63"]))
tickers = sorted(day["ticker"].unique())
LOGS = ["rv5", "rv21", "rv63", "atm30", "rs5", "rs21", "rs63"]


def X(d, cols):
    num = d[cols].copy()
    for c in cols:
        if c in LOGS:
            num[c] = np.log(num[c].clip(lower=0.01))
    return np.column_stack([num, pd.get_dummies(d["ticker"]).reindex(columns=tickers, fill_value=0)])


def forecast(cols):
    out = pd.Series(np.nan, index=day.index)
    for y in range(2016, int(day["date"].max()[:4]) + 1):
        past = day[day["date"] < f"{y}-01-01"]
        past = past[past["date"] < sorted(past["date"].unique())[-H]]     # outcome known before year y
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                              verbose=-1, random_state=0).fit(X(past, cols), np.log(past["rv_fwd"]))
        cur = day["date"].str[:4] == str(y)
        out[cur] = m.predict(X(day[cur], cols))
    return out


base, new = forecast(BASE), forecast(BASE + NEW)
ok = base.notna()
d, truth = day[ok], np.log(day.loc[ok, "rv_fwd"])
for name, f in [("without", base[ok]), ("with", new[ok])]:
    print(f"{name:>8}: median error {np.median(np.abs(np.exp(f) - d['rv_fwd'])) * 100:.2f} vol points, "
          f"log-error {np.mean((f - truth) ** 2):.4f}")
print(f"implied vol itself: median error {np.median(np.abs(d['atm30'] - d['rv_fwd'])) * 100:.2f} vol points")
by = pd.DataFrame({"y": d["date"].str[:4], "b": (base[ok] - truth) ** 2, "n": (new[ok] - truth) ** 2}).groupby("y").mean()
print("log-error by year (without -> with):", {y: f"{r.b:.3f}->{r.n:.3f}" for y, r in by.iterrows()})
