"""
Several kinds of model on the same two jobs, so the choice of model is tested, not assumed.
Each is trained on 2016-2021 and scored on 2022-today (never seen in training).

1. PRICING: predict each option's implied vol from the usual inputs (train.py) -> price error.
   Same random sample of rows for every model (keeps it light on a 17 GB laptop).
2. SIGNAL: rank tickers by next-5-day return from the option smile signals (stock_signals.py) ->
   average daily rank correlation with what happened.

Models: linear (ridge, on spline features), small neural net (MLP), random forest, two kinds of
gradient-boosted trees (LightGBM, sklearn's HistGradientBoosting).

Run:  python model_zoo.py
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

import train as tr

TRAIN_ROWS, TEST_ROWS = 1_000_000, 500_000


def zoo():
    return {
        "linear (ridge + splines)": make_pipeline(StandardScaler(), SplineTransformer(n_knots=6, degree=3), Ridge(alpha=1.0)),
        "neural net (MLP 64-64)": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), early_stopping=True,
                                                                               max_iter=60, random_state=0)),
        "random forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=50, max_features=0.5,
                                               n_jobs=-1, random_state=0),
        "LightGBM": tr.lgbm(),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=0),
    }


# ---------- 1. pricing ----------
need = list(dict.fromkeys(tr.NUMERIC + ["date", "ticker", "type", "strike", "spot", "T", "tbill", "div_yield",
                                         "iv", "iv_ratio", "atm_iv", "atm_used", "mid"]))
need = [c for c in need if c != "sqrt_T"]
q = pd.concat([pd.read_parquet(tr.Path(tr.__file__).parent / "models" / f"dataset_{g}.parquet", columns=need)
               .astype({"date": str, "ticker": str, "type": str}) for g in ("core", "extra")], ignore_index=True)
q = q[q["date"] >= "2016-01-01"]
train = q[q["date"] < "2022-01-01"].sample(TRAIN_ROWS, random_state=0)
test = q[(q["date"] >= "2022-01-01") & ~q["atm_used"]].sample(TEST_ROWS, random_state=0)
del q
tickers = sorted(train["ticker"].unique())
Xtr, Xte = tr.features(train, tickers).values, tr.features(test, tickers).values
print(f"PRICING: train {len(train):,} options (2016-2021), test {len(test):,} (2022-today), {Xtr.shape[1]} inputs")
rows = []
for name, m in zoo().items():
    t0 = time.time()
    m.fit(Xtr, train["iv_ratio"].values)
    iv = np.clip(m.predict(Xte) * test["atm_iv"].values, 0.02, 3)
    px = tr.price_of(test, iv)
    rows.append({"model": name, "IV error (vol pts)": np.median(np.abs(iv - test["iv"].values)) * 100,
                 "price error %": np.median(np.abs(px - test["mid"].values) / test["mid"].values) * 100,
                 "minutes": (time.time() - t0) / 60})
    print(f"  {name}: done", flush=True)
print(pd.DataFrame(rows).sort_values("price error %").to_string(index=False, float_format=lambda x: f"{x:.2f}"))
del train, test, Xtr, Xte

# ---------- 2. signal ----------
SIG = ["put_skew", "call_skew", "vrp", "term", "d_put_skew", "d_atm"]
hist = pd.read_csv(tr.DATA / "signals" / "history.csv")
hist = hist[hist["source"] == "databento"].sort_values(["ticker", "date"])
g = hist.groupby("ticker")
hist["d_put_skew"] = hist["put_skew"] - g["put_skew"].shift(5)
hist["d_atm"] = np.log(hist["atm"] / g["atm"].shift(5))
s = pd.read_csv(tr.DATA / "databento" / "underlying.csv").sort_values(["ticker", "date"])
s["ret"] = s.groupby("ticker")["close"].transform(lambda c: np.log(c / c.shift(1)))
s = s.merge(pd.read_csv(tr.DATA / "events" / "splits.csv"), on=["ticker", "date"], how="left")
s["ret"] += np.log(s["ratio"].fillna(1.0))
s["next5"] = s.groupby("ticker")["ret"].transform(lambda r: r.rolling(5).sum().shift(-5))
d = hist.merge(s[["ticker", "date", "next5"]], on=["ticker", "date"]).dropna(subset=SIG + ["next5"])
d["y"] = d["next5"] - d.groupby("date")["next5"].transform("mean")
tr_, te = d[(d["date"] >= "2014-01-01") & (d["date"] < "2021-12-24")], d[d["date"] >= "2022-01-01"]   # gap: no overlap
print(f"\nSIGNAL: train {len(tr_):,} ticker-days (2014-2021), test {len(te):,} (2022-today)")
rows = []
for name, m in zoo().items():
    if name == "LightGBM":                                     # the small version stock_signals.py uses
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.02, num_leaves=7, min_child_samples=200,
                              subsample=0.8, subsample_freq=1, verbose=-1, random_state=0)
    m.fit(tr_[SIG].values, tr_["y"].values)
    te = te.assign(fc=m.predict(te[SIG].values))
    ic = te.groupby("date").apply(lambda x: x["fc"].rank().corr(x["next5"].rank()))
    weekly = ic.iloc[::5]                                      # every 5th day: non-overlapping 5-day outcomes
    rows.append({"model": name, "rank corr (avg)": ic.mean(), "t": weekly.mean() / weekly.std() * np.sqrt(len(weekly))})
print(pd.DataFrame(rows).sort_values("rank corr (avg)", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
