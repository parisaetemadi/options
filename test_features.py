"""
Does a new data source earn its place? (We only add inputs that measurably help -- the training
data is already ~17 million rows on a 17 GB laptop.)

Trains the same LightGBM twice on 2016-2021 -- with and without the candidate inputs -- and
compares errors on 2022-today, overall and per ticker. Uses a random sample of rows to stay light.

Run:  python test_features.py data/attention/daily_features.csv wiki_views_log wiki_spike sec_8k sec_8k_20d
"""
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import train as tr

TRAIN_ROWS, TEST_ROWS = 3_000_000, 1_500_000
path, cols = sys.argv[1], sys.argv[2:]

need = list(dict.fromkeys(tr.NUMERIC + ["date", "ticker", "type", "strike", "spot", "T", "tbill", "div_yield",
                                         "iv", "iv_ratio", "atm_iv", "atm_used", "mid", "sqrt_T"]))
have = pq.ParquetFile(tr.Path(tr.__file__).parent / "models" / "dataset_core.parquet").schema.names
q = pd.read_parquet(tr.Path(tr.__file__).parent / "models" / "dataset_core.parquet", columns=[c for c in need if c in have])
q = q.astype({"date": str, "ticker": str, "type": str})
extra = pd.read_csv(path)
if "wiki_spike" in extra:
    extra["wiki_spike"] = extra["wiki_spike"].clip(upper=10)       # cap bot-traffic spikes
q = q.merge(extra[["ticker", "date"] + cols], on=["ticker", "date"], how="left")
q = q[q["date"] >= "2016-01-01"]
train = q[q["date"] < "2022-01-01"].sample(TRAIN_ROWS, random_state=0)
test = q[(q["date"] >= "2022-01-01") & ~q["atm_used"]].sample(TEST_ROWS, random_state=0)
tickers = sorted(q["ticker"].unique())


def fit_and_score(extra_cols):
    X = lambda d: pd.concat([tr.features(d, tickers), d[extra_cols]], axis=1)
    m = tr.lgbm().fit(X(train), train["iv_ratio"])
    iv = np.clip(m.predict(X(test)) * test["atm_iv"].values, 0.02, 3)
    price = tr.price_of(test, iv)
    err_iv = np.abs(iv - test["iv"].values)
    err_px = np.abs(price - test["mid"].values) / test["mid"].values
    by = pd.DataFrame({"t": test["ticker"].values, "iv": err_iv}).groupby("t")["iv"].median()
    return np.median(err_iv), np.median(err_px), by


base = fit_and_score([])
new = fit_and_score(cols)
print(f"Candidate inputs: {', '.join(cols)}  (tested on {len(test):,} options, 2022-today)")
print(f"IV error:    without {base[0]:.3%}   with {new[0]:.3%}   change {new[0] - base[0]:+.3%}")
print(f"price error: without {base[1]:.2%}   with {new[1]:.2%}   change {new[1] - base[1]:+.2%}")
print("IV error by ticker (without -> with):",
      {t: f"{base[2][t]:.2%} -> {new[2][t]:.2%}" for t in base[2].index})
