"""
Honest predictions for every year: for each year Y, train the model only on data BEFORE Y,
then predict Y. Stitched together this gives out-of-sample predictions 2016 -> today, so a
strategy can be chosen on the early years and confirmed once on the later ones.

Saves models/walkforward_predictions.parquet (same columns as train.py's test predictions).

Run:  python walkforward.py          (after train.py has built models/dataset.parquet)
"""
from pathlib import Path

import numpy as np
import pandas as pd

import train as tr

FIRST_TEST_YEAR = 2016          # needs ~3 years of history before it to learn from
MAX_TRAIN_ROWS = 4_000_000      # random sample per year keeps each fit to a few minutes
OUT = Path(__file__).parent / "models" / "walkforward_predictions.parquet"
COLS = ["date", "ticker", "type", "strike", "expiration", "spot", "bid", "ask", "mid", "T", "tbill",
        "div_yield", "iv", "iv_pred", "atm_iv", "atm_used"]

q = tr.load_dataset()
tickers = sorted(q["ticker"].unique())
years = range(FIRST_TEST_YEAR, int(q["date"].max()[:4]) + 1)
parts, rows = [], []
for y in years:
    past = q[q["date"] < f"{y}-01-01"]
    fit_on = past.sample(min(len(past), MAX_TRAIN_ROWS), random_state=y)
    model = tr.lgbm().fit(tr.features(fit_on, tickers), fit_on["iv_ratio"])
    test = q[q["date"].str[:4] == str(y)].copy()
    test["iv_pred"] = tr.predict(model, test, tickers)
    s = test[~test["atm_used"]]
    price = tr.price_of(s, s["iv_pred"])
    rows.append({"year": y, "trained on": len(past), "tested on": len(s),
                 "IV error": np.median(np.abs(s["iv_pred"] - s["iv"])),
                 "price error": np.median(np.abs(price - s["mid"]) / s["mid"])})
    print(f"{y}: trained on {len(past):,} options, tested on {len(s):,}: "
          f"IV error {rows[-1]['IV error']:.2%}, price error {rows[-1]['price error']:.1%}", flush=True)
    parts.append(test[COLS])

pd.concat(parts, ignore_index=True).to_parquet(OUT, index=False)
print(f"\nSaved {OUT}")
