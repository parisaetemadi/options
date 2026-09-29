"""
Train the option pricing model on Databento's closing bid/ask quotes (data/databento/).

The model predicts each option's implied volatility (IV) from things known at the close:
  - the option: how far the strike is from the stock price, time to expiry
  - the stock: how much it has been moving lately (5/21/63 days), recent returns
  - the market: yesterday's VIX ("fear gauge"), and which ticker it is
Plug the IV into Black-Scholes (with the real T-bill rate that day) and you get a price.

Walk-forward test: train on older dates, test on the newest dates the model never saw.
Two models are compared: a neural net and LightGBM (gradient-boosted trees).

Run:  python train.py
"""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pricing import bs_price, implied_vol_many

DATA = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "models" / "iv_model_databento.joblib"
TEST_SHARE = 0.2            # newest 20% of dates are the test set
MAX_SPREAD = 0.5            # skip quotes whose bid/ask spread is over 50% of the price (unreliable)
NN_MAX_ROWS = 400_000       # the neural net trains on a random sample (it's slow on millions of rows)


# ---------- 1. Market data: VIX and T-bill rate (free, Yahoo) ----------
def market_data():
    path = DATA / "market" / "vix_rates.csv"
    frames = {}
    for name, sym in (("vix", "^VIX"), ("tbill", "^IRX")):
        s = yf.Ticker(sym).history(start="2013-01-01", auto_adjust=False)["Close"]
        s.index = s.index.tz_localize(None).strftime("%Y-%m-%d")
        frames[name] = s
    m = pd.DataFrame(frames).dropna().rename_axis("date").reset_index()
    m["tbill"] = m["tbill"] / 100                      # Yahoo gives 4.5 for 4.5%
    # sanity checks: VIX has never been outside ~9-90, T-bill rates 0-6% in this period
    assert m["vix"].between(8, 90).all(), "VIX value out of range -- check the data"
    assert m["tbill"].between(-0.01, 0.07).all(), "T-bill rate out of range -- check the data"
    path.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(path, index=False)
    m["vix_prev"] = m["vix"].shift(1)                   # yesterday's VIX: known before today's close
    return m[["date", "vix_prev", "tbill"]]


# ---------- 2. Stock features, using only the past ----------
def stock_features():
    s = pd.read_csv(DATA / "databento" / "underlying.csv").sort_values(["ticker", "date"])
    g = s.groupby("ticker")["close"]
    s["ret"] = g.transform(lambda c: np.log(c / c.shift(1)))
    for days in (5, 21, 63):
        s[f"rv{days}"] = s.groupby("ticker")["ret"].transform(lambda r: r.rolling(days).std() * np.sqrt(252))
    s["ret5"] = g.transform(lambda c: np.log(c / c.shift(5)))
    s["ret21"] = g.transform(lambda c: np.log(c / c.shift(21)))
    return s.drop(columns=["close", "ret"])


# ---------- 3. Load quotes, clean them, back out IV ----------
files = sorted((DATA / "databento" / "quotes").glob("*/*.parquet"))
q = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
q = q.astype({"date": str, "ticker": str, "type": str, "expiration": str})
n0 = len(q)
q["mid"] = (q["bid"] + q["ask"]) / 2
q = q[(q["bid"] > 0) & (q["ask"] > q["bid"]) & (q["mid"] >= 0.05)
      & ((q["ask"] - q["bid"]) / q["mid"] <= MAX_SPREAD)]
print(f"Quotes: {n0:,} loaded, {len(q):,} clean ({len(files)} days, {q['date'].min()} to {q['date'].max()})")

q = q.merge(stock_features(), on=["ticker", "date"]).merge(market_data(), on="date")
q["T"] = (pd.to_datetime(q["expiration"]) - pd.to_datetime(q["date"])).dt.days / 365
q["moneyness"] = q["strike"] / q["spot"]

ivs = []
for i in range(0, len(q), 1_000_000):             # in chunks to keep memory small
    c = q.iloc[i:i + 1_000_000]
    ivs.append(implied_vol_many(c["mid"], c["spot"], c["strike"], c["T"], c["tbill"].values, c["type"]))
q["iv"] = np.concatenate(ivs)
q = q[q["iv"].between(0.02, 3)].dropna(subset=["rv63", "vix_prev"]).reset_index(drop=True)
print(f"Usable options with an implied vol: {len(q):,}")


# ---------- 4. Walk-forward split ----------
dates = np.sort(q["date"].unique())
cutoff = dates[int(len(dates) * (1 - TEST_SHARE))]
train, test = q[q["date"] < cutoff], q[q["date"] >= cutoff].copy()
print(f"Train: {len(train):,} options before {cutoff}   Test: {len(test):,} options from {cutoff}\n")

TICKERS = sorted(q["ticker"].unique())
FEATURES = ["log_m", "sqrt_T", "rv5", "rv21", "rv63", "ret5", "ret21", "vix_prev"] + [f"is_{t}" for t in TICKERS]

def features(d):
    X = pd.DataFrame({"log_m": np.log(d["moneyness"]), "sqrt_T": np.sqrt(d["T"])}, index=d.index)
    for col in ["rv5", "rv21", "rv63", "ret5", "ret21", "vix_prev"]:
        X[col] = d[col]
    for t in TICKERS:
        X[f"is_{t}"] = (d["ticker"] == t).astype(float)
    return X[FEATURES]


# ---------- 5. Train both models ----------
models = {}
sample = train.sample(min(len(train), NN_MAX_ROWS), random_state=0)
models["neural net"] = make_pipeline(StandardScaler(), MLPRegressor(
    hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=200, early_stopping=True, random_state=0)
).fit(features(sample), sample["iv"])
models["LightGBM"] = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                                       min_child_samples=200, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.9, verbose=-1, random_state=0
                                       ).fit(features(train), train["iv"])


# ---------- 6. Test on unseen dates ----------
def score(iv_pred):
    price = bs_price(test["type"].values, test["spot"], test["strike"], test["T"], test["tbill"].values, iv_pred)
    return {"IV error": np.median(np.abs(iv_pred - test["iv"])),
            "price error": np.median(np.abs(price - test["mid"]) / test["mid"]),
            "inside bid/ask": np.mean((price >= test["bid"]) & (price <= test["ask"]))}

results = {"baseline (Black-Scholes, 21-day vol)": score(test["rv21"].values)}
for name, m in models.items():
    results[name] = score(np.clip(m.predict(features(test)), 0.02, 3))
table = pd.DataFrame(results).T
print("Results on unseen dates (median errors; 'inside bid/ask' = share of prices a trader could actually deal at):")
print(table.to_string(formatters={c: "{:.1%}".format for c in table.columns}))

best = min(models, key=lambda k: results[k]["price error"])
test["iv_pred"] = np.clip(models[best].predict(features(test)), 0.02, 3)
test["price_pred"] = bs_price(test["type"].values, test["spot"], test["strike"], test["T"], test["tbill"].values, test["iv_pred"])
by = test.groupby("ticker").apply(lambda g: pd.Series({
    "options": len(g),
    "IV error": np.median(np.abs(g["iv_pred"] - g["iv"])),
    "price error": np.median(np.abs(g["price_pred"] - g["mid"]) / g["mid"]),
    "inside bid/ask": np.mean((g["price_pred"] >= g["bid"]) & (g["price_pred"] <= g["ask"]))}), include_groups=False)
print(f"\nBest model: {best}. Per ticker:")
print(by.to_string(formatters={"options": "{:,.0f}".format, **{c: "{:.1%}".format for c in by.columns[1:]}}))

MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump({"model": models[best], "name": best, "features": FEATURES, "tickers": TICKERS, "trained_until": cutoff},
            MODEL_PATH)
print(f"\nSaved {best} to {MODEL_PATH}")
