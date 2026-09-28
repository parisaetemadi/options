"""
Train the option pricing model on the Massive history in data/massive/.

The model predicts an option's implied volatility (IV) using only things we know
that morning: how far the strike is from the stock price, time to expiry, how much
the stock has been moving lately, and which ticker it is. Plug that IV into
Black-Scholes and you get a price.

Testing is done "walk-forward": train on older dates, test on the newest dates
the model has never seen, the same way it would be used for trading.

Run:  python train_massive.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pricing import bs_price, implied_vol

DATA = Path(__file__).parent / "data" / "massive"
MODEL_PATH = Path(__file__).parent / "models" / "iv_model.joblib"
RATE = 0.04           # rough interest rate over 2024-2026
MIN_VOLUME = 10       # skip options that barely traded that day (stale prices)
TEST_SHARE = 0.2      # newest 20% of dates are the test set


# ---------- 1. Load and join the three files ----------
contracts = pd.read_csv(DATA / "contracts.csv")
bars = pd.concat([pd.read_csv(p) for p in sorted((DATA / "bars").glob("*.csv"))], ignore_index=True)
stock = pd.read_csv(DATA / "underlying.csv").sort_values(["ticker", "date"])

# How much each stock has been moving, using only past days (no peeking ahead)
stock["ret"] = stock.groupby("ticker")["close"].transform(lambda c: np.log(c / c.shift(1)))
for days in (5, 21, 63):
    stock[f"rv{days}"] = stock.groupby("ticker")["ret"].transform(
        lambda r: r.rolling(days).std() * np.sqrt(252))
stock["ret5"] = stock.groupby("ticker")["close"].transform(lambda c: np.log(c / c.shift(5)))

df = (bars.merge(contracts, on="contract")
          .merge(stock.rename(columns={"close": "spot"}), on=["ticker", "date"]))
df["T"] = (pd.to_datetime(df["expiration"]) - pd.to_datetime(df["date"])).dt.days / 365
df["moneyness"] = df["strike"] / df["spot"]
print(f"Daily option prices loaded: {len(df):,}")


# ---------- 2. Keep clean, out-of-the-money, actually-traded options ----------
otm = ((df["type"] == "call") & (df["moneyness"] > 1)) | ((df["type"] == "put") & (df["moneyness"] < 1))
df = df[otm
        & df["moneyness"].between(0.8, 1.2)
        & df["T"].between(7 / 365, 1)
        & (df["volume"] >= MIN_VOLUME)
        & (df["close"] >= 0.05)].dropna(subset=["rv63"]).copy()

# Back out the market's implied volatility from each closing price
df["iv"] = [implied_vol(p, s, k, t, RATE, kind)
            for p, s, k, t, kind in zip(df["close"], df["spot"], df["strike"], df["T"], df["type"])]
df = df[df["iv"].between(0.02, 3)].reset_index(drop=True)
print(f"Usable OTM options: {len(df):,}  ({df['date'].min()} to {df['date'].max()})")


# ---------- 3. Walk-forward split: old dates train, newest dates test ----------
dates = np.sort(df["date"].unique())
cutoff = dates[int(len(dates) * (1 - TEST_SHARE))]
train, test = df[df["date"] < cutoff], df[df["date"] >= cutoff]
print(f"Train: {len(train):,} options before {cutoff}   Test: {len(test):,} options from {cutoff}")


# ---------- 4. Train ----------
TICKERS = sorted(df["ticker"].unique())

def features(d):
    one_hot = np.column_stack([(d["ticker"] == t).astype(float) for t in TICKERS])
    return np.column_stack([np.log(d["moneyness"]), np.sqrt(d["T"]),
                            d["rv5"], d["rv21"], d["rv63"], d["ret5"], one_hot])

model = make_pipeline(StandardScaler(),
                      MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=500,
                                   early_stopping=True, random_state=0))
model.fit(features(train), train["iv"])


# ---------- 5. Test on the unseen newest dates ----------
test = test.assign(iv_pred=model.predict(features(test)).clip(0.02))
test["price_pred"] = bs_price(test["type"].values, test["spot"], test["strike"], test["T"], RATE, test["iv_pred"])
# Baseline to beat: Black-Scholes using the stock's recent (21-day) volatility
test["price_base"] = bs_price(test["type"].values, test["spot"], test["strike"], test["T"], RATE, test["rv21"])

def report(g):
    return pd.Series({
        "options": len(g),
        "IV error": (g["iv_pred"] - g["iv"]).abs().median(),
        "model price error": ((g["price_pred"] - g["close"]).abs() / g["close"]).median(),
        "baseline price error": ((g["price_base"] - g["close"]).abs() / g["close"]).median(),
    })

pct = "{:.1%}".format
fmt = {"IV error": pct, "model price error": pct, "baseline price error": pct}
print("\nResults on unseen dates (median errors):")
print(report(test).to_frame("all").T.to_string(formatters=fmt))
print(test.groupby("ticker").apply(report, include_groups=False).to_string(formatters=fmt))

MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump({"model": model, "tickers": TICKERS, "rate": RATE}, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
