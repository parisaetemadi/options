"""
Step 2: Real option data from Yahoo Finance -- out-of-the-money options only.
Trains ONE neural net on all tickers together; each stock's historical vol is an input.
Install once:  pip install numpy scipy pandas scikit-learn matplotlib yfinance
Run (during US market hours, 9:30-16:00 New York time):  python real_options.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]   # highly liquid options
RATE = 0.04        # rough interest rate; close enough for learning
SMOOTHING = 0.01   # neural net: higher = smoother curve, lower = follows data more closely
MIN_OPTIONS = 20   # skip a ticker if fewer usable quotes than this (e.g. market closed)


# ---------- Black-Scholes + backing out IV ----------
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol(price, S, K, T, r, kind):
    """Find the volatility that makes Black-Scholes match the market price."""
    f = bs_call if kind == "call" else bs_put
    try:
        return brentq(lambda s: f(S, K, T, r, s) - price, 0.001, 5.0)
    except ValueError:
        return np.nan   # no volatility fits (bad or stale price)


# ---------- 0. Warn if the market is closed (quotes will be stale) ----------
now_ny = pd.Timestamp.now(tz="America/New_York")
market_open = now_ny.replace(hour=9, minute=30, second=0)
market_close = now_ny.replace(hour=16, minute=0, second=0)
if now_ny.weekday() >= 5 or not market_open <= now_ny <= market_close:
    print(f"WARNING: US market is closed ({now_ny:%a %H:%M} New York time). "
          "Bid/ask quotes may be stale or zero -- run during market hours for real results.\n")


# ---------- 1. Download data for one ticker ----------
def load_options(ticker):
    tk = yf.Ticker(ticker)
    S = tk.history(period="1d")["Close"].iloc[-1]

    # Historical volatility: how much it actually moved over the past year
    closes = tk.history(period="1y")["Close"]
    daily_returns = np.log(closes / closes.shift(1)).dropna()
    hist_vol = daily_returns.std() * np.sqrt(252)   # 252 trading days per year

    # Option chains for every expiration between 1 week and 1 year out.
    # Out-of-the-money only: puts below the current price, calls above it.
    # OTM options trade more and their prices are almost pure volatility, so the IV is cleaner.
    today = pd.Timestamp.today().normalize()
    rows = []
    for exp in tk.options:
        T = (pd.Timestamp(exp) - today).days / 365
        if T < 7 / 365 or T > 1:
            continue
        chain = tk.option_chain(exp)
        for kind, table in (("put", chain.puts), ("call", chain.calls)):
            for _, c in table.iterrows():
                if c["bid"] <= 0 or c["ask"] <= 0:
                    continue                          # no real market
                m = c["strike"] / S                   # moneyness
                if not 0.8 <= m <= 1.2:
                    continue                          # skip far-away strikes
                if (kind == "call" and m <= 1) or (kind == "put" and m >= 1):
                    continue                          # in-the-money: skip
                mid = (c["bid"] + c["ask"]) / 2
                rows.append({"exp": exp, "T": T, "type": kind, "strike": c["strike"],
                             "moneyness": m, "price": mid})

    df = pd.DataFrame(rows, columns=["exp", "T", "type", "strike", "moneyness", "price"])
    df["iv"] = [implied_vol(p, S, k, t, RATE, kind)
                for p, k, t, kind in zip(df["price"], df["strike"], df["T"], df["type"])]
    df = df.dropna().sort_values(["exp", "strike"]).reset_index(drop=True)
    return S, hist_vol, df


# ---------- 2. Download every ticker into one table ----------
tables, spots = [], {}
for ticker in TICKERS:
    print(f"Downloading {ticker}...")
    S, hist_vol, df = load_options(ticker)
    print(f"  {len(df)} OTM options ({(df['type'] == 'put').sum()} puts, "
          f"{(df['type'] == 'call').sum()} calls), historical vol {hist_vol:.1%}")
    if len(df) < MIN_OPTIONS:
        print("  too few usable quotes -- skipping")
        continue
    df["ticker"], df["hist_vol"] = ticker, hist_vol
    tables.append(df)
    spots[ticker] = S

if not tables:
    raise SystemExit("\nNo ticker had enough live quotes -- run during market hours.")
data = pd.concat(tables, ignore_index=True)
print(f"\nTotal options for training: {len(data)}")


# ---------- 3. One neural net for all tickers: learn IV from (moneyness, time, stock's vol) ----------
def features(d):
    # log-moneyness and sqrt(time) make the pattern easier to learn;
    # historical vol tells the net how jumpy this particular stock is
    return np.column_stack([np.log(d["moneyness"]), np.sqrt(d["T"]), d["hist_vol"]])

train_df, test_df = train_test_split(data, test_size=0.2, random_state=0)
model = make_pipeline(StandardScaler(),   # rescales inputs to similar sizes
                      MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=5000,
                                   alpha=SMOOTHING, solver="lbfgs", random_state=0))
model.fit(features(train_df), train_df["iv"])

test_df = test_df.assign(err=np.abs(model.predict(features(test_df)) - test_df["iv"]))
print(f"\nNeural net IV error on unseen options: avg {test_df['err'].mean():.2%}, "
      f"worst {test_df['err'].max():.2%}")
print("Per ticker:")
print(test_df.groupby("ticker", sort=False)["err"].agg(["mean", "max", "count"])
      .rename(columns={"mean": "avg_err", "max": "worst_err", "count": "test_options"})
      .to_string(formatters={"avg_err": "{:.2%}".format, "worst_err": "{:.2%}".format}))


# ---------- 4. Plot each ticker's smile: market vs the one neural net ----------
tickers = list(spots)
cols = 4
rows = -(-len(tickers) // cols)   # round up
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
for ax, ticker in zip(axes.flat, tickers):
    df = data[data["ticker"] == ticker]
    one = df[df["exp"] == df["exp"].unique()[len(df["exp"].unique()) // 3]]   # ~1-3 months out
    ax.plot(one["strike"], one["iv"], "o", ms=3, label="Market IV")
    ax.plot(one["strike"], model.predict(features(one)), "-", label="Neural net")
    ax.axhline(one["hist_vol"].iloc[0], color="gray", ls="--", label="Historical vol")
    ax.axvline(spots[ticker], color="red", ls=":", label="Current price")
    ax.set_title(f"{ticker}, expiring {one['exp'].iloc[0]}", fontsize=10)
for ax in list(axes.flat)[len(tickers):]:
    ax.axis("off")
axes.flat[0].legend(fontsize=7)
fig.supxlabel("Strike"); fig.supylabel("Implied volatility")
plt.tight_layout()
plt.show()
