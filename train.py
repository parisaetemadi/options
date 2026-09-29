"""
Train the option pricing model on Databento's closing bid/ask quotes (data/databento/).

The model predicts each option's implied volatility (IV) from things known at the close:
  - today's at-the-money (ATM) IV for that stock and expiry, measured from the two options
    closest to the stock price -- the model learns the smile AROUND this level
  - the option: how far the strike is from the stock price, time to expiry
  - the stock: how much it has been moving lately (5/21/63 days), recent returns
  - the market: yesterday's VIX ("fear gauge"), and which ticker it is
  - scheduled events: days until the next earnings (or Tesla delivery report), Fed decision,
    CPI and jobs report, and whether each happens before the option expires
Plug the IV into Black-Scholes (with the real T-bill rate and the stock's dividend yield)
and you get a price. Run fetch_events.py first to refresh earnings/dividends.

Walk-forward test: train on older dates, test on the newest dates the model never saw.
Two models are compared: a neural net and LightGBM (gradient-boosted trees).

Run:  python train.py            (core tickers; add --all to include the extra group, --refresh to rebuild data)
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
    # prices here are the real traded ones (not split-adjusted), so undo the fake move on split days
    sp = pd.read_csv(DATA / "events" / "splits.csv")
    s = s.merge(sp, on=["ticker", "date"], how="left")
    s["ret"] += np.log(s["ratio"].fillna(1.0))
    s = s.drop(columns="ratio")
    for days in (5, 21, 63):
        s[f"rv{days}"] = s.groupby("ticker")["ret"].transform(lambda r: r.rolling(days).std() * np.sqrt(252))
    s["ret5"] = s.groupby("ticker")["ret"].transform(lambda r: r.rolling(5).sum())
    s["ret21"] = s.groupby("ticker")["ret"].transform(lambda r: r.rolling(21).sum())
    return s.drop(columns=["close", "ret"])


# ---------- 2b. Scheduled events (earnings, Tesla deliveries) and dividend yield ----------
NO_EVENT_DAYS = 120   # "no event coming up" (ETFs, or more than ~4 months away)

def add_events(q):
    ev = pd.read_csv(DATA / "events" / "earnings.csv")
    ev = ev[ev["type"].isin(["earnings", "deliveries"])]      # only events known in advance
    ev = ev.assign(event_day=pd.to_datetime(ev["reaction_day"]))[["ticker", "event_day"]].sort_values("event_day")
    q = q.assign(day=pd.to_datetime(q["date"])).sort_values("day")
    # next event strictly after today (today's reaction is already in today's prices)
    q = pd.merge_asof(q, ev, left_on="day", right_on="event_day", by="ticker",
                      direction="forward", allow_exact_matches=False)
    days = (q["event_day"] - q["day"]).dt.days
    q["days_to_event"] = days.fillna(NO_EVENT_DAYS).clip(upper=NO_EVENT_DAYS)
    q["event_before_expiry"] = (q["event_day"] <= pd.to_datetime(q["expiration"])).astype(float)
    # economic announcements (same for every ticker): Fed decisions, CPI, jobs reports
    mac = pd.read_csv(DATA / "events" / "macro.csv")
    for kind in ("fomc", "cpi", "jobs"):
        e = pd.DataFrame({"mday": pd.to_datetime(mac.loc[mac["type"] == kind, "reaction_day"])}).sort_values("mday")
        q = pd.merge_asof(q.sort_values("day"), e, left_on="day", right_on="mday",
                          direction="forward", allow_exact_matches=False)
        q[f"days_to_{kind}"] = (q["mday"] - q["day"]).dt.days.fillna(NO_EVENT_DAYS).clip(upper=NO_EVENT_DAYS)
        q[f"{kind}_before_expiry"] = (q["mday"] <= pd.to_datetime(q["expiration"])).astype(float)
        q = q.drop(columns="mday")
    dy = pd.read_csv(DATA / "events" / "div_yield.csv")
    q = q.merge(dy, on=["ticker", "date"], how="left")
    q["div_yield"] = q["div_yield"].fillna(0.0)                # AMZN, TSLA pay none
    return q.drop(columns=["day", "event_day"])


def load_dataset(groups=("core",), refresh=False):
    """All clean quotes with IV, ATM level and every model input, for these ticker groups
    (see download_databento.py). Cached in models/dataset_<groups>.parquet; refresh=True rebuilds
    it, e.g. after new data is downloaded."""
    CACHE = Path(__file__).parent / "models" / f"dataset_{'_'.join(groups)}.parquet"
    if CACHE.exists() and not refresh:
        q = pd.read_parquet(CACHE)
        return q.astype({c: str for c in ("date", "ticker", "type", "expiration")})
    # ---------- 3. Load quotes, clean them, back out IV ----------
    files = sorted(f for g in groups for f in (DATA / "databento" / "quotes" / g).glob("*/*.parquet"))
    q = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    q = q.astype({"date": str, "ticker": str, "type": str, "expiration": str})
    n0 = len(q)
    q["mid"] = (q["bid"] + q["ask"]) / 2
    q = q[(q["bid"] > 0) & (q["ask"] > q["bid"]) & (q["mid"] >= 0.05)
          & ((q["ask"] - q["bid"]) / q["mid"] <= MAX_SPREAD)]
    print(f"Quotes: {n0:,} loaded, {len(q):,} clean ({len(files)} days, {q['date'].min()} to {q['date'].max()})")

    q = add_events(q.merge(stock_features(), on=["ticker", "date"]).merge(market_data(), on="date"))
    q["T"] = (pd.to_datetime(q["expiration"]) - pd.to_datetime(q["date"])).dt.days / 365
    q["moneyness"] = q["strike"] / q["spot"]

    ivs = []
    for i in range(0, len(q), 1_000_000):             # in chunks to keep memory small
        c = q.iloc[i:i + 1_000_000]
        ivs.append(implied_vol_many(c["mid"], c["spot"], c["strike"], c["T"], c["tbill"].values, c["type"],
                                    q=c["div_yield"].values))
    q["iv"] = np.concatenate(ivs)
    q = q[q["iv"].between(0.02, 3)].dropna(subset=["rv63", "vix_prev"]).reset_index(drop=True)
    print(f"Usable options with an implied vol: {len(q):,}")


    # ---------- 4. Today's at-the-money (ATM) volatility, per stock and expiry ----------
    # From the closest OTM put (just below the stock price) and closest OTM call (just above),
    # interpolated to exactly the stock price. The model then learns the smile AROUND this level.
    q["log_m"] = np.log(q["moneyness"])
    g = ["date", "ticker", "expiration"]
    put_i = q[q["type"] == "put"].groupby(g)["log_m"].idxmax()     # closest put below the price
    call_i = q[q["type"] == "call"].groupby(g)["log_m"].idxmin()   # closest call above the price
    pair = pd.DataFrame({"p": put_i, "c": call_i}).dropna().astype(int)
    lp, lc = q.loc[pair["p"], "log_m"].values, q.loc[pair["c"], "log_m"].values
    pair = pair[(lp > -0.05) & (lc < 0.05)]                         # both within 5% of the price
    lp, lc = q.loc[pair["p"], "log_m"].values, q.loc[pair["c"], "log_m"].values
    w = -lp / (lc - lp)
    pair["atm_iv"] = (1 - w) * q.loc[pair["p"], "iv"].values + w * q.loc[pair["c"], "iv"].values
    # the two options each ATM value is built from are excluded when scoring (they'd be "answered" already)
    q["atm_used"] = q.index.isin(np.r_[pair["p"].values, pair["c"].values])
    q = q.merge(pair[["atm_iv"]].reset_index(), on=g)
    # the 30-day ATM level for the stock (interpolated across expiries): the overall mood that day
    def atm30(d):
        d = d.sort_values("T")
        return np.sqrt(np.interp(30 / 365, d["T"], d["atm_iv"] ** 2 * d["T"]) / (30 / 365))
    lvl = q.drop_duplicates(g).groupby(["date", "ticker"])[["T", "atm_iv"]].apply(atm30).rename("atm30").reset_index()
    q = q.merge(lvl, on=["date", "ticker"])
    q["x"] = q["log_m"] / (q["atm_iv"] * np.sqrt(q["T"]))          # moneyness in "standard deviations"
    q["iv_ratio"] = q["iv"] / q["atm_iv"]                          # what the model predicts
    print(f"Options with an ATM level for their expiry: {len(q):,}")


    q = q.reset_index(drop=True)
    # keep it compact on a 17 GB laptop: 32-bit numbers (plenty of precision here) and categories
    for c in q.columns:
        if q[c].dtype == "float64":
            q[c] = q[c].astype("float32")
    for c in ("date", "ticker", "type", "expiration"):
        q[c] = q[c].astype("category")
    CACHE.parent.mkdir(exist_ok=True)
    q.to_parquet(CACHE, index=False)
    return q

NUMERIC = ["x", "log_m", "sqrt_T", "atm_iv", "atm30", "rv5", "rv21", "rv63", "ret5", "ret21", "vix_prev",
           "days_to_event", "event_before_expiry",
           "days_to_fomc", "fomc_before_expiry", "days_to_cpi", "cpi_before_expiry", "days_to_jobs", "jobs_before_expiry"]


def feature_names(tickers):
    return NUMERIC + [f"is_{t}" for t in tickers]


def features(d, tickers):
    X = d.assign(sqrt_T=np.sqrt(d["T"]))[NUMERIC].copy()
    for t in tickers:
        X[f"is_{t}"] = (d["ticker"] == t).astype(float)
    return X[feature_names(tickers)]


def lgbm():
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1, random_state=0)


def price_of(d, iv):
    return bs_price(d["type"].values, d["spot"], d["strike"], d["T"], d["tbill"].values, iv, d["div_yield"].values)


def predict(m, d, tickers):
    return np.clip(m.predict(features(d, tickers)) * d["atm_iv"].values, 0.02, 3)


if __name__ == "__main__":
    import sys
    groups = ("core", "extra") if "--all" in sys.argv else ("core",)
    q = load_dataset(groups, refresh="--refresh" in sys.argv)
    print(f"Dataset: {len(q):,} options, {q['date'].nunique():,} days ({q['date'].min()} to {q['date'].max()}), "
          f"tickers: {', '.join(sorted(q['ticker'].unique()))}")

    # ---------- 5. Walk-forward split ----------
    dates = np.sort(q["date"].unique())
    cutoff = dates[int(len(dates) * (1 - TEST_SHARE))]
    train, test = q[q["date"] < cutoff], q[q["date"] >= cutoff].copy()
    print(f"Train: {len(train):,} options before {cutoff}   Test: {len(test):,} options from {cutoff}\n")
    TICKERS = sorted(q["ticker"].unique())
    FEATURES = feature_names(TICKERS)


    # ---------- 6. Train both models (they predict IV / ATM IV: the smile shape) ----------
    models = {}
    sample = train.sample(min(len(train), NN_MAX_ROWS), random_state=0)
    models["neural net"] = make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=200, early_stopping=True, random_state=0)
    ).fit(features(sample, TICKERS), sample["iv_ratio"])
    models["LightGBM"] = lgbm().fit(features(train, TICKERS), train["iv_ratio"])


    # ---------- 7. Test on unseen dates (not counting the options used for the ATM level) ----------
    score_set = test[~test["atm_used"]]

    def score(d, iv_pred):
        price = price_of(d, iv_pred)
        return {"IV error": np.median(np.abs(iv_pred - d["iv"])),
                "price error": np.median(np.abs(price - d["mid"]) / d["mid"]),
                "inside bid/ask": np.mean((price >= d["bid"]) & (price <= d["ask"]))}

    results = {"baseline: Black-Scholes, 21-day vol": score(score_set, score_set["rv21"].values),
               "baseline: flat smile (ATM vol everywhere)": score(score_set, score_set["atm_iv"].values)}
    for name, m in models.items():
        results[name] = score(score_set, predict(m, score_set, TICKERS))
    table = pd.DataFrame(results).T
    print("Results on unseen dates (median errors; 'inside bid/ask' = share of prices a trader could actually deal at):")
    print(table.to_string(formatters={c: "{:.1%}".format for c in table.columns}))

    best = min(models, key=lambda k: results[k]["price error"])
    test["iv_pred"] = predict(models[best], test, TICKERS)
    test["price_pred"] = price_of(test, test["iv_pred"])
    s = test[~test["atm_used"]]
    by = s.groupby("ticker").apply(lambda d: pd.Series({
        "options": len(d),
        "IV error": np.median(np.abs(d["iv_pred"] - d["iv"])),
        "price error": np.median(np.abs(d["price_pred"] - d["mid"]) / d["mid"]),
        "inside bid/ask": np.mean((d["price_pred"] >= d["bid"]) & (d["price_pred"] <= d["ask"]))}), include_groups=False)
    print(f"\nBest model: {best}. Per ticker:")
    print(by.to_string(formatters={"options": "{:,.0f}".format, **{c: "{:.1%}".format for c in by.columns[1:]}}))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": models[best], "name": best, "features": FEATURES, "tickers": TICKERS, "trained_until": cutoff},
                MODEL_PATH)
    # test-period predictions, for backtest.py
    test[["date", "ticker", "type", "strike", "expiration", "spot", "bid", "ask", "mid", "T", "tbill", "div_yield",
          "iv", "iv_pred", "atm_iv", "atm_used"]].to_parquet(MODEL_PATH.with_name("test_predictions.parquet"), index=False)
    print(f"\nSaved {best} to {MODEL_PATH} (and its test-period predictions for backtesting)")
