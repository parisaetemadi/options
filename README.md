# Options

Goal: build a model that prices options well enough to trade them.

## Files

| File | What it does |
|---|---|
| `pricing.py` | Black-Scholes prices and implied volatility, shared by every script |
| `download_massive.py` | Downloads ~2 years of daily option prices from Massive (free plan, 5 calls/min). Runs by itself on GitHub every 6 hours |
| `live_iv_model.py` | Pulls today's option quotes from Yahoo and trains one neural net on all tickers (run during market hours) |
| `learn_bs_neural_net.py` | Practice: a neural net learns the Black-Scholes formula on made-up data |
| `data/` | Downloaded history (written by the GitHub job, don't edit by hand) |

Tickers: SPY, QQQ, IWM, AAPL, MSFT, NVDA, AMZN, TSLA.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Put your Massive key in `.env` as `MASSIVE_API_KEY=...` (never committed).
