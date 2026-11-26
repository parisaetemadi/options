# Options

> **Purpose.** Research toward pricing options well enough to trade them. Black-Scholes, scheduled quote downloads, neural-net pricing experiments and backtests.

Goal: build a model that prices options well enough to trade them.

## Files

| File | What it does |
|---|---|
| `pricing.py` | Black-Scholes prices and implied volatility, shared by every script |
| `download_massive.py` | Downloads ~2 years of daily option prices from Massive (free plan, 5 calls/min). Runs by itself on GitHub every 6 hours |
| `snapshot_yahoo.py` | Saves real option quotes (bid/ask/volume/open interest) from Yahoo every weekday afternoon. Runs by itself on GitHub |
| `live_iv_model.py` | Pulls today's option quotes from Yahoo and trains one neural net on all tickers (run during market hours) |
| `learn_bs_neural_net.py` | Practice: a neural net learns the Black-Scholes formula on made-up data |
| `data/massive/` | Daily option prices from Massive, past ~2 years (written by the GitHub job) |
| `data/yahoo/` | One file per day of Yahoo quotes, from now on (written by the GitHub job) |

Tickers: SPY, QQQ, IWM, AAPL, MSFT, NVDA, AMZN, TSLA.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Put your Massive key in `.env` as `MASSIVE_API_KEY=...` (never committed).

## Agent rules

- Commit straight to `main` and push — no branches, no PRs, no merges, never watch a PR.
- One commit per session. Test it yourself before committing; don't report untested work.
- Don't ask permission for deploys, settings, or deleting what's been replaced.
- Save tokens: bullets over prose, read only the files named, targeted edits, batched tool calls.
- Comments: one bullet list per file or section, never a comment per chunk of code.
- Full detail: [CLAUDE.md](CLAUDE.md)
