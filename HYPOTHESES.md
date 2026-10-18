# Pre-registered hypotheses (written 2026-10-18, before any of these tests were run)

Five ideas, each from published research and none tested in this repo before. Each is tested
**once**, with the exact rules below, by `hypotheses_test.py`. No parameter is changed after seeing
results. A failed idea is dropped. Any variant tried later counts as a new idea and is checked only on
future (paper) data.

Earlier tests, so these don't repeat them: earnings/Fed/CPI/jobs trades held *through* the news
(event_trades.py), volatility risk premium sell/buy (vrp.py, vol_longshort.py), selling near-the-money
options to expiry (option_strategies.py), intraday move rules, time-of-day straddles, the 11:15 iron fly,
and the calm-day filter (intraday_test.py, iron_fly_test.py, calm_filter_test.py). All failed.

## Rules for all five

- Costs: buy at the ask, sell at the bid, IBKR $0.70 per contract to open and $0.70 to close.
  Half-spread fills are shown for information only and never decide anything.
- One contract per leg. No stock hedging (options only). Every position has a capped loss: bought
  options, or sold options covered by bought ones of the same type.
- In-the-money options are missing from the daily data (it keeps out-of-the-money strikes only).
  Their quotes are rebuilt from the same-strike put or call with put-call parity, as in
  strategy_search.panel_for. A quote still missing at the exit is valued with Black-Scholes at the
  entry implied volatility, minus a full extra spread. Trades over a stock split are skipped.
- "Near-the-money pair" = the out-of-the-money put with the highest strike at or below the stock and
  the out-of-the-money call with the lowest strike at or above it, same expiry, both with a bid > 0.
- Statistic: the t-stat of P&L summed by entry date (Newey-West when holding periods overlap).
- **Pass** = t >= 2.6 at full costs (5 tests: Bonferroni for 5% two-sided), **and** average P&L > 0 in
  both halves of the entry dates (split at the median date). Anything else fails.

## H1. Pre-earnings volatility run-up (Gao, Xing & Zhang 2018)
Implied volatility rises in the days before earnings, so options bought a week before and sold just
*before* the news gain from it, without taking the earnings move.
- Universe: the 34 stocks in data/events/earnings.csv, rows with type == "earnings", 2013-2026.
- Exit: the close of the trading day before reaction_day (the last close before the news).
  Entry: the close 5 trading days before the exit.
- Expiry: the nearest one on or after reaction_day (so it still contains the event), >= 7 days at entry.
- BUY the near-the-money pair at entry, sell it at the exit.

## H2. Weekend decay (Jones & Shemesh 2018)
Options lose value over weekends that the Friday price doesn't fully reflect, so selling on Friday and
buying back on Monday earns that decay.
- Tickers: SPY, QQQ, IWM, NVDA, TSLA. Daily data 2013-2026.
- Entry: the close of the last trading day of each week. Exit: the next trading day's close.
- Expiry: the nearest with >= 7 days to go at entry.
- SELL the near-the-money pair, BUY wings: the put with the highest strike <= 0.95 x stock and the call
  with the lowest strike >= 1.05 x stock (iron condor; loss capped).

## H3. Overnight decay (Muravyev & Ni 2020)
Option returns are negative overnight and positive during the day: options are too expensive at the
close relative to the next morning.
- Intraday 15-min data: SPY, QQQ (May 2023-Sep 2026) and NVDA, TSLA (Sep 2025-Sep 2026), pooled.
- Entry: the 15:45 snapshot. Exit: the 9:45 snapshot of the next trading day.
- Expiry: the nearest one after today.
- SELL the at-the-money call and put (strike nearest the put-call parity stock price, both with a bid),
  BUY the call at the first strike >= 1.01 x stock and the put at the last strike <= 0.99 x stock
  (iron fly, as in iron_fly_test.py). A missing exit quote is valued at expiry value from the 9:45
  parity price; a bought option with no bid sells for $0.

## H4. Index put spreads (Bondarenko 2014: index puts are overpriced)
Out-of-the-money index puts cost more than they pay out on average; a put spread keeps the loss capped.
- Tickers: SPY, QQQ, IWM. Daily data 2013-2026.
- Entry: the first trading day after each standard monthly expiry (3rd Friday). Expiry: the next
  standard monthly expiry. One trade per ticker per month, no overlap.
- SELL the put with the strike nearest 0.95 x stock, BUY the put with the strike nearest 0.90 x stock
  (strictly lower). Hold to expiry, settled at intrinsic value from the close on the last trading day
  on or before expiry. Commission to open; if the sold put ends in the money, also $0.70 x 2 to close.
- Statistic: t-stat of P&L summed by entry month.

## H5. Pre-Fed volatility run-up (the H1 idea on scheduled Fed decisions)
- Tickers: SPY, QQQ, IWM. Fed decision days = data/events/macro.csv type == "fomc", up to 2026-09.
- Exit: the close of the trading day before the decision. Entry: the close 5 trading days before that.
- Expiry: the nearest one on or after the decision day, >= 7 days at entry.
- BUY the near-the-money pair at entry, sell it at the exit.

## Amendment 1 (2026-10-18, before any valid result was seen)
The daily data keeps only options with >= 7 days to expiry, so an option expiring right after the news
(H1, H5) or within the week (H2) is already missing on the exit day. The first run valued 100% of those
exits by formula, which is meaningless; that output was discarded. Fix, applied the same to every trade:
**the expiry is the nearest one with at least 7 days left on the exit day** (H1/H5: and on or after the
news day, which that already implies). Nothing else changes. H3 and H4 don't need exit quotes from the
daily data and are unaffected (H4 had already run: t 2.22, FAIL).
