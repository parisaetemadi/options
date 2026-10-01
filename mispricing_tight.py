"""
The pricing model's mispricings, traded only where trading is cheap: options whose bid/ask spread
is at most 3% of the price (the earlier strategy_search.py paid 5-10% spreads on far-out options).
Delta-hedged daily, held h days. Rules picked on 2016-2021 at full bid/ask; the chosen rule is
checked ONCE on 2022-today at full bid/ask and at half the spread.

Run:  python mispricing_tight.py      (needs models/walkforward_predictions.parquet)
"""
import itertools

import numpy as np
import pandas as pd

import strategy_search as ss

MAX_SPREAD = 0.03
GRID = {"side": ["sell", "buy"], "gap": [0.01, 0.02, 0.03], "hold": [5, 10, 20]}
SELECT = range(2016, 2022)

p = pd.read_parquet(ss.PRED)
p = p[~p["atm_used"] & (np.abs(np.log(p["strike"] / p["spot"])) >= ss.MIN_ABS_LOG_M)
      & ((p["ask"] - p["bid"]) / p["mid"] <= MAX_SPREAD)].copy()
p["gap"] = p["iv"] - p["iv_pred"]
p["delta"] = ss.delta(p)
p["day_no"] = p["date"].map(ss.day_no)
p = p.dropna(subset=["day_no"]).astype({"day_no": int})
print(f"{len(p):,} tight-spread options (<= {MAX_SPREAD:.0%}), by ticker: {p['ticker'].value_counts().to_dict()}")

configs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
out = {(i, f): [] for i in range(len(configs)) for f in (1.0, 0.5)}
for year in sorted(p["date"].str[:4].unique()):
    py = p[p["date"].str[:4] == year]
    first, last = ss.day_no[py["date"].min()], min(ss.day_no[py["date"].max()] + max(GRID["hold"]), len(ss.all_days) - 1)
    panel = ss.panel_for(ss.all_days[first:last + 1])
    for i, c in enumerate(configs):
        sel = ss.pick(py, c["side"], c["gap"], 5)                  # at most 5 per ticker per day, biggest gaps first
        for f in (1.0, 0.5):
            t = ss.trade_pnl(sel, c["hold"], panel, c["side"], fill=f)
            out[(i, f)].append(t[["date", "ticker", "pnl", "mid", "dropped"]])
    print(f"{year} done", flush=True)
    del panel

rows = []
for i, c in enumerate(configs):
    t = pd.concat(out[(i, 1.0)], ignore_index=True)
    s = ss.stats(t[t["date"].str[:4].astype(int).isin(SELECT)], c["hold"])
    rows.append({**c, **{f"sel {k}": v for k, v in s.items()}})
res = pd.DataFrame(rows).sort_values("sel t (overlap-corrected)", ascending=False)
print("\nRules on 2016-2021 (full bid/ask), $ per contract, delta-hedged:")
print(res.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
i = res.index[0]
print(f"Chosen: {configs[i]}")
for f in (1.0, 0.5):
    t = pd.concat(out[(i, f)], ignore_index=True)
    conf = t[t["date"] >= "2022-01-01"]
    s = ss.stats(conf, configs[i]["hold"])
    print(f"  CONFIRMATION 2022-today, fill {f}: {s['trades']:,} trades, ${s.get('$/trade', np.nan):,.2f}/trade "
          f"({conf['pnl'].sum() / (conf['mid'].sum() * 100) * 100:+.1f}% of premium), "
          f"{s.get('% days +', np.nan):.0%} of days +, t {s.get('t (overlap-corrected)', np.nan):.1f}")
    if f == 1.0:
        print("  by year:", t.groupby(t["date"].str[:4])["pnl"].mean().round(1).to_dict())
