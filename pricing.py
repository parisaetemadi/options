"""
Black-Scholes pricing and implied volatility -- shared by every script.
q is the stock's dividend yield (0 if it pays none).
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bs_call(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_price(kind, S, K, T, r, sigma, q=0.0):
    """kind is "call" or "put" (or an array of them)."""
    return np.where(np.asarray(kind) == "call", bs_call(S, K, T, r, sigma, q), bs_put(S, K, T, r, sigma, q))


def implied_vol(price, S, K, T, r, kind):
    """Find the volatility that makes Black-Scholes match the market price."""
    f = bs_call if kind == "call" else bs_put
    try:
        return brentq(lambda s: f(S, K, T, r, s) - price, 0.001, 5.0)
    except ValueError:
        return np.nan   # no volatility fits (bad or stale price)


def implied_vol_many(price, S, K, T, r, kind, q=0.0, iters=60):
    """Implied vol for many options at once (bisection, vectorized). NaN where no vol fits."""
    price, S, K, T = (np.asarray(x, dtype=float) for x in (price, S, K, T))
    is_call = np.asarray(kind) == "call"
    lo, hi = np.full(price.shape, 1e-3), np.full(price.shape, 5.0)
    f = lambda s: np.where(is_call, bs_call(S, K, T, r, s, q), bs_put(S, K, T, r, s, q)) - price
    bad = (f(lo) > 0) | (f(hi) < 0)          # price outside what any volatility can produce
    for _ in range(iters):
        mid = (lo + hi) / 2
        above = f(mid) > 0
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return np.where(bad, np.nan, (lo + hi) / 2)
