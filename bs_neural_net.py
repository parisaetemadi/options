"""
Step 1: Teach a neural net to price options (Black-Scholes).
Install once:  pip install numpy scipy scikit-learn
Run:           python bs_neural_net.py
"""
import numpy as np
from scipy.stats import norm
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


# ---------- 1. The Black-Scholes formula (call option) ----------
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# Quick sanity check: stock $100, strike $100, 1 year, 5% rate, 20% vol
print("Example price:", round(black_scholes_call(100, 100, 1, 0.05, 0.20), 2))  # ~10.45


# ---------- 2. Make fake training data ----------
# Trick: set strike K = 1, so S is "stock price relative to strike".
# This keeps all numbers small, which neural nets like.
rng = np.random.default_rng(0)
n = 50_000
S     = rng.uniform(0.7, 1.3, n)    # stock is 70%..130% of strike
T     = rng.uniform(0.05, 2.0, n)   # 2 weeks .. 2 years
r     = rng.uniform(0.0, 0.05, n)   # 0% .. 5% interest
sigma = rng.uniform(0.1, 0.6, n)    # 10% .. 60% volatility

X = np.column_stack([S, T, r, sigma])      # inputs
y = black_scholes_call(S, 1.0, T, r, sigma)  # correct answers

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ---------- 3. Train a small neural net ----------
model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=200,
                     early_stopping=True, random_state=0)
model.fit(X_train, y_train)


# ---------- 4. Check it on data it has never seen ----------
pred = model.predict(X_test)
err = np.abs(pred - y_test)
print(f"Average error: {err.mean():.4f}  (per $1 of strike)")
print(f"Worst error:   {err.max():.4f}")

# Same example as above, now priced by the neural net (scale back by K=100)
nn_price = model.predict([[100 / 100, 1, 0.05, 0.20]])[0] * 100
print("Neural net price for the example:", round(nn_price, 2))
