"""Black-Scholes-Merton pricing, Greeks and implied volatility.

All functions are vectorised with NumPy broadcasting, so you can pass arrays of
strikes, maturities or volatilities. ``q`` is a continuous dividend yield.
``is_call`` may be a bool or a boolean array.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2 * np.pi)


def _pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _d1_d2(S, K, T, r, sigma, q):
    S, K, T, sigma = (np.asarray(a, dtype=float) for a in (S, K, T, sigma))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    sigma: ArrayLike,
    q: float = 0.0,
    is_call: ArrayLike = True,
) -> np.ndarray:
    """European option price."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    S, K, T = (np.asarray(a, dtype=float) for a in (S, K, T))
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    call = S * df_q * ndtr(d1) - K * df_r * ndtr(d2)
    put = K * df_r * ndtr(-d2) - S * df_q * ndtr(-d1)
    return np.where(is_call, call, put)


def greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    sigma: ArrayLike,
    q: float = 0.0,
    is_call: ArrayLike = True,
) -> dict[str, np.ndarray]:
    """Analytic Greeks.

    Vega and rho are per 1.00 change (multiply by 0.01 for per-1%), theta is per
    year (divide by 365 for per-day).
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    S, K, T, sigma = (np.asarray(a, dtype=float) for a in (S, K, T, sigma))
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    sqrt_t = np.sqrt(T)
    pdf_d1 = _pdf(d1)

    delta = np.where(is_call, df_q * ndtr(d1), df_q * (ndtr(d1) - 1))
    gamma = df_q * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * df_q * pdf_d1 * sqrt_t
    common = -S * df_q * pdf_d1 * sigma / (2 * sqrt_t)
    theta_call = common - r * K * df_r * ndtr(d2) + q * S * df_q * ndtr(d1)
    theta_put = common + r * K * df_r * ndtr(-d2) - q * S * df_q * ndtr(-d1)
    theta = np.where(is_call, theta_call, theta_put)
    rho = np.where(is_call, K * T * df_r * ndtr(d2), -K * T * df_r * ndtr(-d2))
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(
    price_: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    q: float = 0.0,
    is_call: ArrayLike = True,
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> np.ndarray:
    """Vectorised implied volatility using safeguarded Newton-Raphson.

    Each option keeps a bracket [lo, hi] that always contains the root. A
    Newton step is taken when it stays inside the bracket, otherwise the
    method falls back to bisection, so it converges for deep ITM/OTM options
    where plain Newton diverges. Prices outside the no-arbitrage bounds
    return NaN.
    """
    p, S, K, T = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (price_, S, K, T)))
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), p.shape)
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    lower = np.where(
        is_call, np.maximum(S * df_q - K * df_r, 0), np.maximum(K * df_r - S * df_q, 0)
    )
    upper = np.where(is_call, S * df_q, K * df_r)
    valid = (p > lower) & (p < upper) & (T > 0)

    a = np.full(p.shape, lo)
    b = np.full(p.shape, hi)
    x = np.full(p.shape, 0.2)
    for _ in range(max_iter):
        f = price(S, K, T, r, x, q, is_call) - p
        if np.all(np.abs(f[valid]) < tol):
            break
        # price is increasing in vol: shrink the bracket
        b = np.where(f > 0, x, b)
        a = np.where(f <= 0, x, a)
        v = greeks(S, K, T, r, x, q, is_call)["vega"]
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = x - f / v
        inside = (newton > a) & (newton < b) & np.isfinite(newton)
        x = np.where(inside, newton, 0.5 * (a + b))
    return np.where(valid, x, np.nan)
