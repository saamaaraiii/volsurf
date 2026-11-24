"""SVI volatility smile fitting and static-arbitrage checks.

Gatheral's raw SVI parameterisation of total implied variance w = sigma^2 T as
a function of log-moneyness k = ln(K / F):

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + s^2))

It has five parameters, fits equity smiles very well and has known
conditions for being free of arbitrage, which is why it is an industry
standard for building implied volatility surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    s: float

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, float)
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.s**2))

    def implied_vol(self, k: np.ndarray, T: float) -> np.ndarray:
        return np.sqrt(np.maximum(self.total_variance(k), 0) / T)

    def durrleman_g(self, k: np.ndarray) -> np.ndarray:
        """Durrleman's condition: the smile has no butterfly arbitrage iff g(k) >= 0
        (equivalently, the implied risk-neutral density is non-negative)."""
        k = np.asarray(k, float)
        x = k - self.m
        root = np.sqrt(x**2 + self.s**2)
        w = self.total_variance(k)
        w1 = self.b * (self.rho + x / root)
        w2 = self.b * self.s**2 / root**3
        return (1 - k * w1 / (2 * w)) ** 2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2


@dataclass(frozen=True)
class SliceFit:
    T: float
    params: SVIParams
    rmse_iv_bp: float
    butterfly_free: bool


def fit_slice(
    k: np.ndarray,
    iv: np.ndarray,
    T: float,
    weights: np.ndarray | None = None,
    previous: SVIParams | None = None,
) -> SliceFit:
    """Fit raw SVI to one expiry's smile (log-moneyness vs implied vol).

    No-arbitrage conditions are added as penalty terms: minimum variance must
    be non-negative, Durrleman's g(k) must be non-negative (no butterfly
    arbitrage) and, if ``previous`` (the fitted slice for the previous expiry)
    is given, total variance must not fall below it (no calendar arbitrage).
    """
    k, iv = np.asarray(k, float), np.asarray(iv, float)
    mask = np.isfinite(k) & np.isfinite(iv)
    k, iv = k[mask], iv[mask]
    w_mkt = iv**2 * T
    wts = np.ones_like(k) if weights is None else np.asarray(weights, float)[mask]

    grid = np.linspace(k.min() - 0.5, k.max() + 0.5, 400)
    floor = previous.total_variance(grid) if previous is not None else None
    scale = float(np.sqrt(np.mean(w_mkt**2)))

    def resid(x):
        p = SVIParams(*x)
        fit = wts * (p.total_variance(k) - w_mkt)
        # soft constraints: non-negative minimum variance and no butterfly arbitrage
        min_var = x[0] + x[1] * x[4] * np.sqrt(1 - x[2] ** 2)
        g = p.durrleman_g(grid) if min_var > 0 else np.full_like(grid, -1.0)
        pen_var = 1e3 * scale * np.maximum(-min_var, 0) * np.sqrt(wts.sum())
        pen_bfly = 1e3 * scale * np.sqrt(np.maximum(-g, 0)) * np.sqrt(wts.mean())
        pen_cal = (
            1e3
            * scale
            * np.sqrt(np.maximum(floor - p.total_variance(grid), 0))
            * np.sqrt(wts.mean())
            if floor is not None
            else np.empty(0)
        )
        return np.concatenate([fit, [pen_var], pen_bfly, pen_cal])

    order = np.argsort(k)
    atm = float(np.interp(0.0, k[order], w_mkt[order]))
    lower = [-1.0, 1e-6, -0.999, -2.0, 1e-4]
    upper = [max(w_mkt.max(), 1e-3) * 2, 5.0, 0.999, 2.0, 5.0]
    best = None
    # SVI's objective is not convex, so try several starting points and keep the best
    for rho0 in (-0.7, -0.3, 0.0):
        for m0 in (-0.1, 0.0, 0.1):
            for s0 in (0.05, 0.2):
                x0 = np.clip([atm * 0.5, 0.1 * np.sqrt(T) + 0.01, rho0, m0, s0], lower, upper)
                sol = least_squares(resid, x0, bounds=(lower, upper))
                if best is None or sol.cost < best.cost:
                    best = sol
    params = SVIParams(*best.x)
    fit_iv = params.implied_vol(k, T)
    return SliceFit(
        T=T,
        params=params,
        rmse_iv_bp=float(np.sqrt(np.mean((fit_iv - iv) ** 2)) * 1e4),
        butterfly_free=bool(np.all(params.durrleman_g(grid) >= -1e-6)),
    )


def calendar_arbitrage_free(fits: list[SliceFit], k_grid: np.ndarray | None = None) -> bool:
    """Total variance must be non-decreasing in maturity at every log-moneyness."""
    k_grid = np.linspace(-0.5, 0.5, 201) if k_grid is None else k_grid
    fits = sorted(fits, key=lambda f: f.T)
    w = np.array([f.params.total_variance(k_grid) for f in fits])
    return bool(np.all(np.diff(w, axis=0) >= -1e-6))
