"""Monte Carlo pricing with variance reduction.

Every pricer returns an ``MCResult`` with the estimate *and* its standard
error, so you can see how much each variance-reduction technique helps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr


@dataclass(frozen=True)
class MCResult:
    price: float
    stderr: float
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        return self.price - 1.96 * self.stderr, self.price + 1.96 * self.stderr


def _control_variate(y: np.ndarray, x: np.ndarray, x_mean: float) -> np.ndarray:
    """Adjust samples y using a control x whose true mean is known."""
    cov = np.cov(y, x)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
    return y - beta * (x - x_mean)


def _summarise(samples: np.ndarray, n_paths: int) -> MCResult:
    return MCResult(
        float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(len(samples))), n_paths
    )


def european(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    is_call: bool = True,
    n_paths: int = 100_000,
    antithetic: bool = True,
    control_variate: bool = True,
    seed: int | None = None,
) -> MCResult:
    """European option under GBM.

    * Antithetic variates pair each normal draw z with -z.
    * The control variate is the discounted terminal stock price, whose
      expectation S*exp(-qT) is known exactly.
    """
    rng = np.random.default_rng(seed)
    half = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal(half)
    if antithetic:
        z = np.concatenate([z, -z])
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    disc = np.exp(-r * T)
    payoff = disc * (np.maximum(ST - K, 0) if is_call else np.maximum(K - ST, 0))
    if control_variate:
        payoff = _control_variate(payoff, disc * ST, S * np.exp(-q * T))
    if antithetic:  # pairs are correlated: average them before taking the std error
        payoff = 0.5 * (payoff[:half] + payoff[half:])
    return _summarise(payoff, n_paths)


def geometric_asian_price(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, n_obs: int = 252
) -> float:
    """Closed-form price of a discretely monitored geometric-average Asian call."""
    n = n_obs
    mu = np.log(S) + (r - q - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    var = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    d2 = (mu - np.log(K)) / np.sqrt(var)
    d1 = d2 + np.sqrt(var)
    return float(np.exp(-r * T) * (np.exp(mu + 0.5 * var) * ndtr(d1) - K * ndtr(d2)))


def arithmetic_asian(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    n_obs: int = 252,
    n_paths: int = 50_000,
    control_variate: bool = True,
    seed: int | None = None,
) -> MCResult:
    """Arithmetic-average Asian call (no closed form exists).

    The geometric-average Asian, which *does* have a closed form and is very
    highly correlated with the arithmetic one, is used as the control. This
    typically cuts the standard error by more than an order of magnitude.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_obs
    z = rng.standard_normal((n_paths, n_obs))
    log_paths = np.log(S) + np.cumsum(
        (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z, axis=1
    )
    disc = np.exp(-r * T)
    arith = disc * np.maximum(np.exp(log_paths).mean(axis=1) - K, 0)
    if control_variate:
        geo = disc * np.maximum(np.exp(log_paths.mean(axis=1)) - K, 0)
        arith = _control_variate(arith, geo, geometric_asian_price(S, K, T, r, sigma, q, n_obs))
    return _summarise(arith, n_paths)
