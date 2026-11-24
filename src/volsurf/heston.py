"""Heston (1993) stochastic volatility model.

    dS = (r - q) S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + sigma sqrt(v) dW2,     d<W1, W2> = rho dt

European prices come from the characteristic function by Fourier inversion
(Gil-Pelaez). The "little Heston trap" form of Albrecher et al. (2007) keeps
the complex logarithm on its principal branch, which avoids the
discontinuities of Heston's original formula at long maturities.

The integral uses Gauss-Legendre quadrature with a cut-off chosen per option
from the decay rate of the characteristic function, so a whole option chain
is priced in one vectorised call. That speed is what makes
calibration practical.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares

from . import black_scholes as bs
from .monte_carlo import MCResult


@dataclass(frozen=True)
class HestonParams:
    kappa: float  # speed of mean reversion of variance
    theta: float  # long-run variance
    sigma: float  # volatility of variance ("vol of vol")
    rho: float  # correlation between spot and variance shocks
    v0: float  # current variance

    @property
    def feller(self) -> float:
        """2*kappa*theta - sigma^2. Positive means variance never hits zero."""
        return 2 * self.kappa * self.theta - self.sigma**2

    def as_array(self) -> np.ndarray:
        return np.array([self.kappa, self.theta, self.sigma, self.rho, self.v0])

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


# Gauss-Legendre nodes on [0, 1], rescaled per option to [0, u_max].
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(512)
_X = 0.5 * (_NODES + 1)
_WX = 0.5 * _WEIGHTS


def _u_max(T: np.ndarray, p: HestonParams) -> np.ndarray:
    """Integration cut-off. For large u, |phi(u)| decays roughly like
    exp(-u * (v0 + kappa*theta*T) * sqrt(1 - rho^2) / sigma), so pick u_max
    where that is ~1e-15. Short maturities and low variance need a longer range."""
    rate = (p.v0 + p.kappa * p.theta * T) * np.sqrt(1 - p.rho**2) / p.sigma
    return np.clip(35.0 / rate, 100.0, 5000.0)


def char_func(u: np.ndarray, T: np.ndarray, p: HestonParams, r: float, q: float, S: float):
    """Characteristic function of ln(S_T). ``u`` and ``T`` broadcast together."""
    kappa, theta, sigma, rho, v0 = p.kappa, p.theta, p.sigma, p.rho, p.v0
    iu = 1j * u
    b = kappa - rho * sigma * iu
    d = np.sqrt(b**2 + sigma**2 * (iu + u**2))
    g = (b - d) / (b + d)
    edt = np.exp(-d * T)
    C = (r - q) * iu * T + kappa * theta / sigma**2 * (
        (b - d) * T - 2 * np.log((1 - g * edt) / (1 - g))
    )
    D = (b - d) / sigma**2 * (1 - edt) / (1 - g * edt)
    return np.exp(C + D * v0 + iu * np.log(S))


def price(
    S: float,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    params: HestonParams,
    q: float = 0.0,
    is_call: np.ndarray | bool = True,
) -> np.ndarray:
    """European option prices under Heston for arrays of strikes and maturities."""
    K, T = np.broadcast_arrays(
        np.atleast_1d(np.asarray(K, float)), np.atleast_1d(np.asarray(T, float))
    )
    umax = _u_max(T, params)[:, None]
    u = _X[None, :] * umax
    w = _WX[None, :] * umax
    t = T[:, None]
    fwd = S * np.exp((r - q) * T)
    # Work with log-moneyness ln(K/F) and the characteristic function of ln(S_T/F)
    # (spot 1, zero carry). Using ln K directly makes the integrand oscillate
    # ~ln(K)/(2*pi) times faster, which a fixed quadrature cannot resolve.
    k = np.log(K / fwd)[:, None]
    phi = char_func(u, t, params, 0.0, 0.0, 1.0)
    phi_shift = char_func(u - 1j, t, params, 0.0, 0.0, 1.0)
    e = np.exp(-1j * u * k)
    p1 = 0.5 + np.sum(np.real(e * phi_shift / (1j * u)) * w, axis=1) / np.pi
    p2 = 0.5 + np.sum(np.real(e * phi / (1j * u)) * w, axis=1) / np.pi
    call = S * np.exp(-q * T) * p1 - K * np.exp(-r * T) * p2
    # truncating the integral can leave tiny errors for very short, far-OTM
    # options; clamp to the no-arbitrage bounds
    call = np.clip(call, np.maximum(S * np.exp(-q * T) - K * np.exp(-r * T), 0), S * np.exp(-q * T))
    put = call - S * np.exp(-q * T) + K * np.exp(-r * T)  # put-call parity
    return np.where(is_call, call, put)


def implied_vols(S, K, T, r, params: HestonParams, q: float = 0.0) -> np.ndarray:
    """Black-Scholes implied vols of Heston prices, using OTM options for stability."""
    K, T = np.broadcast_arrays(
        np.atleast_1d(np.asarray(K, float)), np.atleast_1d(np.asarray(T, float))
    )
    otm_call = K >= S * np.exp((r - q) * T)
    prices = price(S, K, T, r, params, q, otm_call)
    return bs.implied_vol(prices, S, K, T, r, q, otm_call)


def monte_carlo(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    q: float = 0.0,
    is_call: bool = True,
    n_paths: int = 100_000,
    n_steps: int = 200,
    seed: int | None = None,
) -> MCResult:
    """Full-truncation Euler scheme (Lord et al., 2010). Used to cross-check the
    Fourier pricer, and a template for path-dependent options under Heston."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    p = params
    x = np.full(n_paths, np.log(S))
    v = np.full(n_paths, p.v0)
    sq = np.sqrt(1 - p.rho**2)
    for _ in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = p.rho * z1 + sq * rng.standard_normal(n_paths)
        vp = np.maximum(v, 0)
        x += (r - q - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
        v += p.kappa * (p.theta - vp) * dt + p.sigma * np.sqrt(vp * dt) * z2
    ST = np.exp(x)
    payoff = np.exp(-r * T) * (np.maximum(ST - K, 0) if is_call else np.maximum(K - ST, 0))
    return MCResult(float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(n_paths)), n_paths)


@dataclass(frozen=True)
class CalibrationResult:
    params: HestonParams
    rmse_iv_bp: float  # root-mean-square implied-vol error in basis points
    max_abs_iv_bp: float
    n_options: int
    n_evals: int
    success: bool


_LOWER = np.array([1e-3, 1e-4, 1e-2, -0.999, 1e-4])
_UPPER = np.array([15.0, 1.0, 3.0, 0.999, 1.0])


def calibrate(
    S: float,
    K: np.ndarray,
    T: np.ndarray,
    market_iv: np.ndarray,
    r: float,
    q: float = 0.0,
    x0: HestonParams | None = None,
) -> CalibrationResult:
    """Fit Heston parameters to a set of market implied volatilities.

    The objective is the vega-weighted price error, which to first order equals
    the implied-vol error but avoids inverting Black-Scholes on every
    iteration. Calibration uses OTM options only, which are the liquid ones.
    """
    K, T, market_iv = (np.asarray(a, float) for a in (K, T, market_iv))
    otm_call = K >= S * np.exp((r - q) * T)
    target = bs.price(S, K, T, r, market_iv, q, otm_call)
    vega = np.maximum(bs.greeks(S, K, T, r, market_iv, q, otm_call)["vega"], 1e-8)

    def residuals(x: np.ndarray) -> np.ndarray:
        model = price(S, K, T, r, HestonParams(*x), q, otm_call)
        return (model - target) / vega

    if x0 is None:
        atm_var = float(np.median(market_iv) ** 2)
        x0 = HestonParams(kappa=2.0, theta=atm_var, sigma=0.5, rho=-0.5, v0=atm_var)
    start = np.clip(x0.as_array(), _LOWER + 1e-6, _UPPER - 1e-6)
    sol = least_squares(residuals, start, bounds=(_LOWER, _UPPER), x_scale="jac", method="trf")
    fitted = HestonParams(*sol.x)
    iv_err = implied_vols(S, K, T, r, fitted, q) - market_iv
    return CalibrationResult(
        params=fitted,
        rmse_iv_bp=float(np.sqrt(np.nanmean(iv_err**2)) * 1e4),
        max_abs_iv_bp=float(np.nanmax(np.abs(iv_err)) * 1e4),
        n_options=len(K),
        n_evals=int(sol.nfev),
        success=bool(sol.success),
    )
