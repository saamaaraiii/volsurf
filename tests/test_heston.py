import numpy as np
import pytest
from scipy.integrate import quad

from volsurf import black_scholes as bs
from volsurf import heston
from volsurf.data import DEFAULT_TRUE_PARAMS as P
from volsurf.data import synthetic_chain
from volsurf.heston import HestonParams

S, r, q = 100.0, 0.04, 0.01


def reference_price(K, T, p):
    """Gil-Pelaez inversion with SciPy's adaptive quadrature: slow but independent."""
    fwd = S * np.exp((r - q) * T)
    k = np.log(K / fwd)

    def f1(u):
        return np.real(np.exp(-1j * u * k) * heston.char_func(u - 1j, T, p, 0, 0, 1) / (1j * u))

    def f2(u):
        return np.real(np.exp(-1j * u * k) * heston.char_func(u, T, p, 0, 0, 1) / (1j * u))

    p1 = 0.5 + quad(f1, 1e-10, np.inf, limit=500)[0] / np.pi
    p2 = 0.5 + quad(f2, 1e-10, np.inf, limit=500)[0] / np.pi
    return S * np.exp(-q * T) * p1 - K * np.exp(-r * T) * p2


@pytest.mark.parametrize("T", [0.02, 0.1, 0.5, 2.0, 5.0])
def test_fourier_pricer_matches_adaptive_quadrature(T):
    K = np.array([70.0, 90.0, 100.0, 110.0, 140.0])
    fast = heston.price(S, K, T, r, P, q)
    ref = np.array([reference_price(k, T, P) for k in K])
    np.testing.assert_allclose(fast, np.maximum(ref, 0), atol=1e-8)


def test_reduces_to_black_scholes_with_tiny_vol_of_vol():
    p = HestonParams(kappa=1.0, theta=0.04, sigma=1e-4, rho=0.0, v0=0.04)
    K = np.array([80.0, 100.0, 125.0])
    np.testing.assert_allclose(
        heston.price(S, K, 1.0, r, p, q), bs.price(S, K, 1.0, r, 0.2, q), atol=1e-4
    )


def test_put_call_parity():
    K, T = np.array([90.0, 100.0, 110.0]), 0.75
    c = heston.price(S, K, T, r, P, q, True)
    p = heston.price(S, K, T, r, P, q, False)
    np.testing.assert_allclose(c - p, S * np.exp(-q * T) - K * np.exp(-r * T), atol=1e-10)


def test_monte_carlo_agrees_with_fourier():
    mc = heston.monte_carlo(S, 100.0, 0.5, r, P, q, n_paths=200_000, n_steps=250, seed=2)
    exact = float(heston.price(S, 100.0, 0.5, r, P, q)[0])
    assert abs(mc.price - exact) < 4 * mc.stderr + 0.02  # small Euler discretisation bias


def test_negative_rho_gives_downward_skew():
    iv = heston.implied_vols(S, np.array([80.0, 100.0, 120.0]), 0.5, r, P, q)
    assert iv[0] > iv[1] > iv[2]


def test_calibration_recovers_true_parameters():
    chain = synthetic_chain(S=S, r=r, q=q, noise_bp=10, seed=4)
    ivs = bs.implied_vol(chain["mid"], S, chain["strike"], chain["T"], r, q, chain["is_call"])
    res = heston.calibrate(S, chain["strike"], chain["T"], ivs, r, q)
    assert res.success
    assert res.rmse_iv_bp < 20
    for name, rel in [
        ("kappa", 0.25),
        ("theta", 0.10),
        ("sigma", 0.10),
        ("rho", 0.05),
        ("v0", 0.05),
    ]:
        assert getattr(res.params, name) == pytest.approx(getattr(P, name), rel=rel), name
