import numpy as np
import pytest

from volsurf import binomial_price
from volsurf import black_scholes as bs

S, K, T, r, sigma, q = 100.0, 105.0, 0.75, 0.04, 0.25, 0.015


def test_known_value():
    # Hull, Options Futures & Other Derivatives, Example 15.6: S=42, K=40, r=10%, sigma=20%, T=0.5
    assert bs.price(42, 40, 0.5, 0.10, 0.2) == pytest.approx(4.76, abs=5e-3)
    assert bs.price(42, 40, 0.5, 0.10, 0.2, is_call=False) == pytest.approx(0.81, abs=5e-3)


def test_put_call_parity():
    c = bs.price(S, K, T, r, sigma, q, True)
    p = bs.price(S, K, T, r, sigma, q, False)
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T))


@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_match_finite_differences(is_call):
    g = bs.greeks(S, K, T, r, sigma, q, is_call)
    h = 1e-4

    def f(**kw):
        a = dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q, is_call=is_call)
        a.update(kw)
        return bs.price(**a)

    assert g["delta"] == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-6)
    assert g["gamma"] == pytest.approx((f(S=S + h) - 2 * f() + f(S=S - h)) / h**2, rel=1e-3)
    assert g["vega"] == pytest.approx((f(sigma=sigma + h) - f(sigma=sigma - h)) / (2 * h), rel=1e-6)
    assert g["rho"] == pytest.approx((f(r=r + h) - f(r=r - h)) / (2 * h), rel=1e-6)
    assert g["theta"] == pytest.approx(-(f(T=T + h) - f(T=T - h)) / (2 * h), rel=1e-6)


def test_implied_vol_round_trip_across_strikes():
    strikes = np.linspace(40, 250, 50)
    vols = np.linspace(0.08, 1.2, 50)
    is_call = strikes > S
    prices = bs.price(S, strikes, T, r, vols, q, is_call)
    iv = bs.implied_vol(prices, S, strikes, T, r, q, is_call)
    # a price of ~1e-30 carries no information about volatility; skip those
    informative = bs.greeks(S, strikes, T, r, vols, q, is_call)["vega"] > 1e-6
    assert informative.sum() > 40
    np.testing.assert_allclose(iv[informative], vols[informative], atol=1e-6)


def test_implied_vol_rejects_arbitrage_prices():
    assert np.isnan(bs.implied_vol(150.0, S, K, T, r, q, True))  # above spot
    assert np.isnan(bs.implied_vol(0.0, S, K, T, r, q, True))


def test_binomial_converges_to_black_scholes():
    exact = bs.price(S, K, T, r, sigma, q, False)
    errors = [abs(binomial_price(S, K, T, r, sigma, q, False, steps=n) - exact) for n in (50, 2000)]
    assert errors[1] < errors[0]
    assert errors[1] < 2e-3


def test_american_put_early_exercise_premium():
    euro = binomial_price(S, 120, 1.0, 0.08, 0.2, 0.0, is_call=False, steps=1000)
    amer = binomial_price(S, 120, 1.0, 0.08, 0.2, 0.0, is_call=False, american=True, steps=1000)
    assert amer > euro
    assert amer >= 120 - S - 1e-9  # never below intrinsic value


def test_american_call_no_dividend_equals_european():
    euro = binomial_price(S, K, T, r, sigma, 0.0, is_call=True, steps=800)
    amer = binomial_price(S, K, T, r, sigma, 0.0, is_call=True, american=True, steps=800)
    assert amer == pytest.approx(euro, abs=1e-10)
