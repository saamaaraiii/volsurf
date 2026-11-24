import pytest

from volsurf import black_scholes as bs
from volsurf import monte_carlo as mc

ARGS = dict(S=100.0, K=105.0, T=0.5, r=0.04, sigma=0.25, q=0.01)


@pytest.mark.parametrize("is_call", [True, False])
def test_european_within_confidence_interval(is_call):
    res = mc.european(**ARGS, is_call=is_call, n_paths=200_000, seed=11)
    exact = float(bs.price(**ARGS, is_call=is_call))
    assert abs(res.price - exact) < 4 * res.stderr


def test_variance_reduction_lowers_stderr():
    plain = mc.european(**ARGS, antithetic=False, control_variate=False, seed=1)
    reduced = mc.european(**ARGS, seed=1)
    assert reduced.stderr < plain.stderr


def test_geometric_asian_closed_form_matches_simulation():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.3
    exact = mc.geometric_asian_price(S, K, T, r, sigma, n_obs=12)
    import numpy as np

    rng = np.random.default_rng(3)
    dt = T / 12
    z = rng.standard_normal((400_000, 12))
    logp = np.log(S) + np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z, axis=1)
    payoff = np.exp(-r * T) * np.maximum(np.exp(logp.mean(axis=1)) - K, 0)
    assert payoff.mean() == pytest.approx(exact, abs=4 * payoff.std() / np.sqrt(len(payoff)))


def test_asian_control_variate_is_much_tighter():
    a = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.3, n_obs=50, n_paths=20_000, seed=5)
    plain = mc.arithmetic_asian(**a, control_variate=False)
    cv = mc.arithmetic_asian(**a)
    assert cv.stderr < plain.stderr / 10
    assert abs(cv.price - plain.price) < 4 * plain.stderr
    # arithmetic average >= geometric average, so the Asian call is worth more
    assert cv.price > mc.geometric_asian_price(100, 100, 1, 0.05, 0.3, n_obs=50)
