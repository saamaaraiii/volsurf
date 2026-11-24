import numpy as np
import pytest

from volsurf import build_surface, svi
from volsurf.data import synthetic_chain


def test_svi_recovers_known_smile():
    true = svi.SVIParams(a=0.02, b=0.12, rho=-0.6, m=0.05, s=0.15)
    k = np.linspace(-0.6, 0.4, 30)
    fit = svi.fit_slice(k, true.implied_vol(k, 0.5), 0.5)
    assert fit.rmse_iv_bp < 1
    assert fit.butterfly_free


def test_durrleman_flags_arbitrage():
    # a very steep smile with tiny curvature implies a negative density somewhere
    bad = svi.SVIParams(a=-0.05, b=0.9, rho=-0.95, m=0.0, s=0.02)
    assert np.any(bad.durrleman_g(np.linspace(-1, 1, 401)) < 0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_surface_is_arbitrage_free_and_accurate(seed):
    surf = build_surface(synthetic_chain(seed=seed))
    assert surf.butterfly_arbitrage_free
    assert surf.calendar_arbitrage_free
    assert surf.summary()["rmse_iv_bp"].max() < 30  # data has ~15bp noise


def test_surface_interpolation_between_expiries():
    surf = build_surface(synthetic_chain(seed=0))
    Ts = [s.T for s in surf.slices]
    k = np.array([0.0])
    w_mid = surf.total_variance(k, 0.5 * (Ts[2] + Ts[3]))
    assert surf.total_variance(k, Ts[2]) <= w_mid <= surf.total_variance(k, Ts[3])
    assert surf.implied_vol(np.array([100.0]), 0.75).shape == (1,)
