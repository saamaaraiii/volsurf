"""Build an arbitrage-checked implied volatility surface from an option chain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import svi
from .data import add_implied_vols


@dataclass
class VolSurface:
    spot: float
    r: float
    q: float
    slices: list[svi.SliceFit]
    quotes: pd.DataFrame

    @property
    def calendar_arbitrage_free(self) -> bool:
        return svi.calendar_arbitrage_free(self.slices)

    @property
    def butterfly_arbitrage_free(self) -> bool:
        return all(s.butterfly_free for s in self.slices)

    def total_variance(self, k: np.ndarray, T: float) -> np.ndarray:
        """Interpolate linearly in total variance between expiries (this keeps
        the surface calendar-arbitrage free if the slices are)."""
        Ts = np.array([s.T for s in self.slices])
        if T <= Ts[0]:
            return self.slices[0].params.total_variance(k) * T / Ts[0]
        if T >= Ts[-1]:
            return self.slices[-1].params.total_variance(k) * T / Ts[-1]
        i = int(np.searchsorted(Ts, T))
        w0 = self.slices[i - 1].params.total_variance(k)
        w1 = self.slices[i].params.total_variance(k)
        a = (T - Ts[i - 1]) / (Ts[i] - Ts[i - 1])
        return (1 - a) * w0 + a * w1

    def implied_vol(self, K: np.ndarray, T: float) -> np.ndarray:
        fwd = self.spot * np.exp((self.r - self.q) * T)
        k = np.log(np.asarray(K, float) / fwd)
        return np.sqrt(np.maximum(self.total_variance(k, T), 0) / T)

    def grid(self, k: np.ndarray, Ts: np.ndarray) -> np.ndarray:
        """Implied vols on a (maturity x log-moneyness) grid, for plotting."""
        return np.array([np.sqrt(np.maximum(self.total_variance(k, T), 0) / T) for T in Ts])

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                dict(
                    T=s.T,
                    rmse_iv_bp=s.rmse_iv_bp,
                    butterfly_free=s.butterfly_free,
                    **{f"svi_{k}": v for k, v in vars(s.params).items()},
                )
                for s in self.slices
            ]
        )


def build_surface(chain: pd.DataFrame, min_points: int = 5) -> VolSurface:
    quotes = add_implied_vols(chain)
    slices = []
    for T, grp in quotes.groupby("T"):
        if len(grp) < min_points:
            continue
        # weight by the inverse of the bid-ask width in vol: trust tight quotes more
        width = (grp["iv_ask"] - grp["iv_bid"]).clip(lower=1e-4).fillna(grp["iv"] * 0.1)
        prev = slices[-1].params if slices else None
        slices.append(
            svi.fit_slice(grp["log_moneyness"], grp["iv"], float(T), 1 / width.to_numpy(), prev)
        )
    if not slices:
        raise ValueError("not enough quotes to fit any expiry")
    a = chain.attrs
    return VolSurface(a["spot"], a["r"], a["q"], slices, quotes)
