"""Option-chain data: a synthetic generator with known ground truth, and a
loader for live chains from Yahoo Finance.

Both return a DataFrame with one row per option and the columns
``expiry, T, strike, is_call, bid, ask, mid``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import black_scholes as bs
from . import heston
from .heston import HestonParams

# Roughly what equity-index options look like: strong negative spot/vol correlation.
DEFAULT_TRUE_PARAMS = HestonParams(kappa=1.8, theta=0.045, sigma=0.55, rho=-0.72, v0=0.03)


def synthetic_chain(
    S: float = 100.0,
    r: float = 0.04,
    q: float = 0.013,
    params: HestonParams = DEFAULT_TRUE_PARAMS,
    expiries: tuple[float, ...] = (1 / 12, 2 / 12, 3 / 12, 6 / 12, 1.0, 2.0),
    moneyness: np.ndarray | None = None,
    spread_bp: float = 40.0,
    noise_bp: float = 15.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate an OTM option chain from a Heston model, then add realistic
    noise to the mid implied vols and a bid-ask spread.

    Because the true parameters are known, this is the ideal testbed for a
    calibration routine: does it recover what generated the data?
    """
    rng = np.random.default_rng(seed)
    moneyness = np.linspace(0.7, 1.3, 25) if moneyness is None else moneyness
    rows = []
    for T in expiries:
        width = np.sqrt(T)  # wider strike range for longer expiries
        fwd = S * np.exp((r - q) * T)
        K = fwd * np.exp(np.log(moneyness) * width)
        iv = heston.implied_vols(S, K, np.full_like(K, T), r, params, q)
        iv = iv + rng.normal(0, noise_bp / 1e4, size=iv.shape)
        is_call = K >= fwd
        for k, v, c in zip(K, iv, is_call, strict=True):
            if not np.isfinite(v):
                continue
            bid = bs.price(S, k, T, r, v - spread_bp / 2e4, q, c)
            ask = bs.price(S, k, T, r, v + spread_bp / 2e4, q, c)
            rows.append(
                dict(
                    expiry=f"{T:.3f}y",
                    T=T,
                    strike=float(k),
                    is_call=bool(c),
                    bid=float(bid),
                    ask=float(ask),
                    mid=float(0.5 * (bid + ask)),
                )
            )
    df = pd.DataFrame(rows)
    df.attrs.update(spot=S, r=r, q=q, source="synthetic-heston")
    return df


def yahoo_chain(
    ticker: str = "SPY",
    r: float = 0.04,
    q: float = 0.013,
    max_expiries: int = 8,
    min_days: int = 7,
    max_rel_spread: float = 0.25,
) -> pd.DataFrame:
    """Download a live option chain from Yahoo Finance (needs ``pip install yfinance``).

    Keeps out-of-the-money options with a two-sided quote and a reasonable
    bid-ask spread. Outside market hours Yahoo often reports zero bids/asks,
    in which case the result will be empty.
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="5d")["Close"].iloc[-1])
    now = datetime.now(timezone.utc)
    rows = []
    for exp in tk.options:
        T = (datetime.strptime(exp, "%Y-%m-%d").replace(tzinfo=timezone.utc) - now).days / 365
        if T * 365 < min_days:
            continue
        chain = tk.option_chain(exp)
        fwd = spot * np.exp((r - q) * T)
        for frame, is_call in ((chain.calls, True), (chain.puts, False)):
            f = frame[(frame["bid"] > 0) & (frame["ask"] > frame["bid"])]
            f = f[(f["strike"] >= fwd) if is_call else (f["strike"] < fwd)]
            mid = 0.5 * (f["bid"] + f["ask"])
            f = f[(f["ask"] - f["bid"]) / mid <= max_rel_spread]
            for _, row in f.iterrows():
                rows.append(
                    dict(
                        expiry=exp,
                        T=T,
                        strike=float(row.strike),
                        is_call=is_call,
                        bid=float(row.bid),
                        ask=float(row.ask),
                        mid=float(0.5 * (row.bid + row.ask)),
                    )
                )
        if len({r_["expiry"] for r_ in rows}) >= max_expiries:
            break
    df = pd.DataFrame(rows, columns=["expiry", "T", "strike", "is_call", "bid", "ask", "mid"])
    df.attrs.update(spot=spot, r=r, q=q, source=f"yahoo:{ticker}")
    return df


def add_implied_vols(df: pd.DataFrame) -> pd.DataFrame:
    """Add mid, bid and ask implied vols plus log-moneyness columns."""
    S, r, q = df.attrs["spot"], df.attrs["r"], df.attrs["q"]
    out = df.copy()
    args = (S, out["strike"].to_numpy(), out["T"].to_numpy(), r, q, out["is_call"].to_numpy())
    out["iv"] = bs.implied_vol(out["mid"].to_numpy(), *args)
    out["iv_bid"] = bs.implied_vol(out["bid"].to_numpy(), *args)
    out["iv_ask"] = bs.implied_vol(out["ask"].to_numpy(), *args)
    fwd = S * np.exp((r - q) * out["T"])
    out["log_moneyness"] = np.log(out["strike"] / fwd)
    out.attrs = dict(df.attrs)
    return out.dropna(subset=["iv"]).reset_index(drop=True)
