"""Cox-Ross-Rubinstein binomial tree for European and American options."""

from __future__ import annotations

import numpy as np


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    is_call: bool = True,
    american: bool = False,
    steps: int = 500,
) -> float:
    """Price an option on a recombining CRR tree.

    Backward induction is vectorised across each time slice, so a 2,000-step
    tree takes milliseconds. With ``american=True`` the value at every node is
    the larger of continuing and exercising now.
    """
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0 < p < 1:
        raise ValueError("arbitrage in tree: increase steps or check inputs")
    disc = np.exp(-r * dt)

    j = np.arange(steps + 1)
    spot = S * u ** (steps - 2 * j)  # terminal prices, highest first
    payoff = np.maximum(spot - K, 0) if is_call else np.maximum(K - spot, 0)
    values = payoff
    for _ in range(steps):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if american:
            spot = spot[:-1] / u
            exercise = np.maximum(spot - K, 0) if is_call else np.maximum(K - spot, 0)
            values = np.maximum(values, exercise)
    return float(values[0])
