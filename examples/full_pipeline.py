"""End-to-end demo: chain -> implied vols -> SVI surface -> Heston calibration,
plus pricer convergence studies. Saves the README charts to ``docs/``.

    python examples/full_pipeline.py                 # synthetic chain, known truth
    python examples/full_pipeline.py --ticker SPY    # live Yahoo chain (market hours)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from volsurf import binomial_price, build_surface, heston
from volsurf import black_scholes as bs
from volsurf import monte_carlo as mc
from volsurf.data import DEFAULT_TRUE_PARAMS, synthetic_chain, yahoo_chain

OUT = Path(__file__).resolve().parents[1] / "docs"
OUT.mkdir(exist_ok=True)


def surface_and_calibration(chain) -> None:
    S, r, q = chain.attrs["spot"], chain.attrs["r"], chain.attrs["q"]
    t0 = time.perf_counter()
    surf = build_surface(chain)
    t_svi = time.perf_counter() - t0
    quotes = surf.quotes
    t0 = time.perf_counter()
    cal = heston.calibrate(S, quotes["strike"], quotes["T"], quotes["iv"], r, q)
    t_cal = time.perf_counter() - t0

    print(
        f"Source: {chain.attrs['source']}  |  {len(quotes)} options, {len(surf.slices)} expiries\n"
    )
    print("SVI fit per expiry")
    print(surf.summary()[["T", "rmse_iv_bp", "butterfly_free"]].round(3).to_markdown(index=False))
    print(f"\nCalendar-arbitrage free: {surf.calendar_arbitrage_free}  (fit time {t_svi:.2f}s)\n")

    print("Heston calibration")
    is_synth = chain.attrs["source"].startswith("synthetic")
    header = "| Parameter | Fitted |" + (" True |" if is_synth else "")
    print(header + "\n|---|---:|" + ("---:|" if is_synth else ""))
    for name, val in cal.params.to_dict().items():
        line = f"| {name} | {val:.4f} |"
        if is_synth:
            line += f" {getattr(DEFAULT_TRUE_PARAMS, name):.4f} |"
        print(line)
    print(
        f"\nIV RMSE {cal.rmse_iv_bp:.1f} bp, max error {cal.max_abs_iv_bp:.1f} bp, "
        f"{cal.n_evals} model evaluations, {t_cal:.2f}s, Feller {cal.params.feller:+.4f}\n"
    )

    # --- 3D surface, plotted against standardised moneyness ln(K/F)/sqrt(T) so each
    # expiry shows the strike range it was actually fitted on
    x = np.linspace(-0.35, 0.26, 50)
    Ts = np.linspace(min(s.T for s in surf.slices), max(s.T for s in surf.slices), 50)
    X, Tg = np.meshgrid(x, Ts)
    Z = np.array([np.sqrt(surf.total_variance(x * np.sqrt(T), T) / T) for T in Ts]) * 100
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Tg, Z, cmap="viridis", edgecolor="none", alpha=0.85)
    ax.scatter(
        quotes["log_moneyness"] / np.sqrt(quotes["T"]),
        quotes["T"],
        quotes["iv"] * 100,
        color="k",
        s=5,
    )
    ax.set_xlabel(r"$\ln(K/F)/\sqrt{T}$")
    ax.set_ylabel("maturity (years)")
    ax.set_zlabel("implied vol (%)")
    ax.view_init(elev=25, azim=-50)
    ax.set_title("Arbitrage-checked SVI implied volatility surface")
    fig.tight_layout()
    fig.savefig(OUT / "vol_surface.png", dpi=130)

    # --- smiles: market vs SVI vs Heston
    n = len(surf.slices)
    cols = 3
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(12, 3.2 * np.ceil(n / cols)))
    for ax, sl in zip(axes.flat, surf.slices, strict=False):
        g = quotes[quotes["T"] == sl.T]
        ax.errorbar(
            g["log_moneyness"],
            g["iv"] * 100,
            yerr=[(g["iv"] - g["iv_bid"]) * 100, (g["iv_ask"] - g["iv"]) * 100],
            fmt="o",
            ms=3,
            color="k",
            alpha=0.6,
            label="market bid/ask",
        )
        kk = np.linspace(g["log_moneyness"].min(), g["log_moneyness"].max(), 100)
        ax.plot(kk, sl.params.implied_vol(kk, sl.T) * 100, color="#1f77b4", label="SVI")
        fwd = S * np.exp((r - q) * sl.T)
        hv = heston.implied_vols(S, fwd * np.exp(kk), np.full_like(kk, sl.T), r, cal.params, q)
        ax.plot(kk, hv * 100, "--", color="#d62728", label="Heston (calibrated)")
        ax.set_title(f"T = {sl.T:.2f}y")
        ax.set_xlabel("ln(K/F)")
        ax.set_ylabel("IV (%)")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "smiles.png", dpi=130)


def convergence_studies() -> None:
    S, K, T, r, sigma, q = 100.0, 105.0, 0.5, 0.04, 0.25, 0.01
    exact = float(bs.price(S, K, T, r, sigma, q))
    paths = np.array([1_000, 4_000, 16_000, 64_000, 256_000])
    variants = {
        "plain": dict(antithetic=False, control_variate=False),
        "antithetic": dict(antithetic=True, control_variate=False),
        "antithetic + control variate": dict(antithetic=True, control_variate=True),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, kw in variants.items():
        se = [mc.european(S, K, T, r, sigma, q, n_paths=int(n), seed=1, **kw).stderr for n in paths]
        axes[0].loglog(paths, se, "o-", label=name)
    axes[0].loglog(paths, 3 / np.sqrt(paths), "k:", label=r"$\propto 1/\sqrt{N}$")
    axes[0].set_xlabel("paths")
    axes[0].set_ylabel("standard error")
    axes[0].set_title("Monte Carlo: European call")
    axes[0].legend(fontsize=8)

    steps = np.array([10, 25, 50, 100, 250, 500, 1000, 2000])
    err = [abs(binomial_price(S, K, T, r, sigma, q, steps=int(n)) - exact) for n in steps]
    axes[1].loglog(steps, err, "o-", label="CRR tree |error|")
    axes[1].loglog(steps, 0.5 / steps, "k:", label=r"$\propto 1/N$")
    axes[1].set_xlabel("tree steps")
    axes[1].set_ylabel("absolute pricing error")
    axes[1].set_title("Binomial tree convergence to Black-Scholes")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=130)

    print("Pricer cross-check (S=100, K=105, T=0.5, r=4%, q=1%, sigma=25%)\n")
    print(
        "| Method | Price | Std. error | Error vs. exact (in std. errors) |\n|---|---:|---:|---:|"
    )
    print(f"| Black-Scholes (closed form) | {exact:.4f} | - | - |")
    tree = binomial_price(S, K, T, r, sigma, q, steps=2000)
    print(f"| CRR binomial, 2,000 steps | {tree:.4f} | - | - |")
    for name, kw in variants.items():
        res = mc.european(S, K, T, r, sigma, q, n_paths=256_000, seed=1, **kw)
        z = (res.price - exact) / res.stderr
        print(f"| Monte Carlo, {name} (256k) | {res.price:.4f} | {res.stderr:.4f} | {z:+.2f} |")
    a_plain = mc.arithmetic_asian(
        S, 100, 1.0, r, sigma, q, n_paths=50_000, control_variate=False, seed=1
    )
    a_cv = mc.arithmetic_asian(S, 100, 1.0, r, sigma, q, n_paths=50_000, seed=1)
    print(
        f"\nArithmetic Asian call: plain {a_plain.price:.4f} ± {a_plain.stderr:.4f}, "
        f"geometric control variate {a_cv.price:.4f} ± {a_cv.stderr:.5f} "
        f"({a_plain.stderr / a_cv.stderr:.0f}x lower std. error)"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", help="use a live Yahoo Finance chain instead of synthetic data")
    args = ap.parse_args()
    chain = yahoo_chain(args.ticker) if args.ticker else synthetic_chain(seed=0)
    surface_and_calibration(chain)
    convergence_studies()
    print(f"\nCharts saved to {OUT}")
