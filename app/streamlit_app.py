"""Interactive volatility surface explorer.

pip install -e ".[app]"
streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from volsurf import binomial_price, build_surface, heston
from volsurf import black_scholes as bs
from volsurf import monte_carlo as mc
from volsurf.data import synthetic_chain, yahoo_chain
from volsurf.heston import HestonParams

st.set_page_config(page_title="volsurf", layout="wide")
st.title("volsurf: implied volatility surface explorer")

with st.sidebar:
    st.header("Data")
    source = st.radio("Option chain", ["Synthetic (Heston)", "Yahoo Finance"])
    if source == "Yahoo Finance":
        ticker = st.text_input("Ticker", "SPY")
        r = st.number_input("Risk-free rate", value=0.04, step=0.005, format="%.3f")
        q = st.number_input("Dividend yield", value=0.013, step=0.001, format="%.3f")
    else:
        st.caption("True Heston parameters used to generate the chain")
        kappa = st.slider("kappa (mean reversion)", 0.1, 5.0, 1.8)
        theta = st.slider("theta (long-run variance)", 0.01, 0.15, 0.045)
        sigma = st.slider("sigma (vol of vol)", 0.1, 1.2, 0.55)
        rho = st.slider("rho (spot/vol correlation)", -0.95, 0.5, -0.72)
        v0 = st.slider("v0 (current variance)", 0.01, 0.15, 0.03)
        noise = st.slider("Quote noise (bp of vol)", 0.0, 50.0, 15.0)
        seed = st.number_input("Seed", value=0, step=1)


@st.cache_data(show_spinner="Loading option chain...")
def load_chain(source, *args):
    if source == "Yahoo Finance":
        return yahoo_chain(*args)
    k, th, sg, rh, v, nz, sd = args
    return synthetic_chain(params=HestonParams(k, th, sg, rh, v), noise_bp=nz, seed=sd)


args = (
    (ticker, r, q)
    if source == "Yahoo Finance"
    else (kappa, theta, sigma, rho, v0, noise, int(seed))
)
chain = load_chain(source, *args)
if chain.empty:
    st.error(
        "No usable two-sided quotes. Yahoo often reports zero bids/asks outside US market hours."
    )
    st.stop()

S, r, q = chain.attrs["spot"], chain.attrs["r"], chain.attrs["q"]
surf = build_surface(chain)
quotes = surf.quotes
cal = heston.calibrate(S, quotes["strike"], quotes["T"], quotes["iv"], r, q)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Spot", f"{S:,.2f}")
c2.metric("Options used", len(quotes))
c3.metric("Heston IV RMSE", f"{cal.rmse_iv_bp:.1f} bp")
c4.metric(
    "Arbitrage-free",
    "Yes" if surf.butterfly_arbitrage_free and surf.calendar_arbitrage_free else "No",
)

tab_surface, tab_smiles, tab_heston, tab_pricer = st.tabs(
    ["Surface", "Smiles", "Heston calibration", "Pricer"]
)

with tab_surface:
    x = np.linspace(-0.35, 0.26, 50)
    Ts = np.linspace(surf.slices[0].T, surf.slices[-1].T, 50)
    Z = np.array([np.sqrt(surf.total_variance(x * np.sqrt(T), T) / T) for T in Ts]) * 100
    fig = go.Figure(go.Surface(x=x, y=Ts, z=Z, colorscale="Viridis", colorbar=dict(title="IV %")))
    fig.add_trace(
        go.Scatter3d(
            x=quotes["log_moneyness"] / np.sqrt(quotes["T"]),
            y=quotes["T"],
            z=quotes["iv"] * 100,
            mode="markers",
            marker=dict(size=2, color="black"),
            name="quotes",
        )
    )
    fig.update_layout(
        height=650,
        scene=dict(xaxis_title="ln(K/F)/√T", yaxis_title="Maturity (y)", zaxis_title="IV (%)"),
    )
    st.plotly_chart(fig, width="stretch")

with tab_smiles:
    sl = st.select_slider("Expiry", options=surf.slices, format_func=lambda s: f"{s.T:.3f}y")
    g = quotes[quotes["T"] == sl.T]
    kk = np.linspace(g["log_moneyness"].min(), g["log_moneyness"].max(), 200)
    fwd = S * np.exp((r - q) * sl.T)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=g["log_moneyness"],
            y=g["iv"] * 100,
            mode="markers",
            name="mid",
            error_y=dict(
                type="data",
                symmetric=False,
                array=(g["iv_ask"] - g["iv"]) * 100,
                arrayminus=(g["iv"] - g["iv_bid"]) * 100,
            ),
        )
    )
    fig.add_trace(go.Scatter(x=kk, y=sl.params.implied_vol(kk, sl.T) * 100, name="SVI"))
    hv = heston.implied_vols(S, fwd * np.exp(kk), np.full_like(kk, sl.T), r, cal.params, q)
    fig.add_trace(go.Scatter(x=kk, y=hv * 100, name="Heston", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="ln(K/F)", yaxis_title="IV (%)", height=450)
    st.plotly_chart(fig, width="stretch")
    st.write(
        f"SVI fit error {sl.rmse_iv_bp:.1f} bp · butterfly-arbitrage free: {sl.butterfly_free}"
    )

with tab_heston:
    st.dataframe(pd.DataFrame({"fitted": cal.params.to_dict()}).T.style.format("{:.4f}"))
    st.write(
        f"Fitted {cal.n_options} options in {cal.n_evals} model evaluations. "
        f"IV RMSE {cal.rmse_iv_bp:.1f} bp, max {cal.max_abs_iv_bp:.1f} bp. "
        f"Feller condition 2κθ − σ² = {cal.params.feller:+.4f} "
        f"({'satisfied' if cal.params.feller > 0 else 'violated: variance can touch zero'})."
    )

with tab_pricer:
    a, b = st.columns(2)
    K = a.number_input("Strike", value=float(round(S)))
    T = a.number_input("Maturity (years)", value=0.5, min_value=0.01)
    is_call = a.radio("Type", ["Call", "Put"], horizontal=True) == "Call"
    vol = float(surf.implied_vol(np.array([K]), T)[0])
    b.metric("Surface implied vol", f"{vol * 100:.2f}%")
    res = mc.european(S, K, T, r, vol, q, is_call, n_paths=200_000, seed=0)
    rows = {
        "Black-Scholes": float(bs.price(S, K, T, r, vol, q, is_call)),
        "Binomial (European)": binomial_price(S, K, T, r, vol, q, is_call, steps=1000),
        "Binomial (American)": binomial_price(
            S, K, T, r, vol, q, is_call, american=True, steps=1000
        ),
        f"Monte Carlo (±{1.96 * res.stderr:.3f})": res.price,
        "Heston (calibrated)": float(heston.price(S, K, T, r, cal.params, q, is_call)[0]),
    }
    b.dataframe(pd.DataFrame({"price": rows}).style.format("{:.4f}"))
    gr = bs.greeks(S, K, T, r, vol, q, is_call)
    st.dataframe(pd.DataFrame({k: [float(v)] for k, v in gr.items()}).style.format("{:.4f}"))
