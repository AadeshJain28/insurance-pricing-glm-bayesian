"""Streamlit demo — quote a motor policy: technical price -> optimal commercial price.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from insurance_pricing.features.build_features import prepare_risk_features  # noqa: E402
from insurance_pricing.pricing.optimize import optimal_price  # noqa: E402
from insurance_pricing.pricing.run_pricing import calibrate_logistic_demand  # noqa: E402

MODELS = ROOT / "models" / "risk_glms.joblib"

st.set_page_config(page_title="Insurance Pricing", page_icon="🚗", layout="wide")


@st.cache_resource
def load_models():
    import joblib

    return joblib.load(MODELS)


st.title("🚗 Motor Insurance Pricing")
st.caption("Technical price from frequency-severity GLMs → profit-optimal commercial price")

if not MODELS.exists():
    st.error("Models not found. Run `python -m insurance_pricing.models.risk_models` first.")
    st.stop()

bundle = load_models()

# ------------------------- policy inputs -------------------------
st.subheader("Policy details")
c1, c2, c3 = st.columns(3)
with c1:
    driv_age = st.slider("Driver age", 18, 90, 42)
    veh_age = st.slider("Vehicle age (years)", 0, 20, 5)
    veh_power = st.slider("Vehicle power", 4, 15, 6)
with c2:
    bonus = st.slider("Bonus-Malus (100 = neutral)", 50, 150, 100)
    density = st.select_slider("Area density (people/km²)",
                               options=[10, 50, 200, 1000, 5000, 27000], value=200)
    exposure = st.slider("Exposure (years)", 0.1, 1.0, 1.0, step=0.1)
with c3:
    region = st.selectbox("Region", ["R11", "R24", "R25", "R31", "R52", "R53", "R72", "R82", "R93"])
    brand = st.selectbox("Vehicle brand", [f"B{i}" for i in [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14]])
    gas = st.selectbox("Fuel", ["Regular", "Diesel"])
    area = st.selectbox("Area", list("ABCDEF"))

policy = pd.DataFrame([{
    "VehPower": veh_power, "VehAge": veh_age, "DrivAge": driv_age, "BonusMalus": bonus,
    "Density": float(density), "VehBrand": brand, "VehGas": gas, "Area": area, "Region": region,
    "ClaimNb": 0, "ClaimAmount": 0.0, "Exposure": exposure, "PurePremium": 0.0,
}])
policy = prepare_risk_features(policy)

freq = float(np.clip(bundle["frequency"].predict(policy), 1e-6, None)[0])
sev = float(np.clip(bundle["severity"].predict(policy), 1.0, None)[0])
tweedie = float(np.clip(bundle["tweedie"].predict(policy), 1.0, None)[0])

st.subheader("Technical price (expected claim cost per year)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Claim frequency", f"{freq:.4f}", help="Expected claims per year of exposure")
m2.metric("Severity if a claim occurs", f"€{sev:,.0f}")
m3.metric("Freq × Severity", f"€{freq * sev:,.2f}")
m4.metric("Tweedie (used)", f"€{tweedie:,.2f}",
          help="Direct pure-premium model — better calibrated than the product; see README")

# ------------------------- commercial price -------------------------
st.subheader("Commercial price")
st.info(
    "⚠️ **The demand curve below is an assumption, not an estimate.** The available quote data "
    "cannot identify a price elasticity — conversion *rises* with price (corr +0.51) because "
    "premium is set from risk. Treat the optimum as conditional on the slider.",
    icon="⚠️",
)
d1, d2 = st.columns(2)
elasticity = d1.slider("Assumed price elasticity", -5.0, -0.3, -1.5, step=0.1)
q_ref = d2.slider("Assumed conversion at 1.5× cost", 0.05, 0.60, 0.20, step=0.05)

demand = calibrate_logistic_demand(tweedie * 1.5, q_ref, elasticity)
res = optimal_price(tweedie, demand, tweedie * 0.5, tweedie * 3.0, 200)

p1, p2, p3 = st.columns(3)
p1.metric("Optimal price", f"€{res.best_price:,.2f}", f"{res.best_price / tweedie:.2f}× cost")
p2.metric("Conversion at optimum", f"{res.best_conversion:.1%}")
p3.metric("Expected profit / quote", f"€{res.best_profit:,.2f}")

chart = pd.DataFrame({"price": res.prices, "expected profit": res.profits,
                      "conversion": demand(res.prices) * res.profits.max()}).set_index("price")
st.line_chart(chart)
st.caption(
    "Profit = (price − technical cost) × P(convert). Raising price lifts margin but cuts conversion; "
    "the peak is the profit-maximising quote. Conversion is rescaled for display."
)
