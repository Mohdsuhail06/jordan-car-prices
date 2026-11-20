"""Streamlit demo: estimate a used car's price in Jordan and show WHY.

Run:  streamlit run app.py
"""
import json

import numpy as np
import pandas as pd
import streamlit as st

from jocars.cleaning import clean, parse_year
from jocars.config import ROOT
from jocars.data import load_raw
from jocars.predict import explain_price, load_model, predict_price

st.set_page_config(page_title="Jordan Car Price Estimator", page_icon="🚗", layout="centered")

LABELS = {
    "car_age": "Car age", "make": "Brand", "make_model": "Brand + model", "mileage_km": "Mileage",
    "km_per_year": "Km per year", "body_type": "Body type", "fuel_type": "Fuel type",
    "transmission": "Transmission", "seats": "Seats", "color": "Color",
    "feat_paintcondition": "Paint", "feat_body_condition": "Body condition",
    "feat_battery_range": "Battery range", "feat_battery_capacity": "Battery capacity",
    "interior_color": "Interior color", "mileage_200k_plus": "200k+ km", "city": "City",
    "feat_license": "License", "feat_insurance": "Insurance", "feat_carcustoms": "Customs",
    "condition": "Condition",
}


@st.cache_data
def listings() -> pd.DataFrame:
    """Cleaned listings, used only to fill the dropdown menus with real values."""
    df = clean(load_raw())
    return df.assign(year_n=parse_year(df["year"]))


@st.cache_data
def test_metrics() -> dict:
    return json.loads((ROOT / "reports" / "test_metrics.json").read_text())["test"]


def options(df: pd.DataFrame, col: str) -> list[str]:
    """Values of a column, most common first."""
    return df[col].dropna().value_counts().index.tolist()


df = listings()
metrics = test_metrics()

st.title("🚗 Jordan used-car price estimator")
st.caption(
    "A LightGBM model trained on 8,867 real listings from Jordanian car marketplaces (JUCars-2024). "
    f"On unseen cars, the typical error is **{metrics['median_ape']:.0%}** and "
    f"**{metrics['within_20pct']:.0%}** of cars are within ±20% of their listed price."
)

c1, c2 = st.columns(2)
make = c1.selectbox("Brand", options(df, "make"), index=options(df, "make").index("Toyota"))
models_for_make = options(df[df["make"] == make], "model") or ["Other"]
model_name = c2.selectbox("Model", models_for_make)

same = df[(df["make"] == make) & (df["model"] == model_name)]
ref = same if len(same) else df[df["make"] == make]

c1, c2 = st.columns(2)
year = c1.number_input("Model year", 1970, 2026, int(ref["year_n"].median()), step=1)
condition = c2.radio("Condition", ["Used", "New"], horizontal=True)
typical_km = ref["mileage_km"].median() if ref["mileage_km"].notna().any() else 100_000
mileage = 0 if condition == "New" else st.slider(
    "Mileage (km)", 0, 300_000, int(round(typical_km / 5_000) * 5_000), step=5_000)


def pick(label: str, col: str, container=st):
    """Selectbox whose default is the most common value for this brand/model."""
    opts = options(df, col)
    common = ref[col].mode()
    default = opts.index(common.iloc[0]) if len(common) and common.iloc[0] in opts else 0
    return container.selectbox(label, opts, index=default)


c1, c2, c3 = st.columns(3)
fuel = pick("Fuel type", "fuel_type", c1)
body = pick("Body type", "body_type", c2)
transmission = pick("Transmission", "transmission", c3)

with st.expander("More details (optional)"):
    c1, c2 = st.columns(2)
    city = pick("City", "city", c1)
    color = pick("Color", "color", c2)
    paint = pick("Paint", "feat_paintcondition", c1)
    body_cond = pick("Body condition", "feat_body_condition", c2)
    customs = pick("Customs", "feat_carcustoms", c1)
    license_ = pick("License", "feat_license", c2)
    seats = pick("Seats", "seats", c1)
    interior = pick("Interior color", "interior_color", c2)
    battery_cap = battery_range = None
    if fuel in ("Electric", "Plug-in - Hybrid"):
        battery_cap = pick("Battery capacity", "feat_battery_capacity", c1)
        battery_range = pick("Battery range", "feat_battery_range", c2)

# mileage_km in the data is the top of a 10k bracket; 200k+ is stored as a text flag
mileage_text = "200,000" if mileage >= 200_000 else None
car = {
    "make": make, "model": model_name, "year": str(year), "condition": condition,
    "mileage_km": None if mileage_text else (0 if mileage == 0 else mileage // 10_000 * 10_000 + 4_999),
    "mileage_text": mileage_text, "fuel_type": fuel, "engine_type": fuel, "body_type": body,
    "transmission": transmission, "city": city, "color": color, "interior_color": interior,
    "seats": seats, "feat_paintcondition": paint, "feat_body_condition": body_cond,
    "feat_carcustoms": customs, "feat_license": license_, "feat_insurance": "Compulsory Insurance",
    "feat_battery_capacity": battery_cap, "feat_battery_range": battery_range,
}
car = {k: v for k, v in car.items() if v is not None}

model = load_model()
price = predict_price(car, model)
err = metrics["median_ape"]

st.divider()
st.metric("Estimated listing price", f"{price:,.0f} JOD")
st.write(f"Typical range: **{price * (1 - err):,.0f} – {price * (1 + err):,.0f} JOD** "
         f"(± the model's typical error of {err:.0%}).")

st.subheader("Why this price?")
exp = explain_price(car, model)
contrib = pd.Series(exp["contributions"]).sort_values(key=np.abs, ascending=False).head(6)
st.caption(f"Starting from the average listing ({exp['base_price']:,.0f} JOD), each feature "
           "multiplies the price up or down:")
for feature, value in contrib.items():
    pct = np.exp(value) - 1
    arrow = "🟢 ▲" if pct > 0 else "🟠 ▼"
    st.write(f"{arrow} **{LABELS.get(feature, feature)}**: {pct:+.0%}")

st.divider()
st.caption(
    "Data: Thalji & Tahat, *JUCars-2024: A dataset of used car listings in Jordan*, Data in Brief 66 (2026), "
    "doi:10.1016/j.dib.2026.112682 — CC BY 4.0. Prices are **listing** prices from 2024, not final sale prices. "
    "For learning purposes; not financial advice."
)
