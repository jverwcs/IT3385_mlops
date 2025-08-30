# apps/pages/03_Used_Cars.py
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Used Car Price", page_icon="🚗", layout="wide")
st.title("🚗 Used Car Price - Real-time Prediction")
st.caption("Loads a local model from **models/**. Supports single & batch.")

# -------------------- Model selector --------------------
# Default to the common names; change if your filename differs (stem = path without .pkl)
with st.sidebar:
    st.markdown("### Navigation")
    st.page_link("Homepage.py", label="Homepage", icon="👑")
    st.page_link("pages/Wheat_Seeds.py", label="Wheat Seeds", icon="🌾")
    st.page_link("pages/Housing_Prices.py", label="Housing Prices", icon="🏠")

# ---------- paths (relative to repo root) ----------
PAGES_DIR = Path(__file__).resolve().parent        # .../apps/pages
APPS_DIR  = PAGES_DIR.parent                       # .../apps
ROOT      = APPS_DIR.parent                        # repo root

DATA_PATH   = ROOT / "datasets" / "clean_car_data.csv"
MODELS_DIR  = ROOT / "models"
MODEL_STEM  = MODELS_DIR / "naomi_car_pipeline"      # PyCaret stem (no .pkl)
MODEL_PKL   = MODELS_DIR / "naomi_car_pipeline.pkl"  # joblib/sklearn pickle

# expected inference features (must match training)
FEATURES = [
    "Brand_Model", "Location", "Year", "Car_Age", "Kilometers_Driven",
    "Fuel_Type", "Transmission", "Owner_Type", "Mileage", "Engine",
    "Power", "Seats", "Brand", "Model"
]

# ---------- loaders ----------
try:
    from pycaret.regression import load_model as pc_load_model, predict_model as pc_predict_model
except Exception:
    pc_load_model = None
    pc_predict_model = None

@st.cache_resource
def load_local_model():
    # Try PyCaret-saved pipeline first
    if pc_load_model and MODEL_PKL.exists():
        try:
            m = pc_load_model(str(MODEL_STEM))
            return m, "pycaret"
        except Exception:
            pass
    # Fallback: raw sklearn/joblib pipeline
    import joblib
    if MODEL_PKL.exists():
        m = joblib.load(MODEL_PKL)
        return m, "joblib"
    raise FileNotFoundError(
        f"Model not found. Place your file at: {MODEL_PKL}\n"
        f"(Or save via PyCaret as stem '{MODEL_STEM.name}')"
    )

@st.cache_data
def load_catalog():
    df = pd.read_csv(DATA_PATH)
    return df

model, model_src = load_local_model()
st.caption(f"Model source → **{model_src}** · File: `{MODEL_PKL.name}`")

# ---------- helpers ----------
def predict_price(df_row: pd.DataFrame) -> float:
    """Return a single numeric prediction from either PyCaret or sklearn pipeline."""
    if pc_predict_model is not None and model_src == "pycaret":
        out = pc_predict_model(model, data=df_row)
        # PyCaret regression uses 'prediction_label' (v3) or 'Label' (v2)
        col = "prediction_label" if "prediction_label" in out.columns else "Label"
        return float(out[col].iloc[0])
    # sklearn/joblib pipeline
    return float(np.asarray(model.predict(df_row))[0])

def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    # ensure all expected columns exist and order them
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    return df[FEATURES]

# ---------- UI: Single prediction ----------
cat = load_catalog()
st.subheader("Single prediction")

c1, c2, c3 = st.columns(3)

brand_model = c1.selectbox("Brand_Model", sorted(cat["Brand_Model"].dropna().unique()))
location    = c1.selectbox("Location", sorted(cat["Location"].dropna().unique()))
fuel        = c1.selectbox("Fuel Type", sorted(cat["Fuel_Type"].dropna().unique()))
trans       = c1.selectbox("Transmission", sorted(cat["Transmission"].dropna().unique()))
owner       = c1.selectbox("Owner Type", sorted(cat["Owner_Type"].dropna().unique()))

age_min, age_max = int(cat["Car_Age"].min()), int(cat["Car_Age"].max())
age_med          = int(cat["Car_Age"].median())
age        = c2.number_input("Car Age (years)", age_min, age_max, age_med)
km         = c2.number_input("Kilometers Driven", int(cat["Kilometers_Driven"].min()),
                             int(cat["Kilometers_Driven"].max()),
                             int(cat["Kilometers_Driven"].median()))
mileage    = c2.number_input("Mileage (kmpl)", float(cat["Mileage"].min()),
                             float(cat["Mileage"].max()),
                             float(cat["Mileage"].median()), step=0.1)

engine     = c3.number_input("Engine (CC)", float(cat["Engine"].min()),
                             float(cat["Engine"].max()),
                             float(cat["Engine"].median()))
power      = c3.number_input("Power (BHP)", float(cat["Power"].min()),
                             float(cat["Power"].max()),
                             float(cat["Power"].median()))
seats      = c3.selectbox("Seats", sorted(cat["Seats"].dropna().unique().astype(int)))

# derive Brand & Model, Year
parts = brand_model.split(" ", 1)
brand = parts[0]
model_name = parts[1] if len(parts) > 1 else ""
year = pd.Timestamp.today().year - int(age)

if st.button("Predict price", type="primary"):
    row = pd.DataFrame([{
        "Brand_Model": brand_model, "Location": location, "Year": year, "Car_Age": age,
        "Kilometers_Driven": km, "Fuel_Type": fuel, "Transmission": trans,
        "Owner_Type": owner, "Mileage": mileage, "Engine": engine, "Power": power,
        "Seats": seats, "Brand": brand, "Model": model_name
    }])
    row = coerce_schema(row)
    try:
        price = predict_price(row)
        st.success(f"💰 Predicted Price: **₹ {price:,.2f}** (lakhs)")
        with st.expander("Input row used for prediction"):
            st.dataframe(row)
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.divider()

# ---------- UI: Batch prediction ----------
st.subheader("Batch prediction (CSV)")
st.write("CSV must have these headers (in order):")
st.code(", ".join(FEATURES), language="text")

up = st.file_uploader("Upload CSV", type=["csv"])
if up is not None:
    try:
        df_in = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in df_in.columns]
        if missing:
            st.warning(f"Missing columns added as NaN: {missing}")
        df_in = coerce_schema(df_in)
        preds = np.asarray(
            pc_predict_model(model, data=df_in)["prediction_label"]
        ) if (pc_predict_model is not None and model_src == "pycaret") else np.asarray(model.predict(df_in))
        out = df_in.copy()
        out["PredictedPrice"] = preds.astype(float)
        st.dataframe(out.head(25))
        st.download_button(
            "Download predictions",
            out.to_csv(index=False).encode("utf-8"),
            file_name="car_price_predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Batch error: {e}")

