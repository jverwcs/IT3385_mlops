# app.py
import os, re
import numpy as np
import pandas as pd
import streamlit as st
import mlflow
import mlflow.pyfunc

# -----------------------------
# Config
# -----------------------------
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME  = os.getenv("MODEL_NAME", "MelbournePriceModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Staging")  # or "Production"
LOCAL_MODEL_PATH = "../models/final_gbr_pipeline"  # fallback to Task 2 saved pipeline

st.set_page_config(page_title="Melbourne House Price — Realtime", layout="wide")

# -----------------------------
# Training schema (from Task 2)
# -----------------------------
TRAIN_FEATURES = [
    # raw numerics
    "Rooms","Bedroom2","Bathroom","Car","Landsize","BuildingArea","Distance",
    "YearBuilt","Propertycount","Postcode",
    # GPS typo columns as in the original CSV / training
    "Lattitude","Longtitude",
    # categoricals
    "Suburb","Type","Method","CouncilArea","Region",
    # engineered features added at inference
    "SaleYear","PropertyAge","BuildingArea_missing","DistanceBin",
]

def align_to_training_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only features used at training; drop anything extra (e.g., Date/Latitude/Longitude)."""
    keep = [c for c in TRAIN_FEATURES if c in df.columns]
    return df[keep]

# -----------------------------
# Helpers (match Task 2)
# -----------------------------
def _clean_str(x):
    # Always use np.nan for missing; avoid pandas NA in object/categorical
    if x is None:
        return np.nan
    try:
        if pd.isna(x):
            return np.nan
    except Exception:
        pass
    return re.sub(r"[^\w\-]+", "_", str(x)).strip("_")

def _sanitize_missing_and_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure there are NO pandas NA dtypes left; convert to numpy-friendly types."""
    df = df.copy()
    # Standardize all missing to np.nan
    df = df.mask(df.isna(), np.nan)

    # Convert problematic extension dtypes to numpy dtypes
    for c in df.columns:
        dt = df[c].dtype
        if pd.api.types.is_extension_array_dtype(dt):
            if pd.api.types.is_integer_dtype(dt) or pd.api.types.is_boolean_dtype(dt):
                df[c] = df[c].astype("float64")
            elif pd.api.types.is_string_dtype(dt):
                df[c] = df[c].astype("object")
        if pd.api.types.is_categorical_dtype(dt):
            df[c] = df[c].astype("object")
    return df

def add_task2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate engineered features & clean categoricals to match training schema.
       Also ensure NO pandas.NA survives (everything -> np.nan-compatible dtypes).
    """
    df = df.copy()

    # --- alias to match training CSV typos ---
    if "Lattitude" not in df.columns and "Latitude" in df.columns:
        df["Lattitude"] = df["Latitude"]
    if "Longtitude" not in df.columns and "Longitude" in df.columns:
        df["Longtitude"] = df["Longitude"]

    # Clean strings first (avoid special chars in categoricals)
    for c in df.select_dtypes(include=["object", "category"]).columns:
        df[c] = df[c].apply(_clean_str)

    # SaleYear from Date
    if "Date" in df.columns:
        sdt = pd.to_datetime(df["Date"], errors="coerce")
        df["SaleYear"] = sdt.dt.year

    # PropertyAge = SaleYear - YearBuilt
    if {"SaleYear", "YearBuilt"}.issubset(df.columns):
        df["PropertyAge"] = df["SaleYear"] - df["YearBuilt"]

    # Missing flag for BuildingArea
    if "BuildingArea" in df.columns:
        df["BuildingArea_missing"] = df["BuildingArea"].isna().astype(int)

    # DistanceBin: quantile bins for many rows; safe equal-width for few rows
    if "Distance" in df.columns:
        non_null = df["Distance"].dropna()
        if len(non_null) >= 5:
            try:
                bins = pd.qcut(non_null, q=5, duplicates="drop", labels=False)
                df.loc[non_null.index, "DistanceBin"] = bins.astype("float64")
            except Exception:
                df["DistanceBin"] = pd.cut(df["Distance"], bins=5, labels=False, include_lowest=True).astype("float64")
        else:
            if non_null.nunique() == 0:
                df["DistanceBin"] = np.nan
            elif non_null.nunique() == 1:
                df["DistanceBin"] = 2.0
            else:
                df["DistanceBin"] = pd.cut(df["Distance"], bins=5, labels=False, include_lowest=True).astype("float64")

    # Final: normalize missing & dtypes (kills any lingering pd.NA)
    df = _sanitize_missing_and_dtypes(df)
    return df

# -----------------------------
# Load model (registry → fallback to local)
# -----------------------------
@st.cache_resource
def load_model():
    mlflow.set_tracking_uri(MLFLOW_URI)
    uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    try:
        model = mlflow.pyfunc.load_model(uri)
        source = f"registry:{uri}"
    except Exception:
        from pycaret.regression import load_model as pc_load_model
        model = pc_load_model(LOCAL_MODEL_PATH)
        source = f"local:{LOCAL_MODEL_PATH}"
    return model, source

model, model_source = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🏠 Melbourne House Price — Real-time Prediction")
st.caption(f"Model source → **{model_source}** · Tracking URI: {MLFLOW_URI}")

tab1, tab2 = st.tabs(["Single prediction", "Batch prediction (CSV)"])

with tab1:
    st.subheader("Enter features")
    c1, c2, c3 = st.columns(3)

    Rooms        = c1.number_input("Rooms", min_value=0, step=1, value=3)
    Bedroom2     = c1.number_input("Bedroom2", min_value=0, step=1, value=3)
    Bathroom     = c1.number_input("Bathroom", min_value=0, step=1, value=1)
    Car          = c1.number_input("Car", min_value=0, step=1, value=1)

    Landsize     = c2.number_input("Landsize", min_value=0.0, value=150.0)
    BuildingArea = c2.number_input("BuildingArea", min_value=0.0, value=120.0)
    YearBuilt    = c2.number_input("YearBuilt", min_value=1800, max_value=2050, value=1990)
    Distance     = c2.number_input("Distance (km to CBD)", min_value=0.0, value=10.0)

    Suburb       = c3.text_input("Suburb", value="Abbotsford")
    Type         = c3.text_input("Type (h/u/t/...)", value="h")
    Method       = c3.text_input("Method (S/SP/PI/...)", value="S")
    CouncilArea  = c3.text_input("CouncilArea", value="Yarra")
    Region       = c3.text_input("Region", value="Northern_Metropolitan")
    Postcode     = c3.number_input("Postcode", min_value=0, step=1, value=3067)
    Latitude     = c3.number_input("Latitude", value=-37.80, format="%.5f")
    Longitude    = c3.number_input("Longitude", value=144.99000, format="%.5f")
    Propertycount= c3.number_input("Propertycount", min_value=0, value=5000)
    Date         = st.text_input("Date (YYYY-MM-DD)", value="2017-05-20")

    if st.button("Predict price"):
        row = pd.DataFrame([{
            "Rooms": Rooms, "Bedroom2": Bedroom2, "Bathroom": Bathroom, "Car": Car,
            "Landsize": Landsize, "BuildingArea": BuildingArea, "YearBuilt": YearBuilt,
            "Distance": Distance, "Suburb": Suburb, "Type": Type, "Method": Method,
            "CouncilArea": CouncilArea, "Region": Region, "Postcode": Postcode,
            "Latitude": Latitude, "Longitude": Longitude, "Propertycount": Propertycount,
            "Date": Date
        }])
        row_fe = add_task2_features(row)
        row_fe = align_to_training_schema(row_fe)    # << filter to training columns
        pred = float(model.predict(row_fe)[0])
        st.success(f"💰 Predicted Price: **${pred:,.0f}**")

with tab2:
    st.subheader("Upload CSV for batch scoring")
    st.caption("CSV may contain the original raw columns; the app will recreate engineered features.")
    up = st.file_uploader("Choose CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        df_fe = add_task2_features(df)
        df_fe = align_to_training_schema(df_fe)      # << filter to training columns
        preds = model.predict(df_fe)
        out = df.copy()
        out["PredictedPrice"] = np.asarray(preds, dtype=float)
        st.write(out.head(20))
        st.download_button(
            "⬇️ Download predictions CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="batch_predictions.csv",
            mime="text/csv"
        )
