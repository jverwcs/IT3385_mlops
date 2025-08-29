# apps/pages/02_Housing.py
import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Page config ----------
st.set_page_config(page_title="Housing — Price Prediction", page_icon="🏠", layout="wide")
st.title("🏠 Melbourne House Price — Real-time Prediction")

# -------------------- Model selector --------------------
# Default to the common names; change if your filename differs (stem = path without .pkl)
with st.sidebar:
    st.markdown("### Navigation")
    st.page_link("Homepage.py", label="Homepage", icon="👑")
    st.page_link("pages/Wheat_Seeds.py", label="Wheat Seeds", icon="🌾")


# ---------- Model loader (local .pkl only; no MLflow) ----------
# file = .../apps/pages/Housing_Test.py
PAGES_DIR = Path(__file__).resolve().parent        # .../apps/pages
APPS_DIR  = PAGES_DIR.parent                       # .../apps
PROJ_ROOT = APPS_DIR.parent                        # ← repo root

MODELS_DIR = PROJ_ROOT / "models"
MODEL_STEM = MODELS_DIR / "final_gbr_pipeline"     # for PyCaret load_model(stem)
MODEL_PKL  = MODELS_DIR / "final_gbr_pipeline.pkl" # for joblib.load


try:
    from pycaret.regression import load_model as pc_load_model
except Exception:
    pc_load_model = None

@st.cache_resource
def load_local_model():
    # Try PyCaret-saved pipeline first (expects stem without .pkl)
    if pc_load_model and MODEL_PKL.exists():
        try:
            m = pc_load_model(str(MODEL_STEM))
            return m, "pycaret"
        except Exception:
            pass
    # Fallback: raw sklearn/joblib pickle
    import joblib
    if MODEL_PKL.exists():
        m = joblib.load(MODEL_PKL)
        return m, "joblib"
    raise FileNotFoundError("Model file not found in models/: final_gbr_pipeline.pkl")

model, model_source = load_local_model()
st.caption(f"Model source → **{model_source}** · File: `models/final_gbr_pipeline.pkl`")

# ---------- Training schema (columns your model expects) ----------
TRAIN_FEATURES = [
    "Rooms","Bedroom2","Bathroom","Car","Landsize","BuildingArea","Distance",
    "YearBuilt","Propertycount","Postcode","Lattitude","Longtitude",
    "Suburb","Type","Method","CouncilArea","Region",
    "SaleYear","PropertyAge","BuildingArea_missing","DistanceBin",
]

def align_to_training_schema(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in TRAIN_FEATURES if c in df.columns]
    return df[keep]

# ---------- Helpers to recreate Task-2 features ----------
def _clean_str(x):
    if x is None:
        return np.nan
    try:
        if pd.isna(x):
            return np.nan
    except Exception:
        pass
    return re.sub(r"[^\w\-]+", "_", str(x)).strip("_")

def _sanitize_missing_and_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().mask(df.isna(), np.nan)
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
    df = df.copy()

    # align naming typos from original dataset
    if "Lattitude" not in df.columns and "Latitude" in df.columns:
        df["Lattitude"] = df["Latitude"]
    if "Longtitude" not in df.columns and "Longitude" in df.columns:
        df["Longtitude"] = df["Longitude"]

    # clean strings
    for c in df.select_dtypes(include=["object", "category"]).columns:
        df[c] = df[c].apply(_clean_str)

    # SaleYear
    if "Date" in df.columns:
        sdt = pd.to_datetime(df["Date"], errors="coerce")
        df["SaleYear"] = sdt.dt.year

    # PropertyAge
    if {"SaleYear","YearBuilt"}.issubset(df.columns):
        df["PropertyAge"] = df["SaleYear"] - df["YearBuilt"]

    # BuildingArea_missing flag
    if "BuildingArea" in df.columns:
        df["BuildingArea_missing"] = df["BuildingArea"].isna().astype(int)

    # DistanceBin
    if "Distance" in df.columns:
        non_null = df["Distance"].dropna()
        if len(non_null) >= 5:
            try:
                q = pd.qcut(non_null, q=5, duplicates="drop", labels=False)
                df.loc[non_null.index, "DistanceBin"] = q.astype("float64")
            except Exception:
                df["DistanceBin"] = pd.cut(df["Distance"], bins=5, labels=False, include_lowest=True).astype("float64")
        else:
            if non_null.nunique() <= 1:
                df["DistanceBin"] = 2.0 if non_null.nunique()==1 else np.nan
            else:
                df["DistanceBin"] = pd.cut(df["Distance"], bins=5, labels=False, include_lowest=True).astype("float64")

    return _sanitize_missing_and_dtypes(df)

# ---------- UI ----------
tab1, tab2 = st.tabs(["Single prediction", "Batch prediction"])

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
        row_fe = align_to_training_schema(row_fe)
        pred = float(np.asarray(model.predict(row_fe))[0])
        st.success(f"💰 Predicted Price: **${pred:,.0f}**")

with tab2:
    st.subheader("Upload CSV for batch scoring")
    up = st.file_uploader("Choose CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        df_fe = add_task2_features(df)
        df_fe = align_to_training_schema(df_fe)
        preds = np.asarray(model.predict(df_fe), dtype=float)
        out = df.copy()
        out["PredictedPrice"] = preds
        st.dataframe(out.head(20))
        st.download_button(
            "⬇️ Download predictions CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="batch_predictions.csv",
            mime="text/csv"
        )