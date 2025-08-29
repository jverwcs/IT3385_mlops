# apps/app.py
import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# -------------------- Page config --------------------
st.set_page_config(page_title="Wheat-Seeds-Model", page_icon="🌾", layout="centered")
st.title("🌾 Wheat Seed Type - Real-time Prediction")
st.caption("Loads PyCaret pipeline. Supports single and batch predictions.")


# Exact input columns the model expects (must match training)
FEATURES = ["Compactness", "Length", "Width", "AsymmetryCoeff", "Groove"]

# -------------------- Model selector --------------------
# Default to the common names; change if your filename differs (stem = path without .pkl)
with st.sidebar:
    st.markdown("### Navigation")
    st.page_link("Homepage.py", label="Homepage", icon="👑")
    st.page_link("pages/Housing.py", label="Housing", icon="🏠")

    
MODEL_STEM = "models/wheat_seeds_best"   # <-- no .pkl

@st.cache_resource(show_spinner=True)
def get_model():
    return load_model(MODEL_STEM)  # PyCaret appends .pkl automatically

try:
    pipeline = get_model()
    st.success(f"✅ Loaded model: {MODEL_STEM}.pkl")
except Exception as e:
    st.error(
        "❌ Could not load the model.\n\n"
        f"- Ensure the file exists at `{MODEL_STEM}.pkl` inside the `models/` folder.\n"
        f"- Exact filename/spelling matters.\n\nError: {e}"
    )
    st.stop()

# -------------------- Helpers --------------------
def extract_label_score(out: pd.DataFrame) -> pd.DataFrame:
    """Normalize PyCaret outputs to ['label','score'] for both single & batch.
       Handles:
       - v2: 'Label','Score'
       - v3: 'prediction_label','prediction_score'
       - per-class cols: 'prediction_score_1/2/3' OR 'prediction_probability_1/2/3'
       - fallback: softmax of 'raw_score_*' if present
    """
    # ---- label ----
    if "prediction_label" in out.columns:
        label_col = "prediction_label"
    elif "Label" in out.columns:
        label_col = "Label"
    else:
        label_col = None

    if label_col is not None:
        label = out[label_col]
    else:
        label = pd.Series([None] * len(out), index=out.index, dtype=object)

    # ---- preferred single-column score ----
    if "Score" in out.columns and not out["Score"].isna().all():
        score = out["Score"]

    elif "prediction_score" in out.columns and not out["prediction_score"].isna().all():
        score = out["prediction_score"]

    else:
        # ---- per-class probabilities/scores ----
        proba_cols = [c for c in out.columns if c.startswith("prediction_probability_")]
        multi_cols = [c for c in out.columns if c.startswith("prediction_score_")]

        def pick_by_label(row, prefix):
            if label_col is None:
                return np.nan
            key = f"{prefix}{int(row[label_col])}"
            return row[key] if key in row.index else np.nan

        if proba_cols:
            # choose prob for the predicted label; if missing, use row-wise max
            vals = out.apply(lambda r: pick_by_label(r, "prediction_probability_"), axis=1)
            if vals.isna().any():
                vals = out[proba_cols].max(axis=1)
            score = vals

        elif multi_cols:
            # choose score for the predicted label; if missing, use row-wise max
            vals = out.apply(lambda r: pick_by_label(r, "prediction_score_"), axis=1)
            if vals.isna().any():
                vals = out[multi_cols].max(axis=1)
            score = vals

        else:
            # ---- raw scores -> softmax -> max ----
            raw_cols = [c for c in out.columns if c.startswith("raw_score_")]
            if raw_cols:
                logits = out[raw_cols].to_numpy(dtype=float)
                logits = logits - np.max(logits, axis=1, keepdims=True)
                prob = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
                score = pd.Series(prob.max(axis=1), index=out.index)
            else:
                score = pd.Series([None] * len(out), index=out.index, dtype=float)

    return pd.DataFrame({"label": label, "score": score})
def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="raise")
    return df

# -------------------- Single prediction (multi input modes) --------------------
st.subheader("Single Prediction")

DEFAULTS = {
    "Compactness": 0.87,
    "Length": 5.50,
    "Width": 3.10,
    "AsymmetryCoeff": 2.00,
    "Groove": 5.10,
}

def _predict_and_show(row_df: pd.DataFrame):
    row_df = row_df[FEATURES].astype(float).round(2)
    out = predict_model(pipeline, data=row_df, raw_score=True)
    res = extract_label_score(out)
    label = res["label"].iloc[0]
    score = res["score"].iloc[0]
    if pd.isna(label):
        st.warning("Prediction produced no label. Check the model pipeline.")
    else:
        msg = f"Type {int(label)}"
        if not pd.isna(score):
            msg += f" (Prob: {float(score):.4f})"
        st.success(msg)
    with st.expander("Raw prediction row"):
        st.dataframe(out)

mode = st.radio("Input mode", ["Table", "Form"], horizontal=True)

if mode == "Table":
    # Option 1: data_editor (best for copy/paste and keyboard entry)
    default_row = pd.DataFrame([DEFAULTS]).round(2)
    edited = st.data_editor(
        default_row,
        num_rows="fixed",
        hide_index=True,
        use_container_width=True,
        column_config={c: st.column_config.NumberColumn(format="%.2f") for c in FEATURES},
        key="single_row_editor",
    )
    if st.button("Predict", type="primary", key="btn_table"):
        try:
            _predict_and_show(edited)
        except Exception as e:
            st.error(f"Prediction error: {e}")

else:
    # Option 2: plain text inputs with strict validation
    vals, errs = {}, []
    cols = st.columns(2)
    for i, name in enumerate(FEATURES):
        with cols[i % 2]:
            s = st.text_input(name, value=f"{DEFAULTS[name]:.2f}", key=f"txt_{name}")
        try:
            vals[name] = round(float(s.strip()), 2)
        except Exception:
            errs.append(f"{name} must be a number")
    if st.button("Predict", type="primary", key="btn_form"):
        if errs:
            st.error(" | ".join(errs))
        else:
            try:
                _predict_and_show(pd.DataFrame([vals]))
            except Exception as e:
                st.error(f"Prediction error: {e}")


st.divider()

# -------------------- Batch prediction --------------------
st.subheader("Batch Prediction (CSV upload)")
st.write("CSV must have these **exact** headers: " + ", ".join(FEATURES))
csv_file = st.file_uploader("Upload CSV", type=["csv"])

if csv_file is not None:
    try:
        df_in = pd.read_csv(csv_file)
        missing = [c for c in FEATURES if c not in df_in.columns]
        if missing:
            st.error(f"CSV missing columns: {missing}")
        else:
            df_in = df_in[FEATURES].copy()
            df_in = coerce_numeric(df_in, FEATURES).round(2)

            out = predict_model(pipeline, data=df_in, raw_score=True)
            res = extract_label_score(out)

            # Show results (just label+score). If you want inputs too, concat with df_in.
            st.dataframe(res)

            st.download_button(
                "Download predictions as CSV",
                res.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing file: {e}")

st.caption("Tip: if your model filename is different, change the **Model path** in the sidebar (no `.pkl`).")