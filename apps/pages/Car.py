import streamlit as st
import pandas as pd
import mlflow.pyfunc

dframe = pd.read_csv('clean_car_data.csv')

# Load MLflow model
MODEL_URI = "models:/CarPriceRegressor/1"
model = mlflow.pyfunc.load_model(MODEL_URI)

# Streamlit app UI
st.set_page_config(page_title="Car Price Prediction", layout="centered")
st.title("Used Car Price Prediction")

brand_model = st.selectbox("Brand_Model", sorted(dframe["Brand_Model"].dropna().unique()))
location = st.selectbox("Location", sorted(dframe["Location"].dropna().unique()))
age = st.number_input("Car Age", int(dframe["Car_Age"].min()), int(dframe["Car_Age"].max()), int(dframe["Car_Age"].median()))
km_driven = st.number_input("Kilometers Driven", int(dframe["Kilometers_Driven"].min()), int(dframe["Kilometers_Driven"].max()), int(dframe["Kilometers_Driven"].median()))
fuel = st.selectbox("Fuel Type", sorted(dframe["Fuel_Type"].dropna().unique()))
transmission = st.selectbox("Transmission", sorted(dframe["Transmission"].dropna().unique()))
owner = st.selectbox("Owner Type", sorted(dframe["Owner_Type"].dropna().unique()))
mileage = st.number_input("Mileage (kmpl)", float(dframe["Mileage"].min()), float(dframe["Mileage"].max()), float(dframe["Mileage"].median()), step=0.1)
engine = st.number_input("Engine (CC)", float(dframe["Engine"].min()), float(dframe["Engine"].max()), float(dframe["Engine"].median()))
power = st.number_input("Power (BHP)", float(dframe["Power"].min()), float(dframe["Power"].max()), float(dframe["Power"].median()))
seats = st.selectbox("Seats", sorted(dframe["Seats"].dropna().unique().astype(int)))

# Auto-derive Brand and Model
brand, model_name = brand_model.split(" ", 1)  # split at first space

# Compute Year from Car_Age
current_year = 2025  
year = current_year - age

if st.button("Predict Price"):
    input_data = pd.DataFrame([{
        "Brand_Model": brand_model,
        "Location": location,
        "Year": year,
        "Car_Age": age,
        "Kilometers_Driven": km_driven,
        "Fuel_Type": fuel,
        "Transmission": transmission,
        "Owner_Type": owner,
        "Mileage": mileage,
        "Engine": engine,
        "Power": power,
        "Seats": seats,
        "Brand": brand,
        "Model": model_name
    }])
    
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: {prediction[0]:.2f} Lakhs")