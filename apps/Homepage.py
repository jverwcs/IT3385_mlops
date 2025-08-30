# apps/app.py
import streamlit as st

st.set_page_config(page_title="Mlops Model App", page_icon="👑", layout="wide")
st.title("MLOps Project - Homepage")


with st.sidebar:
    st.markdown("### Navigation")
    st.page_link("pages/Wheat_Seeds.py", label="Wheat Seeds", icon="🌾")
    st.page_link("pages/Housing_Prices.py", label="Housing Prices", icon="🏠")
    st.page_link("pages/Car_Prices.py", label="Car Prices", icon="🚗")

st.write("Use the sidebar to open and use any of the 3 Models.")