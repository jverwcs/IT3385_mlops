# apps/app.py
import streamlit as st

st.set_page_config(page_title="Mlops Model App", page_icon="👑", layout="wide")
st.title("MLOps Project - Homepage")


with st.sidebar:
    st.markdown("### Navigation")
    st.page_link("pages/Wheat_Seeds.py", label="Wheat Seeds", icon="🌾")
    st.page_link("pages/Housing.py", label="Housing", icon="🏠")

st.write("Use the sidebar to open the Wheat Seeds page.")