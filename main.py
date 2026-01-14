import streamlit as st
import joblib
import numpy as np
import pandas as pd


try:
    model = joblib.load('aqi_rf_model.pkl')
    scaler = joblib.load('aqi_scaler.pkl')
except FileNotFoundError:
    st.error("Error: 'aqi_rf_model.pkl' or 'aqi_scaler.pkl' not found. Please run your training script first.")

def get_aqi_bucket(x):
    if x <= 50: return "Good", "#00e400"
    elif x <= 100: return "Satisfactory", "#ffff00"
    elif x <= 200: return "Moderate", "#ff7e00"
    elif x <= 300: return "Poor", "#ff0000"
    elif x <= 400: return "Very Poor", "#8f3f97"
    else: return "Severe", "#7e0023"

st.set_page_config(page_title="AQI Prediction App", page_icon="🌍")
st.title("🌍 Real-Time Air Quality Predictor")
st.markdown("Enter the pollutant concentrations to predict the Air Quality Index (AQI).")

col1, col2, col3 = st.columns(3)

with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, step=1.0, value=60.0)
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, step=1.0, value=100.0)
    no = st.number_input("NO (µg/m³)", min_value=0.0, step=0.1, value=2.5)

with col2:
    no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, step=0.1, value=30.0)
    nox = st.number_input("NOx (µg/m³)", min_value=0.0, step=0.1, value=18.0)
    nh3 = st.number_input("NH3 (µg/m³)", min_value=0.0, step=0.1, value=8.5)

with col3:
    co = st.number_input("CO (mg/m³)", min_value=0.0, step=0.01, value=0.1)
    so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, step=0.1, value=12.0)
    o3 = st.number_input("O3 (µg/m³)", min_value=0.0, step=1.0, value=125.0)

# --- 4. PREDICTION LOGIC ---
if st.button("Predict AQI"):
    input_data = np.array([[pm25, pm10, no, no2, nox, nh3, co, so2, o3]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    bucket_name, color = get_aqi_bucket(prediction)
    st.markdown("---")
    st.subheader(f"Predicted AQI: **{prediction:.2f}**")
    st.markdown(f"**Health Category:** <span style='color:{color}; font-weight:bold;'>{bucket_name}</span>", unsafe_allow_html=True)
    
    if bucket_name == "Severe" or bucket_name == "Very Poor":
        st.warning("⚠️ High pollution levels detected. Stay indoors and use air purifiers.")