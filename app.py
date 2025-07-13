import streamlit as st
import pandas as pd
import joblib
from PIL import Image


# Load model and scaler
model = joblib.load("lgb_model.pkl")
scaler = joblib.load("scaler.pkl")



# Page config
st.set_page_config(page_title="Food Delivery Time Predictor", layout="centered", page_icon="🍽️")

st.markdown("""
    <h2 style='text-align: center; color: #4B8BBE;'>🚚 Food Delivery Time Predictor</h2>
    <p style='text-align: center;'>Estimate how long your food delivery will take based on live conditions.</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Input fields in two columns
col1, col2 = st.columns(2)

with col1:
    distance = st.slider("🛍️ Distance (in km)", 0.5, 20.0, 5.0, 0.1)
    prep_time = st.slider("🍜 Preparation Time (min)", 1, 60, 20)
    courier_exp = st.slider("🤝 Courier Experience (years)", 0.0, 10.0, 2.0, 0.1)

with col2:
    weather = st.selectbox("☁️ Weather Condition", ['Sunny', 'Rainy', 'Stormy'])
    traffic = st.selectbox("🚗 Traffic Level", ['Low', 'Medium', 'High'])
    time_of_day = st.selectbox("🌜 Time of Day", ['Morning', 'Afternoon', 'Evening', 'Night'])
    vehicle = st.selectbox("🚚 Vehicle Type", ['Car', 'Bike', 'Scooter'])

st.markdown("---")

# Prepare input
user_input = {
    'Distance_km': distance,
    'Preparation_Time_min': prep_time,
    'Courier_Experience_yrs': courier_exp,
    'Weather': weather,
    'Traffic_Level': traffic,
    'Time_of_Day': time_of_day,
    'Vehicle_Type': vehicle
}

input_df = pd.DataFrame([user_input])
input_encoded = pd.get_dummies(input_df)

# Add missing columns
expected_cols = model.feature_name_
for col in expected_cols:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Reorder and scale
input_encoded = input_encoded[expected_cols]
input_scaled = scaler.transform(input_encoded.values)

# Prediction button
if st.button("🔢 Predict Delivery Time"):
    predicted_time = model.predict(input_scaled)[0]
    st.success(f"🍽️ Estimated Delivery Time: **{predicted_time:.2f} minutes**")

    st.markdown("""
    <div style='text-align: center;'>
        <img src='https://cdn-icons-png.flaticon.com/512/1046/1046784.png' width='100'/>
        <p style='font-size: 14px; color: grey;'>Model powered by LightGBM & Scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)
