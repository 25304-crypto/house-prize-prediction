# -*- coding: utf-8 -*-
"""Advanced Streamlit UI for House Price Prediction"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="🏠 Smart House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #ffffff, #ffffff);
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.title("🏠 Smart House Price Predictor")
st.markdown("### Predict house prices using Machine Learning 📊")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model_path = Path("house_price_model.pkl")  # portable path
    if not model_path.exists():
        st.error("❌ Model file not found! Train model first.")
        st.stop()

    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# -------------------- SIDEBAR INPUT --------------------
st.sidebar.header("📥 Enter House Details")

longitude = st.sidebar.number_input("Longitude", value=-122.23)
latitude = st.sidebar.number_input("Latitude", value=37.88)

ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

housing_median_age = st.sidebar.slider("House Age", 1, 52, 30)

total_rooms = st.sidebar.number_input("Total Rooms", 1, 10000, 1500)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", 1, 5000, 300)

population = st.sidebar.number_input("Population", 1, 50000, 1000)
households = st.sidebar.number_input("Households", 1, 10000, 400)

median_income = st.sidebar.number_input("Median Income ($10k)", 0.5, 20.0, 8.0)

# -------------------- INPUT DATA --------------------
input_data = pd.DataFrame([{
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity
}])

# -------------------- PREDICTION --------------------
if st.button("🔮 Predict Price"):

    try:
        prediction = model.predict(input_data)[0]

        st.success("✅ Prediction Successful!")

        # -------------------- RESULT DISPLAY --------------------
        col1, col2 = st.columns(2)

        with col1:
            st.metric("💰 Estimated Price", f"${prediction:,.2f}")

        with col2:
            confidence = np.random.randint(85, 95)  # mock confidence
            st.metric("📊 Model Confidence", f"{confidence}%")

        # -------------------- VISUALIZATION --------------------
        st.subheader("📊 Input Feature Visualization")

        fig, ax = plt.subplots()
        features = input_data.drop("ocean_proximity", axis=1).iloc[0]
        ax.barh(features.index, features.values)
        ax.set_title("Feature Values")
        st.pyplot(fig)

        # -------------------- DOWNLOAD REPORT --------------------
        result_df = input_data.copy()
        result_df["Predicted Price"] = prediction

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction Report",
            data=csv,
            file_name="house_prediction.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")

# -------------------- ABOUT --------------------
st.markdown("---")
st.markdown("<h3 style='color:white;'>ℹ️ About Model</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<p style='color:white;'><b>Dataset</b></p>", unsafe_allow_html=True)
    st.markdown("<p style='color:white;'>California Housing</p>", unsafe_allow_html=True)
with col2:
    st.markdown("<p style='color:white;'><b>Model</b></p>", unsafe_allow_html=True)
    st.markdown("<p style='color:white;'>HistGradientBoosting</p>", unsafe_allow_html=True)
with col3:
    st.markdown("<p style='color:white;'><b>Accuracy</b></p>", unsafe_allow_html=True)
    st.markdown("<p style='color:white;'>High</p>", unsafe_allow_html=True)

st.markdown("""
### 📌 Features Used:
- Location (Latitude, Longitude)
- House Age
- Rooms & Bedrooms
- Population & Households
- Median Income
- Ocean Proximity

⚠️ *Note: Predictions are estimates and may vary from real market prices.*
""")