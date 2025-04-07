
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Streamlit page setup
st.set_page_config(page_title="Real-Time AQI Predictor", page_icon="🌫️", layout="wide")

# Stylish title
st.markdown("<h1 style='text-align: center; color: #0066cc;'>🌫️ Real-Time Air Quality Index (AQI) Predictor</h1>", unsafe_allow_html=True)

# File uploader
st.sidebar.header("📁 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your air quality dataset (CSV)", type=["csv"])

if uploaded_file:
    # Load and preprocess
    df = pd.read_csv(uploaded_file)
    df = df.dropna()

    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop(columns=["AQI"])
    y = df["AQI"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # UI: Input section
    st.subheader("🔧 Customize Input Features")
    cols = st.columns(2)
    user_input = []
    for idx, col_name in enumerate(X.columns):
        with cols[idx % 2]:
            val = st.number_input(f"{col_name}", value=float(np.mean(df[col_name])))
            user_input.append(val)

    # Prediction
    if st.button("🎯 Predict AQI"):
        input_scaled = scaler.transform([user_input])
        prediction = model.predict(input_scaled)[0]

        # Determine AQI category
        if prediction <= 50:
            category, color = "Good", "🟢"
        elif prediction <= 100:
            category, color = "Moderate", "🟡"
        elif prediction <= 150:
            category, color = "Unhealthy for Sensitive Groups", "🟠"
        elif prediction <= 200:
            category, color = "Unhealthy", "🔴"
        elif prediction <= 300:
            category, color = "Very Unhealthy", "🟣"
        else:
            category, color = "Hazardous", "⚫"

        st.markdown(f"""
        <div style="border:2px solid #ddd; padding:20px; border-radius:10px; background-color:#f9f9f9;">
            <h2 style="color:#333;">🎯 Predicted AQI: <span style="color:#0066cc;">{prediction:.2f}</span></h2>
            <h4 style="color:#555;">{color} Category: <b>{category}</b></h4>
        </div>
        """, unsafe_allow_html=True)

    # Optional: Show dataset preview
    with st.expander("🔍 Preview Dataset"):
        st.dataframe(df.head())

# Footer separator
st.markdown("<hr>", unsafe_allow_html=True)