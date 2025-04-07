import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.title("🌫️ Real-Time Air Quality Index Predictor")

uploaded_file = st.file_uploader("Upload your air quality dataset (CSV)", type=["csv"])

if uploaded_file:
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

    st.subheader("📥 Input Features")
    user_input = []
    for col in X.columns:
        val = st.number_input(f"{col}", value=float(np.mean(df[col])))
        user_input.append(val)

    if st.button("Predict AQI"):
        input_scaled = scaler.transform([user_input])
        prediction = model.predict(input_scaled)[0]
        st.success(f"🎯 Predicted AQI: {prediction:.2f}")
