
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# # Streamlit page setup
# st.set_page_config(page_title="Real-Time AQI Predictor", page_icon="🌫️", layout="wide")

# # Stylish title
# st.markdown("<h1 style='text-align: center; color: #0066cc;'>🌫️ Real-Time Air Quality Index (AQI) Predictor</h1>", unsafe_allow_html=True)

# # File uploader
# st.sidebar.header("📁 Upload Your Dataset")
# uploaded_file = st.sidebar.file_uploader("Upload your air quality dataset (CSV)", type=["csv"])

# if uploaded_file:
#     # Load and preprocess
#     df = pd.read_csv(uploaded_file)
#     df = df.dropna()

#     label_encoders = {}
#     for col in df.select_dtypes(include=["object"]).columns:
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col])
#         label_encoders[col] = le

#     X = df.drop(columns=["AQI"])
#     y = df["AQI"]

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#     model = RandomForestRegressor()
#     model.fit(X_train, y_train)

#     # UI: Input section
#     st.subheader("🔧 Customize Input Features")
#     cols = st.columns(2)
#     user_input = []
#     for idx, col_name in enumerate(X.columns):
#         with cols[idx % 2]:
#             val = st.number_input(f"{col_name}", value=float(np.mean(df[col_name])))
#             user_input.append(val)

#     # Prediction
#     if st.button("🎯 Predict AQI"):
#         input_scaled = scaler.transform([user_input])
#         prediction = model.predict(input_scaled)[0]

#         # Determine AQI category
#         if prediction <= 50:
#             category, color = "Good", "🟢"
#         elif prediction <= 100:
#             category, color = "Moderate", "🟡"
#         elif prediction <= 150:
#             category, color = "Unhealthy for Sensitive Groups", "🟠"
#         elif prediction <= 200:
#             category, color = "Unhealthy", "🔴"
#         elif prediction <= 300:
#             category, color = "Very Unhealthy", "🟣"
#         else:
#             category, color = "Hazardous", "⚫"

#         st.markdown(f"""
#         <div style="border:2px solid #ddd; padding:20px; border-radius:10px; background-color:#f9f9f9;">
#             <h2 style="color:#333;">🎯 Predicted AQI: <span style="color:#0066cc;">{prediction:.2f}</span></h2>
#             <h4 style="color:#555;">{color} Category: <b>{category}</b></h4>
#         </div>
#         """, unsafe_allow_html=True)

#     # Optional: Show dataset preview
#     with st.expander("🔍 Preview Dataset"):
#         st.dataframe(df.head())

# # Footer separator
# st.markdown("<hr>", unsafe_allow_html=True)



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

# Sidebar: Choose input method
st.sidebar.header("⚙️ Select Input Method")
mode = st.sidebar.radio("Choose mode:", ["Upload Dataset", "Manual Entry"])

# Sidebar: File uploader (for Upload Dataset mode)
uploaded_file = st.sidebar.file_uploader("📁 Upload your air quality dataset (CSV)", type=["csv"])

# Placeholder for model and feature names
model = None
scaler = None
feature_names = []

# Function to determine AQI category
def get_aqi_category(prediction):
    if prediction <= 50:
        return "Good", "🟢"
    elif prediction <= 100:
        return "Moderate", "🟡"
    elif prediction <= 150:
        return "Unhealthy for Sensitive Groups", "🟠"
    elif prediction <= 200:
        return "Unhealthy", "🔴"
    elif prediction <= 300:
        return "Very Unhealthy", "🟣"
    else:
        return "Hazardous", "⚫"

# If dataset is uploaded
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
    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    st.success("✅ Model trained successfully using uploaded dataset.")

    # Optional: Show dataset preview
    with st.expander("🔍 Preview Dataset"):
        st.dataframe(df.head())

# ==== MODE 1: Upload Dataset Prediction ====
if mode == "Upload Dataset" and model:
    st.subheader("🔧 Customize Input Features (from your dataset)")
    cols = st.columns(2)
    user_input = []
    for idx, col_name in enumerate(feature_names):
        with cols[idx % 2]:
            val = st.number_input(f"{col_name}", value=float(np.mean(df[col_name])))
            user_input.append(val)

    if st.button("🎯 Predict AQI"):
        input_scaled = scaler.transform([user_input])
        prediction = model.predict(input_scaled)[0]
        category, color = get_aqi_category(prediction)

        st.markdown(f"""
        <div style="border:2px solid #ddd; padding:20px; border-radius:10px; background-color:#f9f9f9;">
            <h2 style="color:#333;">🎯 Predicted AQI: <span style="color:#0066cc;">{prediction:.2f}</span></h2>
            <h4 style="color:#555;">{color} Category: <b>{category}</b></h4>
        </div>
        """, unsafe_allow_html=True)

# ==== MODE 2: Manual Pollutant Entry ====
elif mode == "Manual Entry":
    st.subheader("✍️ Manually Enter Pollutant Values")
    
    # Define typical pollutant features
    default_pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]
    manual_input = []
    col2 = st.columns(2)
    for i, pollutant in enumerate(default_pollutants):
        with col2[i % 2]:
            val = st.number_input(f"{pollutant} level (µg/m³)", min_value=0.0, value=50.0)
            manual_input.append(val)

    # Dummy model for manual input (if no dataset)
    if st.button("🎯 Predict AQI (Manual Mode)"):
        dummy_model = RandomForestRegressor()
        dummy_X = np.random.rand(100, len(default_pollutants)) * 100
        dummy_y = dummy_X @ np.random.rand(len(default_pollutants)) + np.random.rand(100) * 50
        dummy_model.fit(dummy_X, dummy_y)

        prediction = dummy_model.predict([manual_input])[0]
        category, color = get_aqi_category(prediction)

        st.markdown(f"""
        <div style="border:2px solid #ddd; padding:20px; border-radius:10px; background-color:#f9f9f9;">
            <h2 style="color:#333;">🎯 Predicted AQI: <span style="color:#0066cc;">{prediction:.2f}</span></h2>
            <h4 style="color:#555;">{color} Category: <b>{category}</b></h4>
        </div>
        """, unsafe_allow_html=True)

# Footer separator
st.markdown("<hr>", unsafe_allow_html=True)