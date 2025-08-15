import pandas as pd
import joblib
import os
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# ===== Paths =====
MODEL_PATH = "Car_Price_Prediction.pkl"  # Ensure this file is in the same folder as app.py
ENCODERS_DIR = "encoders"  # Ensure this folder exists and contains all encoder files

# ===== Load Model =====
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found: {MODEL_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)

# ===== Categorical Columns =====
categorical_cols = [
    'name', 'make', 'model', 'city', 'fueltype',
    'transmission', 'bodytype', 'registrationcity', 'registrationstate'
]

# ===== Load Encoders =====
encoders = {}
missing_files = []
for col in categorical_cols:
    classes_path = os.path.join(ENCODERS_DIR, f"{col}_classes.pkl")
    if not os.path.exists(classes_path):
        missing_files.append(classes_path)
    else:
        encoders[col] = joblib.load(classes_path)

if missing_files:
    st.error("❌ Missing encoder files:\n" + "\n".join(missing_files))
    st.stop()

# ===== Streamlit UI =====
st.title("🚗 Car Price Prediction App")
st.markdown("Enter the car details to get the predicted price.")

# ===== User Inputs =====
user_data = {}
for col in categorical_cols:
    user_data[col] = st.selectbox(f"Select {col.capitalize()}", encoders[col])

user_data['year'] = st.number_input("Manufacturing Year", min_value=1980, max_value=2025, value=2020)
user_data['price'] = st.number_input("Price (₹)", min_value=0.0, value=500000.0, step=1000.0)
user_data['kilometerdriven'] = st.number_input("Kilometers Driven", min_value=0.0, value=50000.0, step=1000.0)
user_data['ownernumber'] = st.number_input("Owner Number", min_value=1, max_value=10, value=1)

# ===== Convert to DataFrame =====
df_input = pd.DataFrame([user_data])

# ===== Feature Engineering =====
current_year = datetime.now().year
df_input['car_age'] = current_year - df_input['year']
df_input['price_per_km'] = df_input['price'] / df_input['kilometerdriven'] if df_input['kilometerdriven'][0] > 0 else 0
df_input['is_1st_owner'] = (df_input['ownernumber'] == 1).astype(int)
df_input['is_automatic'] = (df_input['transmission'] == 'Automatic').astype(int)

# Keep only training features
df_input = df_input[
    ['name', 'make', 'model', 'city', 'fueltype', 'transmission',
     'bodytype', 'registrationcity', 'registrationstate',
     'car_age', 'price_per_km', 'is_1st_owner', 'is_automatic']
]

# ===== Encode categorical features =====
for col in categorical_cols:
    le = LabelEncoder()
    le.classes_ = encoders[col]
    df_input[col] = le.transform(df_input[col].astype(str))

# ===== Prediction =====
if st.button("Predict Price"):
    prediction = model.predict(df_input)
    st.success(f"💰 Predicted Price: ₹{prediction[0]:,.2f}")
