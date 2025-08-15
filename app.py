import pandas as pd
import joblib
import os
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Load model (relative path for deployment)
MODEL_PATH = "Car_Price_Prediction.pkl"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)

# Paths
ENCODERS_DIR = "encoders"

# List of categorical columns
categorical_cols = [
    'name', 'make', 'model', 'city',
    'fueltype', 'transmission', 'bodytype',
    'registrationcity', 'registrationstate'
]

# Load encoders
encoders = {}
for col in categorical_cols:
    classes_path = os.path.join(ENCODERS_DIR, f"{col}_classes.pkl")
    if not os.path.exists(classes_path):
        st.error(f"Encoder file missing for {col}: {classes_path}")
        st.stop()
    encoders[col] = joblib.load(classes_path)

st.title("🚗 Car Price Prediction with XGBoost")

# User Inputs
st.header("Enter Car Details")
user_data = {}
for col in categorical_cols:
    user_data[col] = st.selectbox(f"Select {col.capitalize()}", encoders[col])

user_data['year'] = st.number_input("Manufacturing Year", min_value=1980, max_value=2025, value=2020)
user_data['price'] = st.number_input("Price", min_value=0.0, value=500000.0, step=1000.0)
user_data['kilometerdriven'] = st.number_input("Kilometers Driven", min_value=1.0, value=50000.0, step=1000.0)  # avoid /0
user_data['ownernumber'] = st.number_input("Owner Number", min_value=1, max_value=10, value=1)

# Convert to DataFrame
df_input = pd.DataFrame([user_data])

# Feature Engineering
current_year = datetime.now().year
df_input['car_age'] = current_year - df_input['year']
df_input['price_per_km'] = df_input['price'] / df_input['kilometerdriven']
df_input['is_1st_owner'] = (df_input['ownernumber'] == 1).astype(int)
df_input['is_automatic'] = (df_input['transmission'] == 'Automatic').astype(int)

# Keep only training columns
df_input = df_input[[
    'name', 'make', 'model', 'city', 'fueltype', 'transmission',
    'bodytype', 'registrationcity', 'registrationstate',
    'car_age', 'price_per_km', 'is_1st_owner', 'is_automatic'
]]

# Label Encoding
for col in categorical_cols:
    le = LabelEncoder()
    le.classes_ = encoders[col]
    df_input[col] = le.transform(df_input[col].astype(str))

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(df_input)
    st.success(f"Predicted Price: ₹{prediction[0]:,.2f}")
