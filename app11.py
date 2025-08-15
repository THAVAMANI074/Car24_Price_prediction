import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os

# ===== Load Model =====
MODEL_PATH = r"C:\Users\Thavamani\Downloads\Car 24 Prediction\Project\Car_Price_Prediction.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
else:
    model = joblib.load(MODEL_PATH)

# ===== Categorical column options =====
categorical_choices = {
    'name': ['Toyota', 'Honda', 'Hyundai', 'Ford'],
    'make': ['Toyota', 'Honda', 'Hyundai', 'Ford'],
    'model': ['Corolla', 'Civic', 'Creta', 'EcoSport'],
    'city': ['Delhi', 'Mumbai', 'Chennai', 'Bangalore'],
    'fueltype': ['Petrol', 'Diesel', 'CNG', 'Electric'],
    'transmission': ['Manual', 'Automatic'],
    'bodytype': ['Hatchback', 'Sedan', 'SUV'],
    'registrationcity': ['Delhi', 'Mumbai', 'Chennai', 'Bangalore'],
    'registrationstate': ['DL', 'MH', 'TN', 'KA']
}

# ===== Streamlit UI =====
st.title("🚗 Car Price Prediction with XGBoost")
st.header("Enter Car Details")

user_data = {}
for col, choices in categorical_choices.items():
    user_data[col] = st.selectbox(f"Select {col.capitalize()}", choices)

user_data['year'] = st.number_input("Manufacturing Year", min_value=1980, max_value=2025, value=2020)
user_data['price'] = st.number_input("Price (₹)", min_value=0.0, value=500000.0, step=1000.0)
user_data['kilometerdriven'] = st.number_input("Kilometers Driven", min_value=0.0, value=50000.0, step=1000.0)
user_data['ownernumber'] = st.number_input("Owner Number", min_value=1, max_value=10, value=1)

# ===== Data Preparation =====
df_input = pd.DataFrame([user_data])

current_year = datetime.now().year
df_input['car_age'] = current_year - df_input['year']
df_input['price_per_km'] = df_input['price'] / df_input['kilometerdriven']
df_input['is_1st_owner'] = (df_input['ownernumber'] == 1).astype(int)
df_input['is_automatic'] = (df_input['transmission'] == 'Automatic').astype(int)

df_input = df_input[['name', 'make', 'model', 'city', 'fueltype', 'transmission',
                     'bodytype', 'registrationcity', 'registrationstate',
                     'car_age', 'price_per_km', 'is_1st_owner', 'is_automatic']]

# ===== Encode categorical variables =====
for col in categorical_choices.keys():
    le = LabelEncoder()
    le.fit(categorical_choices[col])
    df_input[col] = le.transform(df_input[col])

# ===== Prediction =====
if st.button("Predict Price"):
    try:
        prediction = model.predict(df_input)
        st.success(f"Predicted Price: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
