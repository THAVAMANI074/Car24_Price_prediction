import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import os

# ====== Model path ======
MODEL_PATH = r"C:\Users\Thavamani\Desktop\ExcelR-Project1\car_price_predictor\best_price_classifier_xgb_10bins.joblib"

# ====== Load Model ======
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"Model not found at: {path}")
        return None

model = load_model(MODEL_PATH)

# ====== Fallback: Upload if missing ======
if model is None:
    uploaded_file = st.file_uploader("Upload your trained model (.joblib)", type=["joblib"])
    if uploaded_file is not None:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Model uploaded successfully! Saved at:\n{MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        st.experimental_rerun()
    st.stop()

# ====== Streamlit UI ======
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")
st.title("🚗 Car Price Prediction App")
st.write("Enter the details below to predict the price category of a car.")

# ====== Mappings ======
fuel_map = {'Petrol': 1, 'Petrol + CNG': 2, 'Diesel': 3}
transmission_map = {'Manual': 1, 'Automatic': 2, 'Unknown': 0}

# ====== User Inputs ======
kilometerdriven = st.number_input("Kilometers Driven", min_value=0, step=100)
ownernumber = st.selectbox("Owner Number", [1, 2, 3, 4])
fueltype = st.selectbox("Fuel Type", list(fuel_map.keys()))
transmission = st.selectbox("Transmission Type", list(transmission_map.keys()))
isc24assured = st.selectbox("C24 Assured", ["No", "Yes"])
age = st.number_input("Car Age (Years)", min_value=0, max_value=30, step=1)

# ====== Convert Inputs ======
fueltype_num = fuel_map[fueltype]
transmission_num = transmission_map[transmission]
isc24assured_num = 1 if isc24assured == "Yes" else 0

# ====== Prepare Input DataFrame ======
input_data = pd.DataFrame([{
    'kilometerdriven': kilometerdriven,
    'ownernumber': ownernumber,
    'fueltype_num': fueltype_num,
    'transmission_num': transmission_num,
    'isc24assured_num': isc24assured_num,
    'age': age
}])

# ====== Feature Engineering ======
input_data['km_per_age'] = input_data['kilometerdriven'] / (input_data['age'] + 1)
input_data['log_km'] = np.log1p(input_data['kilometerdriven'])
input_data['log_age'] = np.log1p(input_data['age'])
input_data['age_squared'] = input_data['age'] ** 2
input_data['km_age_interaction'] = input_data['kilometerdriven'] * input_data['age']
input_data['price_per_km'] = 0  # Placeholder

# Ensure columns match model
input_data = input_data[model.feature_names_in_]

# ====== Price Bins ======
bins = [
    "₹0–₹50,000", "₹50,001–₹150,000", "₹150,001–₹250,000", "₹250,001–₹350,000",
    "₹350,001–₹450,000", "₹450,001–₹600,000", "₹600,001–₹900,000",
    "₹900,001–₹1,200,000", "₹1,200,001–₹1,800,000", "₹1,800,001–₹4,000,000"
]

# ====== Prediction ======
if st.button("Predict Price Category"):
    pred_bin = model.predict(input_data)[0]
    pred_probs = model.predict_proba(input_data)[0]

    # Weighted average
    bin_midpoints = np.array([25000, 100000, 200000, 300000, 400000, 525000, 750000, 1050000, 1500000, 2900000])
    estimated_price = np.sum(bin_midpoints * pred_probs)

    st.subheader("✅ Prediction Result")
    st.write(f"**Predicted Bin:** {pred_bin} ({bins[pred_bin]})")
    st.write(f"**Estimated Price (Weighted Average):** ₹{estimated_price:,.2f}")

    st.subheader("📊 Prediction Probabilities by Bin")
    prob_df = pd.DataFrame({
        'Bin': bins,
        'Probability (%)': np.round(pred_probs * 100, 3)
    })
    st.dataframe(prob_df)

    # ====== Prepare CSV ======
    output_df = input_data.copy()
    output_df['Predicted_Bin'] = pred_bin
    output_df['Estimated_Price'] = estimated_price
    for i, b in enumerate(bins):
        output_df[f'Prob_{b}'] = pred_probs[i]

    csv_buffer = io.StringIO()
    output_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    st.download_button(
        label="📥 Download Prediction as CSV",
        data=csv_data,
        file_name="car_price_prediction.csv",
        mime="text/csv"
    )
