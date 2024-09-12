import streamlit as st
import pandas as pd
import requests

# FastAPI endpoint URL
api_url = "http://127.0.0.1:8000/predict"

# Streamlit UI
st.title("Heart Disease Prediction")

# Input form for single prediction
st.header("Single Prediction")

with st.form(key="single_prediction_form"):
    Age = st.number_input("Enter Age:", min_value=1, max_value=120, value=30)
    Sex = st.selectbox("Enter Sex:", options=["M", "F"])
    ChestPainType = st.selectbox("Enter Chest Pain Type:", options=["TA", "ATA", "NAP", "ASY"])
    RestingECG = st.selectbox("Enter Resting ECG:", options=["Normal", "ST", "LVH"])
    MaxHR = st.number_input("Enter MaxHR:", min_value=60, max_value=210, value=150)
    ExerciseAngina = st.selectbox("Enter Exercise Angina:", options=["Y", "N"])
    Oldpeak = st.number_input("Enter Oldpeak:", min_value=0.0, max_value=10.0, value=1.0)
    ST_Slope = st.selectbox("Enter ST Slope:", options=["Up", "Flat", "Down"])

    submit_button = st.form_submit_button("Predict")

    if submit_button:
        # Prepare the input data in the required format
        input_data = {
            "Age": Age,
            "Sex": Sex,
            "ChestPainType": ChestPainType,
            "RestingECG": RestingECG,
            "MaxHR": MaxHR,
            "ExerciseAngina": ExerciseAngina,
            "Oldpeak": Oldpeak,
            "ST_Slope": ST_Slope,
        }

        # Send the request to the API for prediction
        try:
            response = requests.post(api_url, json=input_data)  # Ensure POST request is used

            if response.status_code == 200:
                prediction = response.json()["prediction"]
                st.success(f"The model predicts: {'Heart Disease Detected' if prediction == 1 else 'No Heart Disease'}")
            else:
                st.error(f"Error during prediction: {response.json()['detail']}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
