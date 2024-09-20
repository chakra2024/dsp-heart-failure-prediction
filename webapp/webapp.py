import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# FastAPI endpoint URL
api_url = "http://127.0.0.1:8000"

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Past Predictions"])

if page == "Prediction":
    st.title("Heart Failure Prediction")
    
    # Select Prediction Mode (Single/Batch)
    prediction_mode = st.radio("Choose Prediction Mode", ("Single Prediction", "Batch Prediction (CSV)"))

    if prediction_mode == "Single Prediction":
        # Input form for single prediction
        with st.form(key="single_prediction_form"):
            Age = st.number_input("Enter Age:", min_value=1, max_value=120, value=30)
            Sex = st.selectbox("Enter Sex:", options=["M", "F"])
            ChestPainType = st.selectbox("Enter Chest Pain Type:", options=["TA", "ATA", "NAP", "ASY"])
            RestingBP = st.number_input("Enter RestingBP:", min_value=60, max_value=210, value=150)
            Cholesterol = st.number_input("Enter Cholesterol:", min_value=90, max_value=500, value=250)
            MaxHR = st.number_input("Enter MaxHR:", min_value=60, max_value=210, value=150)
            ExerciseAngina = st.selectbox("Enter Exercise Angina:", options=["Y", "N"])
            Oldpeak = st.number_input("Enter Oldpeak:", min_value=0.0, max_value=10.0, value=1.0)
            ST_Slope = st.selectbox("Enter ST Slope:", options=["Up", "Flat", "Down"])

            submit_button = st.form_submit_button("Predict")

            if submit_button:
                # Prepare the input data
                input_data = {
                    "Age": Age,
                    "Sex": Sex,
                    "ChestPainType": ChestPainType,
                    "RestingBP": RestingBP,
                    "Cholesterol": Cholesterol,
                    "MaxHR": MaxHR,
                    "ExerciseAngina": ExerciseAngina,
                    "Oldpeak": Oldpeak,
                    "ST_Slope": ST_Slope,
                }

                # Send the request to the API for single prediction
                try:
                    response = requests.post(api_url + "/predict", json=input_data)
                    if response.status_code == 200:
                        # Extract the response
                        prediction_with_features = response.json()["prediction_with_features"]

                        # Convert to DataFrame for display
                        result_df = pd.DataFrame([prediction_with_features])

                        # Display the result as a DataFrame
                        st.success("Prediction with Features:")
                        st.write(result_df)
                        
                        # Display the prediction message
                        st.success(f"Prediction: {'Heart Disease Detected' if prediction_with_features['Prediction'] == 1 else 'No Heart Disease'}")
                    else:
                        st.error(f"Error during prediction: {response.json()['detail']}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


    elif prediction_mode == "Batch Prediction (CSV)":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.write("Uploaded Data:")
                st.write(input_df.head())

                if st.button("Predict for CSV"):
                    # Convert the DataFrame to a list of dictionaries
                    input_data = input_df.to_dict(orient='records')

                    # Send the request to the API for batch prediction
                    try:
                        response = requests.post(f"{api_url}/predict_batch", json=input_data)
                        if response.status_code == 200:
                            # Get the predictions along with the input features
                            predictions_with_features = response.json()["predictions_with_features"]
                            
                            # Convert back to a DataFrame for display
                            result_df = pd.DataFrame(predictions_with_features)
                            
                            # Display the results as a DataFrame
                            st.success("Predictions with Features:")
                            st.write(result_df)
                        else:
                            st.error(f"Error during prediction: {response.json()['detail']}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            except Exception as e:
                st.error(f"Error processing the file: {str(e)}")


elif page == "Past Predictions":
    st.title("Past Predictions")

    # Date selection components (Use simpler date format '%Y-%m-%d')
    start_date = st.sidebar.date_input("Start date", None)
    end_date = st.sidebar.date_input("End date", None)

    # Prediction source dropdown
    source_options = ["all", "webapp", "scheduled"]
    source = st.sidebar.selectbox("Prediction source", options=source_options)

    # Convert date inputs to strings (using simplified format)
    start_date_str = start_date.strftime('%Y-%m-%d') if start_date else None
    end_date_str = end_date.strftime('%Y-%m-%d') if end_date else None

    if st.button("Fetch Past Predictions"):
        params = {}

        if start_date_str:
            params['start_date'] = start_date_str
        if end_date_str:
            params['end_date'] = end_date_str
        if source != "all":
            params['source'] = source

        try:
            # Fetch past predictions from API
            response = requests.get(api_url + "/past_predictions", params=params)
            if response.status_code == 200:
                # Get the past predictions along with features
                past_predictions = response.json()["predictions"]
                
                if past_predictions:
                    # Convert to DataFrame for display
                    past_predictions_df = pd.DataFrame(past_predictions)
                    
                    # Display the past predictions as a DataFrame
                    st.write("Past Predictions:")
                    st.write(past_predictions_df)
                else:
                    st.info("No predictions found for the given filters.")
            else:
                st.error("Error fetching past predictions.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
