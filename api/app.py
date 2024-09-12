from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load all the model components
model = joblib.load(r"C:/Users/SOHAM/Git_Repositories/DataScience_Projects/dsp-heart-failure-prediction/models/log_model.joblib")
one_hot_encoder = joblib.load(r"C:/Users/SOHAM/Git_Repositories/DataScience_Projects/dsp-heart-failure-prediction/models/one_hot_encoder_HD.joblib")
ordinal_encoder = joblib.load(r"C:/Users/SOHAM/Git_Repositories/DataScience_Projects/dsp-heart-failure-prediction/models/ordinal_encoder_HD.joblib")
scaler = joblib.load(r"C:/Users/SOHAM/Git_Repositories/DataScience_Projects/dsp-heart-failure-prediction/models/scaler_HD.joblib")

app = FastAPI()


# Define the request body structure
class HeartDiseasePredictionRequest(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

# Define preprocessing function
def preprocess_input(data: dict):
    # Convert the dictionary into a dataframe
    df = pd.DataFrame([data])

    # Apply the necessary encodings (ensure the column names are correct)
    df['Sex'] = ordinal_encoder.transform(df[['Sex']])
    df[['ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']] = one_hot_encoder.transform(df[['ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']])
    
    # Scale the numerical columns
    df[['Age', 'MaxHR', 'Oldpeak']] = scaler.transform(df[['Age', 'MaxHR', 'Oldpeak']])

    return df

@app.post("/predict")
async def predict(request_data: HeartDiseasePredictionRequest):
    try:
        # Preprocess the input
        input_data = preprocess_input(request_data.dict())

        # Make the prediction
        prediction = model.predict(input_data)

        # Return the prediction
        return {"prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")