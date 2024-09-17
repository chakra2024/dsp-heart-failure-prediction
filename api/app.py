from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
import psycopg2
from psycopg2 import sql

# Establish database connection
def connect_to_db():
    try:
        conn = psycopg2.connect(
            dbname="dsphealth",
            user="postgres",
            password="postgres",
            host="localhost",  # If hosted elsewhere, change this to the correct server address
            port="5432"
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

# Function to insert prediction result into the database
def insert_prediction(input_data, prediction):
    conn = connect_to_db()
    if conn is None:
        print("Error: Could not connect to database.")
        return
    
    cursor = conn.cursor()
    
    try:
        # Insert the input features
        cursor.execute(
            """
            INSERT INTO features (age, sex, chest_pain_type, resting_bp, cholesterol, max_hr, exercise_angina, oldpeak, st_slope)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (input_data['Age'], input_data['Sex'], input_data['ChestPainType'], input_data['RestingBP'], 
             input_data['Cholesterol'], input_data['MaxHR'], input_data['ExerciseAngina'], input_data['Oldpeak'], input_data['ST_Slope'])
        )
        
        # Get the inserted feature's ID
        cursor.execute("SELECT id FROM features ORDER BY id DESC LIMIT 1")
        feature_id = cursor.fetchone()[0]
        
        # Insert the prediction result using the feature ID
        cursor.execute(
            "INSERT INTO predictions (id, prediction) VALUES (%s, %s)", (feature_id, prediction)
        )
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error inserting prediction into database: {e}")
    finally:
        cursor.close()
        conn.close()

# Load all the model components
model_path = r"C:\Users\SOHAM\Git_Repositories\DataScience_Projects\dsp-heart-failure-prediction\models\model.joblib"
preprocessor_path = r"C:\Users\SOHAM\Git_Repositories\DataScience_Projects\dsp-heart-failure-prediction\models\preprocessors.joblib"

# Load preprocessor and model for prediction
preprocessor = joblib.load(preprocessor_path)
model = joblib.load(model_path)

app = FastAPI()

origins = [
    "http://localhost",
    "https://localhost",
    "http://localhost:8080",
    "http://localhost:8501"
]

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define the request body structure
class HeartDiseasePredictionRequest(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

# Root endpoint to show information
@app.get("/")
def read_root():
    return {"message": "Welcome to the Heart Failure Prediction API. Go to /predict to make a prediction."}

# Define preprocessing function
def preprocess_input(data: dict):
    # Convert the dictionary into a dataframe
    df = pd.DataFrame([data])
    # Preprocess data using existing joblib
    processed_data = preprocessor.transform(df)
    return processed_data

# Predict endpoint for single prediction
@app.post("/predict")
async def predict(request_data: HeartDiseasePredictionRequest):
    try:
        # Preprocess the input
        input_data = preprocess_input(request_data.dict())
        
        # Make the prediction
        prediction = model.predict(input_data)[0]  # Get the first value from the prediction array
        
        # Insert the prediction and input into the database
        insert_prediction(request_data.dict(), int(prediction))

        # Add the prediction to the input data
        result = request_data.dict()
        result["Prediction"] = int(prediction)
        
        # Return the input features along with the prediction
        return {"prediction_with_features": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    
    
# Predict endpoint for batch prediction
@app.post("/predict_batch")
def predict_batch_api(input_data: List[HeartDiseasePredictionRequest]):
    try:
        # Convert input data to a DataFrame
        df = pd.DataFrame([item.dict() for item in input_data])
        
        # Preprocess the input data
        processed_data = preprocessor.transform(df)
        
        # Get predictions
        predictions = model.predict(processed_data).tolist()

        # Insert each prediction and input into the database
        for i in range(len(input_data)):
            insert_prediction(input_data[i].dict(), predictions[i])

        # Add predictions to the original DataFrame
        df['Prediction'] = predictions

        # Return the features along with the predictions
        return {"predictions_with_features": df.to_dict(orient='records')}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Fetch past predictions
@app.get("/past_predictions")
def fetch_past_predictions():
    try:
        conn = connect_to_db()
        if conn is None:
            raise HTTPException(status_code=500, detail="Database connection failed.")

        cursor = conn.cursor()
        
        # Join features and predictions to display past predictions with their features
        cursor.execute("""
            SELECT f.age, f.sex, f.chest_pain_type, f.resting_bp, f.cholesterol, 
                   f.max_hr, f.exercise_angina, f.oldpeak, f.st_slope, p.prediction 
            FROM features f 
            JOIN predictions p ON f.id = p.id
        """)
        
        records = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Structure the output data
        results = []
        for record in records:
            result = {
                "Age": record[0],
                "Sex": record[1],
                "ChestPainType": record[2],
                "RestingBP": record[3],
                "Cholesterol": record[4],
                "MaxHR": record[5],
                "ExerciseAngina": record[6],
                "Oldpeak": record[7],
                "ST_Slope": record[8],
                "Prediction": "Risk of heart failure" if record[9] == 1 else "No risk of heart failure"
            }
            results.append(result)
        
        return {"predictions": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching past predictions: {str(e)}")
    
# Fetch Past Predictions    
@app.get("/past_predictions")
def fetch_past_predictions():
    try:
        conn = connect_to_db()
        if conn is None:
            raise HTTPException(status_code=500, detail="Database connection failed.")

        cursor = conn.cursor()
        
        # Join features and predictions to display past predictions with their features
        cursor.execute("""
            SELECT f.age, f.sex, f.chest_pain_type, f.resting_bp, f.cholesterol, 
                   f.max_hr, f.exercise_angina, f.oldpeak, f.st_slope, p.prediction 
            FROM features f 
            JOIN predictions p ON f.id = p.id
        """)
        
        records = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Structure the output data
        results = []
        for record in records:
            result = {
                "Age": record[0],
                "Sex": record[1],
                "ChestPainType": record[2],
                "RestingBP": record[3],
                "Cholesterol": record[4],
                "MaxHR": record[5],
                "ExerciseAngina": record[6],
                "Oldpeak": record[7],
                "ST_Slope": record[8],
                "Prediction": "Risk of heart failure" if record[9] == 1 else "No risk of heart failure"
            }
            results.append(result)
        
        return {"predictions": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching past predictions: {str(e)}")
