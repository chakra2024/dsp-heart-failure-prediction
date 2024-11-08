from typing import List, Union
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from psycopg2 import sql
from datetime import datetime
from fastapi import Request

# Establish database connection
def connect_to_db():
    try:
        conn = psycopg2.connect(
            dbname="dsphealth",
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432"
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


# Function to insert multiple predictions into the database
def batch_insert_predictions(input_data_list, predictions, source):
    conn = connect_to_db()
    if conn is None:
        print("Error: Could not connect to database.")
        return

    cursor = conn.cursor()

    try:

        # Prepare data for batch insertion
        features_data = [
            (
                data['Age'], data['Sex'], data['ChestPainType'], data['RestingBP'], 
                data['Cholesterol'], data['MaxHR'], data['ExerciseAngina'], data['Oldpeak'], data['ST_Slope']
            ) 
            for data in input_data_list
        ]

        cursor.executemany(

            """
            INSERT INTO features (age, sex, chest_pain_type, resting_bp, cholesterol, max_hr, exercise_angina, oldpeak, st_slope)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            , features_data
        )

        # Retrieve feature IDs for recently inserted records
        cursor.execute("SELECT id FROM features ORDER BY id DESC LIMIT %s", (len(input_data_list),))
        feature_ids = [row[0] for row in cursor.fetchall()]

        # Prepare prediction data for batch insertion
        prediction_data = [
            (feature_ids[i], predictions[i], datetime.now(), source)
            for i in range(len(input_data_list))
        ]

        cursor.executemany(
            """
            INSERT INTO predictions (id, prediction, created_at, source)
            VALUES (%s, %s, %s, %s)
            """
            , prediction_data
        )

        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error inserting predictions into database: {e}")
    finally:
        cursor.close()
        conn.close()

# Load all the model components
model_path = Path("models", "model.joblib").resolve()
preprocessor_path = Path("models", "preprocessors.joblib").resolve()

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request body structure for single prediction
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
    return {"message": "Welcome to the Heart Failure Prediction API. Use /predict for predictions and /past_predictions for history."}

# Define preprocessing function
def preprocess_input(data: pd.DataFrame):
    # Preprocess data using existing joblib
    processed_data = preprocessor.transform(data)
    return processed_data

# Predict endpoint for single or batch prediction
@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(None)
):
    try:
        json_data = None

        # Check if JSON data is provided
        if request.headers.get("Content-Type") == "application/json":
            json_data = await request.json()
            print("Received JSON data:", json_data)
        
        # Handle single or batch JSON input
        if json_data:
            if isinstance(json_data, list):  # Batch JSON
                df = pd.DataFrame(json_data)
            else:  # Single JSON input
                df = pd.DataFrame([json_data])
            source = "webapp"
        
        # Handle CSV file upload
        elif file:
            df = pd.read_csv(file.file)
            source = "scheduled"
        
        else:
            raise HTTPException(status_code=400, detail="No input data provided")

        # Preprocess the input data and get predictions
        processed_data = preprocessor.transform(df)
        predictions = model.predict(processed_data).tolist()

        # Prepare output with predictions
        df['Prediction'] = ["Risk of heart failure" if pred == 1 else "No risk of heart failure" for pred in predictions]

        # Insert records if needed for other parts
        input_data_list = df.to_dict(orient="records")
        # Add your batch_insert_predictions function call here if necessary
        
        return {"predictions_with_features": input_data_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

# Fetch Past Predictions    
@app.get("/past_predictions")
def fetch_past_predictions(start_date: str = None, end_date: str = None, source: str = "all"):
    try:
        conn = connect_to_db()
        if conn is None:
            raise HTTPException(status_code=500, detail="Database connection failed.")

        cursor = conn.cursor()

        # Build the query dynamically based on the filters
        query = """
            SELECT f.age, f.sex, f.chest_pain_type, f.resting_bp, f.cholesterol, 
                   f.max_hr, f.exercise_angina, f.oldpeak, f.st_slope, p.prediction, p.source, p.created_at
            FROM features f 
            JOIN predictions p ON f.id = p.id
            WHERE 1=1
        """

        params = []

        # Apply date range filter
        if start_date:
            query += " AND p.created_at >= %s"
            params.append(datetime.strptime(start_date, '%Y-%m-%d'))

        if end_date:
            query += " AND p.created_at <= %s"
            params.append(datetime.strptime(end_date, '%Y-%m-%d'))

        # Apply source filter
        if source != "all":
            query += " AND p.source = %s"
            params.append(source)

        cursor.execute(query, tuple(params))
        records = cursor.fetchall()

        cursor.close()
        conn.close()

        # Structure the output data
        results = [
            {
                "Age": record[0],
                "Sex": record[1],
                "ChestPainType": record[2],
                "RestingBP": record[3],
                "Cholesterol": record[4],
                "MaxHR": record[5],
                "ExerciseAngina": record[6],
                "Oldpeak": record[7],
                "ST_Slope": record[8],
                "Prediction": "Risk of heart failure" if record[9] == 1 else "No risk of heart failure",
                "Source": record[10],
                "Created At": record[11].strftime('%Y-%m-%d %H:%M:%S')
            }
            for record in records
        ]

        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching past predictions: {str(e)}")
