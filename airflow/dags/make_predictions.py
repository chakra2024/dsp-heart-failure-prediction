import os
import pandas as pd
import requests
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from random import choice

# API URL (Make sure this is your correct API URL)
API_URL = 'http://127.0.0.1:8000/predict_batch_dag'

# Check for new files in the processed_data folder
def check_for_new_data(**kwargs):
    processed_data_folder = '/opt/airflow/processed_data'
    all_files = os.listdir(processed_data_folder)

    if not all_files:
        raise ValueError("No new files in the folder")

    file_name = choice(all_files)
    file_path = os.path.join(processed_data_folder, file_name)
    
    ti = kwargs['ti']
    ti.xcom_push(key='file_name', value=file_name)
    ti.xcom_push(key='file_path', value=file_path)

def make_prediction(**kwargs):
    ti = kwargs['ti']
    file_path = ti.xcom_pull(key='file_path')

    # Read the file
    df = pd.read_csv(file_path)

    # Convert dataframe to JSON format expected by API
    input_data = df.to_dict(orient='records')

    # Make the API request
    response = requests.post(API_URL, json=input_data)

    if response.status_code == 200:
        prediction = response.json()
        print("Prediction:", prediction)
    else:
        print(f"Error in prediction: {response.status_code} - {response.text}")

# DAG definition
with DAG(
    dag_id="make_predictions",
    description="Make predictions from the processed data folder every minute",
    schedule_interval="*/1 * * * *",
    start_date=datetime(2024, 9, 15),
    catchup=False,
) as dag:

    check_new_data_task = PythonOperator(
        task_id="check_for_new_data",
        python_callable=check_for_new_data,
        provide_context=True
    )

    make_prediction_task = PythonOperator(
        task_id="make_prediction",
        python_callable=make_prediction,
        provide_context=True
    )

    check_new_data_task >> make_prediction_task
