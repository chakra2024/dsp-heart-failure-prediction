from airflow import DAG
from airflow.operators.python import PythonOperator
import datetime
import os
from glob import glob
import requests
import psycopg2
from psycopg2.extras import Json

path_to_stored_data = r"C:\Users\edwin victor\airflow\airflow\processed_data"

DATABASE_HOST = "localhost"  
DATABASE_PORT = 5432  
DATABASE_NAME = "dsphealth"  
DATABASE_USER = "postgres" 
DATABASE_PASSWORD = "postgres"  

def check_for_new_data(**kwargs):
    new_files = glob(os.path.join(path_to_stored_data, '*.csv'))
    if new_files:
        print(f"New files found: {new_files}")
        return new_files
    else:
        print("No new files found.")
        return []  


def save_prediction_to_db(file_name, prediction_data):
    try:
        conn = psycopg2.connect(
            host=DATABASE_HOST,
            port=DATABASE_PORT,
            dbname=DATABASE_NAME,
            user=DATABASE_USER,
            password=DATABASE_PASSWORD
        )
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (file_name, prediction_data) VALUES (%s, %s)",
            (file_name, Json(prediction_data))
        )
        conn.commit()
        cur.close()
        conn.close()
        print(f"Prediction result for {file_name} saved successfully.")
    except Exception as e:
        print(f"Failed to save prediction for {file_name}: {e}")

def make_predictions(**kwargs):
    ti = kwargs['ti']
    new_files = ti.xcom_pull(task_ids='check_for_new_data')

    if not new_files:
        print("No new files to process.")
        return


    for file in new_files:
        try:
            response = requests.post('http://localhost:8000/predict_batch_dag', json={"file_path": file})
            
            if response.status_code == 200:
                prediction_result = response.json()
                print(f"Predictions for {file}: {prediction_result}")
                save_prediction_to_db(file, prediction_result)
            else:
                print(f"Failed to get predictions for {file}: {response.text}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")


with DAG(
    'prediction_dag',  
    default_args={'owner': 'airflow'},  
    schedule_interval='*/2 * * * *', 
    start_date=datetime.datetime(2024, 9, 20),  
    catchup=False 
) as dag:

    check_for_new_data_task = PythonOperator(
        task_id='check_for_new_data',
        python_callable=check_for_new_data,
        provide_context=True,  
    )

    # Task 2: Make predictions
    make_predictions_task = PythonOperator(
        task_id='make_predictions',
        python_callable=make_predictions,
        provide_context=True,
    )

    check_for_new_data_task >> make_predictions_task
