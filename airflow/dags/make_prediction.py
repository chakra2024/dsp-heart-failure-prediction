from airflow import DAG
from airflow.operators.python import PythonOperator
import datetime
import os
from glob import glob
import requests
import psycopg2
from psycopg2.extras import Json
from airflow.models import Variable
from airflow.utils.dates import days_ago


docker_path_to_stored_data = '/opt/airflow/processed_data'
system_path_to_stored_data = '../airflow/processed_data'
path_to_stored_data = ''

if os.path.exists(docker_path_to_stored_data):
    path_to_stored_data = docker_path_to_stored_data
elif os.path.exists(system_path_to_stored_data):
    path_to_stored_data = system_path_to_stored_data
else:
    raise ValueError("Path dont exit")

DATABASE_HOST = "localhost"
DATABASE_PORT = 5432
DATABASE_NAME = "dsphealth"
DATABASE_USER = "postgres"
DATABASE_PASSWORD = "postgres"

def check_for_new_data(**kwargs):
    last_processed = Variable.get("last_processed_time", default_var=None)
    new_files = []

    for file in glob(os.path.join(path_to_stored_data, '*.csv')):
        modification_time = os.path.getmtime(file)
        file_mod_time = datetime.datetime.fromtimestamp(modification_time)
        
        # Check if the file was modified since the last processed time
        if last_processed is None or file_mod_time > datetime.datetime.strptime(last_processed, '%Y-%m-%d %H:%M:%S'):
            new_files.append(file)
    
    if new_files:
        Variable.set("last_processed_time", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        return new_files
    else:
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
            "INSERT INTO predictions (file_name, prediction_data, created_at, source) VALUES (%s, %s, %s, %s)",
            (file_name, Json(prediction_data), datetime.datetime.now(), 'scheduled')
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Failed to save prediction for {file_name}: {e}")


def make_predictions(**kwargs):
    ti = kwargs['ti']
    new_files = ti.xcom_pull(task_ids='check_for_new_data')

    if not new_files:
        return

    for file in new_files:
        try:
            response = requests.post('http://localhost:8000/predict_batch_dag', json={"file_path": file})
            
            if response.status_code == 200:
                prediction_result = response.json()
                save_prediction_to_db(file, prediction_result)
            else:
                raise ValueError(f"Failed to get predictions for {file}: {response.text}")
        except Exception as e:
            raise RuntimeError(f"Error processing file {file}: {e}")


with DAG(
    'prediction_dag',
    default_args={'owner': 'airflow'},
    schedule_interval='* * * * *',
    start_date=datetime.datetime(2024, 9, 20),
    catchup=False
) as dag:

    check_for_new_data_task = PythonOperator(
        task_id='check_for_new_data',
        python_callable=check_for_new_data,
        provide_context=True,
    )

    make_predictions_task = PythonOperator(
        task_id='make_predictions',
        python_callable=make_predictions,
        provide_context=True,
    )

    check_for_new_data_task >> make_predictions_task
