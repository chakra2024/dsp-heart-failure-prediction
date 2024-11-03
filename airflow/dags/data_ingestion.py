import pandas as pd
import shutil
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from random import choice
from airflow.exceptions import AirflowSkipException
import os

def read_data(**kwargs):
    docker_raw_data_folder = '/opt/airflow/raw_data'
    system_raw_data_folder = '../airflow/raw_data'
    raw_data_folder = ''
    if os.path.exists(docker_raw_data_folder):
        raw_data_folder = docker_raw_data_folder
    elif os.path.exists(system_raw_data_folder):
        raw_data_folder = system_raw_data_folder
    else:
        raise ValueError("Path dont exit")

    tot_dir_files = os.listdir(raw_data_folder)
    
    if not tot_dir_files:
        raise AirflowSkipException("No files in the folder, skipping DAG execution.")
    
    file_name = choice(tot_dir_files)
    file_path = os.path.join(raw_data_folder, file_name)
    df = pd.read_csv(file_path)
    
    ti = kwargs['ti']
    ti.xcom_push(key='file_path', value=file_path)
    ti.xcom_push(key='file_name', value=file_name)
    
    return df

def validate_data(**kwargs):
    return

def send_alerts(**kwargs):
    return

def save_statistics(**kwargs):
    return

def save_file(**kwargs):
    ti = kwargs['ti']
    file_path = ti.xcom_pull(key='file_path')
    file_name = ti.xcom_pull(key='file_name')

    docker_processed_data_folder = '/opt/airflow/processed_data'
    system_processed_data_folder = '../airflow/processed_data'
    processed_data_folder = ''
    if os.path.exists(docker_processed_data_folder):
        processed_data_folder = docker_processed_data_folder
    elif os.path.exists(system_processed_data_folder):
        processed_data_folder = system_processed_data_folder
    else:
        raise ValueError("Path dont exit")
    destination_path = os.path.join(processed_data_folder,file_name)
    shutil.move(file_path, destination_path)

    ti.xcom_push(key='processed_data_data', value=destination_path)



with DAG(
    dag_id="Read_data_and_move_data",
    description="Reading data from raw data folder",
    schedule_interval="* * * * *",
    start_date=datetime(2024, 9, 15),
    catchup=False,
) as dag:
    
    read_data = PythonOperator(
        task_id="read_data",
        python_callable=read_data,
        provide_context=True
    )
    validate_data = PythonOperator(
        task_id = "validate_data",
        python_callable=validate_data,
        provide_context=True
    )
    send_alerts = PythonOperator(
        task_id = "send_alerts",
        python_callable=send_alerts,
        provide_context=True
    )
    save_file = PythonOperator(
        task_id = "save_file",
        python_callable=save_file,
        provide_context=True
    )
    save_statistics = PythonOperator(
        task_id = "save_statistics",
        python_callable=save_statistics,
        provide_context=True
    )

    read_data >> validate_data
    validate_data >> [send_alerts, save_file, save_statistics]




