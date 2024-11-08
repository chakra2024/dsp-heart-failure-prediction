import pandas as pd
import os

og_file_path = r"C:\Users\SOHAM\Git_Repositories\DataScience_Projects\dsp-heart-failure-prediction\airflow\main_data\heart.csv"
file = pd.read_csv(og_file_path)
total_no_of_files = 50
no_of_rows_per_file = len(file) // total_no_of_files

for i in range(total_no_of_files):
    start_row = i * no_of_rows_per_file
    end_row = start_row + no_of_rows_per_file
    split_dataset = file[start_row : end_row]
    destination_folder = r"C:\Users\SOHAM\Git_Repositories\DataScience_Projects\dsp-heart-failure-prediction\airflow\raw_data"
    os.makedirs(destination_folder, exist_ok=True) 
    file_name = f"heart{i+1}.csv"
    file_path = os.path.join(destination_folder, file_name)
    split_dataset.to_csv(file_path, index=False)


