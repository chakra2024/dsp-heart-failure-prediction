import pandas as pd
import os

og_file_path = r"D:\epita class notes\dsp production\project script\raw_dataset"
file = pd.read_csv(og_file_path)
no_of_rows_per_file = 100
total_no_of_rows_in_the_file = len(file)
total_no_of_files = (total_no_of_rows_in_the_file + no_of_rows_per_file - 1) // no_of_rows_per_file

destination_folder = r"D:\epita class notes\dsp production\project script\raw_dataset_1"
os.makedirs(destination_folder, exist_ok=True) 

for i in range(total_no_of_files):
    start_row = i * no_of_rows_per_file
    end_row = min(start_row + no_of_rows_per_file, total_no_of_rows_in_the_file)
    split_dataset = file.iloc[start_row:end_row]
    split_dataset = split_dataset.iloc[:, :-1]
    file_name = f"heart{i+1}.csv"
    file_path = os.path.join(destination_folder, file_name)
    split_dataset.to_csv(file_path, index=False)
