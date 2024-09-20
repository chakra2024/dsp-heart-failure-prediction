# Heart Failure Prediction System
This repository is a complete Heart Failure Prediction System. It combines the power of machine learning models, FastAPI for building the backend API, Streamlit for the web interface, PostgreSQL for data storage, and Airflow for scheduled predictions. The machine learning model helps in predicting heart failure based on patient medical data.

# Table of Contents
Features
Tech Stack
Prerequisites
Installation & Setup
 1. Clone the repository
 2. Set up the virtual environment
 3. Install Python dependencies
 4. Set up PostgreSQL
 5. Airflow Setup
 6. Start the FastAPI backend
 7. Start the Streamlit frontend
Usage
Directory Structure
Contributing
License

# Features
Predict the risk of heart failure using a trained machine learning model.
Single and batch predictions via FastAPI.
Store predictions and patient data in PostgreSQL.
A user-friendly interface using Streamlit for making predictions.
Schedule batch predictions with Apache Airflow.
Fetch and display past predictions with filters based on date and source.

# Tech Stack
FastAPI - API framework for the backend.
Streamlit - Frontend framework for building the web UI.
PostgreSQL - Database for storing patient data and predictions.
Airflow - For scheduling batch predictions.
Scikit-learn - Machine learning library used for model building.
Pydantic - Data validation and settings management.
Docker - Containerization for easy deployment (optional).

# Prerequisites
Before setting up the project, ensure you have the following software installed:

Python 3.9+
PostgreSQL 12+
Airflow (For scheduling predictions)
Docker (Optional but recommended for easier environment management)

# Installation & Setup
1. Clone the repository
 bash command:
 git clone https://github.com/your-username/heart-failure-prediction.git
 cd heart-failure-prediction

2. Set up the virtual environment
 It is recommended to use a virtual environment for managing dependencies. You can use venv or conda:

Using venv:
Code:
python3 -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

Using conda:
Code
conda create --name hf-prediction python=3.9
conda activate hf-prediction

3. Install Python dependencies
All necessary dependencies are listed in requirements.txt.
code:
pip install -r requirements.txt
or privude the name of the packages manually like:
pip install fastapi uvicorn scikit-learn pydantic pandas streamlit apache

4. Set up PostgreSQL
Install PostgreSQL if not already installed:

On Ubuntu: sudo apt-get install postgresql postgresql-contrib
On Windows/Mac: Download from here - https://www.postgresql.org/download/ 

Create a database:
bash code:
psql -U postgres
CREATE DATABASE dsphealth;

Create the required tables:

CREATE TABLE features (
  id SERIAL PRIMARY KEY,
  age INT,
  sex VARCHAR(1),
  chest_pain_type VARCHAR(10),
  resting_bp INT,
  cholesterol INT,
  max_hr INT,
  exercise_angina VARCHAR(1),
  oldpeak FLOAT,
  st_slope VARCHAR(10)
);

CREATE TABLE predictions (
  id INT REFERENCES features(id),
  prediction INT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  source VARCHAR(50)
);
Update the database credentials in app.py for your PostgreSQL database setup.

5. Airflow Setup
If you plan to run Airflow for scheduling batch predictions:

Install Airflow:
bash code:
pip install apache-airflow

Initialize Airflow:
bash code:
airflow db init

Start the Airflow web server and scheduler:
airflow webserver --port 8080
airflow scheduler
Create and configure an Airflow DAG for running batch predictions. Add your DAGs in the dags/ folder.

6. Start the FastAPI backend
Update the model_path and preprocessor_path in the FastAPI script (app.py) to point to your trained model and preprocessor.
Run the FastAPI server:
uvicorn app:app --reload
The API will be available at http://127.0.0.1:8000.

7. Start the Streamlit frontend
Navigate to the webapp.py file and run:

streamlit run webapp.py
The frontend will be available at http://localhost:8501.

# Usage
1. Single Prediction
Go to the Streamlit web app (http://localhost:8501) and enter patient data.
Click on Predict to get the heart failure prediction.
2. Batch Prediction (CSV)
Upload a CSV file with patient data on the Batch Prediction page.
The API will process all entries and return predictions.
3. View Past Predictions
Use the Past Predictions page in the Streamlit app to filter predictions based on date and source.
4. Schedule Predictions
Use Airflow to schedule batch predictions. Create a DAG that triggers predictions from a file and inserts them into the database.

# Directory Structure
.
├── .git
├── airflow
│   ├── config
│   ├── dags
│   │   ├── data_ingestion.py
│   │   └── make_predictions.py
│   ├── logs
│   ├── main_data
│   ├── plugins
│   ├── processed_data
│   ├── raw_data
│   ├── docker-compose.yaml
│   ├── Dockerfile
│   ├── requirements.txt
│   └── split_dataset.py
├── api
│   ├── app.py
│   └── requirements.txt
├── data
│   ├── heart.csv
│   └── testdata.csv
├── models
│   ├── model.joblib
│   └── preprocessors.joblib
├── python_files
│   └── heart_failure_prediction.ipynb
├── webapp
│   ├── webapp.py
├── .gitattributes
└── README.md

# Contributing
Contributions are welcome! Feel free to open a pull request or submit issues if you encounter any problems.
Please follow the standard GitHub flow for submitting contributions:

Fork the repository.
Create a new branch (git checkout -b feature-name).
Make your changes.
Commit your changes (git commit -m "Add feature").
Push to the branch (git push origin feature-name).
Submit a pull request.

# License
This project is licensed under the MIT License. See the LICENSE file for more details.
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
