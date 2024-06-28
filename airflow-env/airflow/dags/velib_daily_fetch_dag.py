from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import pandas as pd
import requests
import pytz
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import json

# Define the apply_ml_model function
def apply_ml_model(**kwargs):
    execution_date = kwargs['execution_date']
    parquet_file = f'/opt/airflow/data/velib_disponibilite_{execution_date.strftime("%Y-%m-%d")}.parquet'
    
    # Check if the Parquet file exists
    if not os.path.exists(parquet_file):
        print(f"File {parquet_file} does not exist.")
        return
    
    # Load the data
    data = pd.read_parquet(parquet_file, engine='pyarrow')

    # Preprocessing
    data['is_installed'] = data['is_installed'].map({'OUI': 1, 'NON': 0})
    data['is_renting'] = data['is_renting'].map({'OUI': 1, 'NON': 0})
    data['is_returning'] = data['is_returning'].map({'OUI': 1, 'NON': 0})

    # Drop unnecessary columns
    data = data.drop(['duedate', 'nom_arrondissement_communes', 'code_insee_commune'], axis=1)

    # Handle missing values
    data = data.dropna()

    # Separate features and target
    X = data.drop('numbikesavailable', axis=1)
    y = data['numbikesavailable']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Prepare transformations for numeric and categorical columns
    numeric_features = ['capacity', 'numdocksavailable', 'mechanical', 'ebike', 'coordonnees_geo_lon', 'coordonnees_geo_lat']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_features = ['stationcode', 'name']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the pipeline with the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    
    # Get the station names corresponding to the test set
    station_names = X_test['name'].values

    # Combine predictions with station names
    predictions_with_stations = list(zip(station_names, y_pred.tolist()))

    # Save to JSON file
    output_dir = '/opt/airflow/data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'predictions_{execution_date.strftime("%Y-%m-%d")}.json')
    with open(output_path, 'w') as f:
        json.dump(predictions_with_stations, f)

    # Print the path of the saved file
    print(f"Predictions saved to {output_path}")

# URL de base de l'API Vélib avec paramètres de base
BASE_URL = 'https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/velib-disponibilite-en-temps-reel/records'

def fetch_and_save_velib_data(**kwargs):
    execution_date = kwargs['execution_date']
    start_date = execution_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + timedelta(days=3, microseconds=-1)
    
    # Fetch data for the specified period
    all_records = fetch_all_data_for_period(start_date, end_date)
    
    # Debugging: Print the number of records fetched
    print(f"Nombre total de records récupérés: {len(all_records)}")

    if len(all_records) == 0:
        print("Aucun record récupéré.")
        return

    # Convert data to DataFrame
    df = pd.json_normalize(all_records, sep='_')
    
    # Debugging: Print the number of rows in the DataFrame
    print(f"Nombre total de lignes dans le DataFrame: {len(df)}")
    
    # Convert duedate column to datetime
    df['duedate'] = pd.to_datetime(df['duedate'])
    
    # Save filtered data to Parquet file
    output_dir = '/opt/airflow/data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'velib_disponibilite_{execution_date.strftime("%Y-%m-%d")}.parquet')
    df.to_parquet(output_path, engine='pyarrow')

    print(f"Les données filtrées ont été enregistrées dans '{output_path}'")

# Fetch data from Velib API
def fetch_data(start_date, end_date, limit=100):
    all_records = []
    offset = 0

    while True:
        params = {
            'limit': limit,
            'offset': offset,
            'refine.duedate': f'[{start_date} TO {end_date}]'
        }

        response = requests.get(BASE_URL, params=params)

        if response.status_code == 200:
            data = response.json()
            if 'results' in data:
                records = data['results']
                all_records.extend(records)
                
                # If fewer than 100 records are returned, this is the last page
                if len(records) < limit:
                    break

                offset += limit
            else:
                break
        else:
            print(f"Erreur {response.status_code} lors de la récupération des données: {response.text}")
            break

    return all_records

# Fetch all data for a specific period with pagination
def fetch_all_data_for_period(start_date, end_date):
    all_records = []
    current_start_date = start_date

    while current_start_date < end_date:
        current_end_date = current_start_date + timedelta(hours=1)
        if current_end_date > end_date:
            current_end_date = end_date

        print(f"Récupération des données de {current_start_date} à {current_end_date}")
        records = fetch_data(current_start_date.isoformat(), current_end_date.isoformat())
        all_records.extend(records)

        current_start_date = current_end_date

    return all_records

# Define the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime.now().replace(hour=23, minute=59, second=59, microsecond=0),
    'depends_on_past': False,
    'retries': 1,
}

dag = DAG(
    'velib_daily_fetch_dag',
    default_args=default_args,
    description='Fetch Velib data daily, store in Parquet format, and apply ML model',
    schedule_interval='@daily',  # Use None to make this DAG only run manually
)

# Define the tasks
fetch_and_save_task = PythonOperator(
    task_id='fetch_and_save_task',
    python_callable=fetch_and_save_velib_data,
    provide_context=True,
    dag=dag,
)

apply_ml_model_task = PythonOperator(
    task_id='apply_ml_model_task',
    python_callable=apply_ml_model,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
fetch_and_save_task >> apply_ml_model_task
