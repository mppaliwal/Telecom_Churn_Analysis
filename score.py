# score.py
import os
import json
import pandas as pd
import joblib

# Called when the service is loaded
def init():
    global model, imputation_values, expected_cols
    
    # Get the path to the deployed model file
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model_output_dir')
    
    # Deserialize the model and imputation values
    model_file = os.path.join(model_path, 'churn_model.joblib')
    imputation_file = os.path.join(model_path, 'imputation_values.joblib')
    
    model = joblib.load(model_file)
    imputation_values = joblib.load(imputation_file)
    
    # Define expected columns for the model input
    expected_cols = ['NetworkScore', 'Age', 'Tenure', 'MonthlyCharge', 'NumOfProducts', 
                     'HasInternetService', 'IsActiveMember', 'Region_North', 'Region_South', 
                     'Region_West', 'Gender_Male', 'Gender_Other']

# Called for every request
def run(raw_data):
    try:
        # 1. Load the JSON data into a pandas DataFrame
        data_dict = json.loads(raw_data)['data']
        data = pd.DataFrame(data_dict)
        
        # 2. Apply the same preprocessing
        # Handle missing values using the saved values from training
        data['NetworkScore'].fillna(imputation_values['NetworkScore'], inplace=True)
        data['Age'].fillna(imputation_values['Age'], inplace=True)
        data['IsActiveMember'].fillna(imputation_values['IsActiveMember'], inplace=True)
        data['EstimatedMonthlyUsage'].fillna(imputation_values['EstimatedMonthlyUsage'], inplace=True)
        
        # One-Hot Encoding
        data_encoded = pd.get_dummies(data, columns=['Region', 'Gender'], drop_first=True)
        
        # Ensure all columns the model expects are present
        for col in expected_cols:
            if col not in data_encoded.columns:
                data_encoded[col] = 0
        
        # Keep only the expected columns in the correct order
        X_test = data_encoded[expected_cols]
        
        # 3. Make predictions
        predictions = model.predict(X_test)
        
        # 4. Return predictions as JSON
        return json.dumps(predictions.tolist())
    except Exception as e:
        error = str(e)
        return json.dumps({'error': error})