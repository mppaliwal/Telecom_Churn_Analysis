# train.py
import argparse
import os
import pandas as pd
from xgboost import XGBClassifier
import joblib

# 1. Define arguments for the script
parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, dest='data_folder', help='Path to the data folder')
parser.add_argument('--model_output_dir', type=str, dest='model_output_dir', help='Directory to save the trained model')
args = parser.parse_args()

# 2. Load the training data
print("Loading data...")
train_path = os.path.join(args.data_folder, 'train_data.csv')
train = pd.read_csv(train_path)

# 3. Apply all preprocessing steps from your notebook
print("Preprocessing data...")
# Modify data types
train['Age'] = train['Age'].astype('Int64')
train['IsActiveMember'] = train['IsActiveMember'].astype('Int64')

# Handle missing values
# IMPORTANT: Save the values used for imputation to use later on the test data
imputation_values = {
    'NetworkScore': train['NetworkScore'].median(),
    'Age': train['Age'].median(),
    'IsActiveMember': train['IsActiveMember'].mode()[0],
    'EstimatedMonthlyUsage': train['EstimatedMonthlyUsage'].mean()
}

train['NetworkScore'].fillna(imputation_values['NetworkScore'], inplace=True)
train['Age'].fillna(imputation_values['Age'], inplace=True)
train['IsActiveMember'].fillna(imputation_values['IsActiveMember'], inplace=True)
train['EstimatedMonthlyUsage'].fillna(imputation_values['EstimatedMonthlyUsage'], inplace=True)

# Drop extra columns
train_new = train.drop(columns=['CustomerID', 'Surname'])

# One-Hot Encoding
train_encoded = pd.get_dummies(train_new, columns=['Region', 'Gender'], drop_first=True)

# Ensure all expected columns are present after encoding
# This handles cases where a category might be missing in some data slice
expected_cols = ['NetworkScore', 'Age', 'Tenure', 'MonthlyCharge', 'NumOfProducts', 
                 'HasInternetService', 'IsActiveMember', 'EstimatedMonthlyUsage', 
                 'Exited', 'Region_North', 'Region_South', 'Region_West', 
                 'Gender_Male', 'Gender_Other']
for col in expected_cols:
    if col not in train_encoded.columns:
        if col != 'Exited': # Don't add the target column if it's not there
            train_encoded[col] = 0

# 4. Train the model
print("Training model...")
X = train_encoded.drop('Exited', axis=1)
Y = train_encoded['Exited']

model = XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X, Y)

# 5. Save the model and imputation values
print("Saving model and imputation values...")
os.makedirs(args.model_output_dir, exist_ok=True)
model_path = os.path.join(args.model_output_dir, 'churn_model.joblib')
imputation_path = os.path.join(args.model_output_dir, 'imputation_values.joblib')

joblib.dump(model, model_path)
joblib.dump(imputation_values, imputation_path)

print("Script finished successfully.")