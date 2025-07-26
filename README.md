**Project Overview**
This project provides an end-to-end solution for predicting customer churn in the telecommunications industry, fully integrated with Azure Machine Learning.
It combines the power of a finely tuned XGBoost model with Generative AI-driven feature engineering to achieve high prediction accuracy. 
The entire workflow, from data preparation and model training (train.py) to real-time inference (score.py), is orchestrated and deployed using Azure ML services.

**Key Features**
Azure ML-Powered MLOps: Engineered a complete MLOps pipeline for seamless model training, registration, and deployment on Azure ML Studio.

Accurate Churn Prediction (XGBoost): Developed and optimized an XGBoost classification model to predict customer churn, achieving high performance on structured telecom data.

Generative AI for Feature Enhancement: Utilized advanced Generative AI techniques to analyze unstructured customer interaction notes, extracting key sentiments and risk labels. 
These AI-derived features significantly enhance the churn prediction model's accuracy.

Real-time Inference Endpoint: Deployed the trained model as a real-time HTTP endpoint on Azure ML, enabling low-latency predictions for incoming customer data.

Modular Codebase: Separated concerns with dedicated scripts for training (train.py) and scoring (score.py) for clarity and maintainability.

**Problem Statement**
Customer churn is a persistent challenge for telecom companies, leading to substantial revenue loss. 
Traditional prediction models often miss valuable insights embedded in unstructured customer interactions. 
This project addresses this by leveraging Azure ML's capabilities to build, deploy, and manage a robust churn prediction system that not only uses structured data but also enriches predictions with Generative AI insights from customer notes.

**Architecture**

The solution leverages key Azure Machine Learning components:

Data Preparation: Raw customer data (structured + unstructured notes) is prepared for training.

Model Training (train.py):

Executed as an Azure ML Job/Experiment.

Reads prepared data.

Performs Generative AI feature extraction on InteractionNotes.

Trains the XGBoost model using combined features.

Registers the trained model in Azure ML Workspace.

Model Deployment (score.py):

The registered model is deployed as a real-time endpoint using Azure ML Endpoints.

score.py defines the init() function (for loading the model and GenAI component) and run() function (for processing incoming requests, extracting GenAI features, and making predictions).

An environment is defined (e.g., in a conda_dependencies.yml or environment.yml) to specify required Python packages.

Real-time Inference: External applications send HTTP POST requests to the Azure ML Endpoint for predictions.


**Technologies Used**
Cloud Platform: Microsoft Azure

Azure Services: Azure Machine Learning Studio, Azure Key Vault

Programming Language: Python

Machine Learning Libraries: Scikit-learn, XGBoost, Pandas, NumPy

Generative AI: OpenAI API

MLOps: Azure ML SDK, Git, GitHub


**Getting Started**
To explore or set up this project, you'll need an Azure subscription and an Azure Machine Learning Workspace.

Prerequisites
An active Azure Subscription.

An Azure Machine Learning Workspace created in your subscription.

Local Setup (for development/testing scripts)
Clone the Repository:

Bash

git clone https://github.com/your_username/Telecom_Churn_Analysis.git
cd Telecom_Churn_Analysis

Install Local Dependencies:

Bash

pip install -r requirements.txt

Running the Training Pipeline via Azure ML Studio

Upload Your Files:

Go to Azure ML Studio (https://www.google.com/search?q=portal.azure.com > Machine Learning service > Launch Studio).

Navigate to Notebooks in the left-hand menu.

You can either:

Clone this Git repository directly within the Notebooks section (recommended for full Git integration).

Or, if you prefer to upload manually, create a new folder (e.g., telecom-churn) and upload your train.py, data/ folder, model.json (if pre-existing for local testing), and any necessary aml_config files like environment.yml.

Create an Environment (cpu_env.yml):

In Azure ML Studio, go to Environments.

Click Create and choose Custom environment.

Give it a name (e.g., telecom-churn-env).

Specify your base Docker image (e.g., mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest).

Add your Python dependencies (from requirements.txt or a similar list) under "Conda dependencies" or "Pip dependencies". This is crucial for GenAI and XGBoost libraries.

Submit the Training Job (Experiment):

In Azure ML Studio, go to Jobs (or "Experiments" in older versions).

Click Create new job.

Basics: Give your job a name (e.g., churn-training-job).

Code: Select your train.py script as the entry script. Specify the Code folder (the folder where train.py resides, e.g., /telecom-churn).

Compute: Select an appropriate compute instance or compute cluster (e.g., a CPU cluster for training).

Environment: Select the custom environment you created earlier, or upload an environment.yml file.

Inputs (if applicable): If your train.py expects data inputs registered in Azure ML, configure them here.

Review and Create the job.

Monitor the job progress in the Jobs section. Once completed, your trained model should be registered automatically (if your train.py includes model registration logic using mlflow or Azure ML SDK). You can verify this under the Models section.

Deploying the Model as a Real-time Endpoint via Azure ML Studio
Register Your Model: Ensure your trained model (e.g., model.json or a .pkl file) is registered in Azure ML Studio under the Models section. If your training job didn't register it automatically, you can manually upload and register it.

Prepare score.py and Dependencies:

Make sure your score.py file is ready. It should have init() and run(raw_data) functions.

Ensure you have an environment.yml or conda_dependencies.yml file listing all Python packages required by score.py (XGBoost, GenAI libraries, pandas, etc.).

Deploy the Endpoint:

In Azure ML Studio, navigate to Endpoints.

Click + Create.

Endpoint Type: Select Real-time endpoint.

Name & Authentication: Give your endpoint a name (e.g., telecom-churn-api). Choose authentication type.

Deployment:

Name: Give your deployment a name (e.g., churn-deployment-v1).

Model: Select the registered model you want to deploy.

Environment: Select or upload the environment that contains all the dependencies for score.py. This is crucial.

Inference Code: Select your score.py file as the Scoring script and the folder containing it as the Code directory.

Compute: Select the compute type (e.g., "CPU" instance, you can use a small VM for testing like Standard_DS2_v2).

Instance Count: Start with 1.

Review and Create the endpoint.

Monitor the deployment status in the Endpoints section. It might take a few minutes for the deployment to become healthy.

Testing the Real-time Endpoint
Once your endpoint deployment shows a "Healthy" status:

Access Endpoint Details:

In Azure ML Studio, go to Endpoints.

Click on your deployed endpoint name (e.g., telecom-churn-api).

Under the "Consume" tab, you will find:

REST endpoint (Scoring URI): This is the URL you'll send requests to.

Primary Key / Secondary Key: These are the API keys required for authentication.

Send a Test Request using Postman:

Open Postman: Launch the Postman application.

Create a New Request:

Set the HTTP Method to POST.

Enter the Request URL from Azure ML Studio (Scoring URI).

Configure Headers:

Add a header: Authorization with value Bearer YOUR_PRIMARY_KEY (replace YOUR_PRIMARY_KEY with the key from Azure ML Studio).

Add another header: Content-Type with value application/json.

Configure Request Body:

Go to the Body tab.

Select raw and choose JSON from the dropdown.

Enter the input data for prediction in JSON format. The keys must match the column names your model expects, including InteractionNotes.

Example JSON Body (ensure these match your exact model input features and InteractionNotes for GenAI):

JSON

{
    "CustomerID": "C67890",
    "Surname": "Smith",
    "NetworkForRegion": "South",
    "Gender": "Female",
    "Age": 45,
    "Tenure": 60,
    "MonthlyCharges": 99.95,
    "NumOfProducts": 3,
    "HasInternetService": "Yes",
    "IsActiveMember": "No",
    "EstimatedSalary": 75000.00,
    "InteractionNotes": "Customer called about a repeated service disruption. Expressed extreme dissatisfaction and considering alternatives."
}
Send the Request: Click the Send button.

View the Response: The API's response will appear in the response section of Postman.

Example JSON Response:

JSON

{
  "churn_prediction": "Yes",
  "churn_probability": 0.91,
  "genai_sentiment": "Very Negative",
  "genai_risk_tags": ["Service Disruption", "Extreme Dissatisfaction", "Churn Intent"]
}

Project Structure
.
├── train.py                 # Script for data preparation, GenAI feature extraction during training, model training, and model registration.
├── score.py                 # Entry script for the Azure ML real-time endpoint (model loading, GenAI feature extraction for inference, prediction).
├── model.json               # Pre-trained machine learning model artifact (e.g., XGBoost model, or weights). This will be loaded by score.py.
├── data/
│   ├── raw/                 # Original dataset (e.g., train.csv, customer_notes.csv).
│   └── processed/           # Processed datasets used for training.
├── environment.yml          # Conda environment definition for Azure ML deployment (used by both train.py and score.py on Azure).
├── requirements.txt         # Python dependencies for local development/testing. (Primarily for `pip install -r requirements.txt` locally).
└── README.md                # This file

