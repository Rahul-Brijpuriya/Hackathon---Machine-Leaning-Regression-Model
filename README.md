# Hackathon---Machine-Leaning-Regression-Model
Choose the industrial dataset , apply Ingestion Layer , Data preprocessing  and train different ML model with the evaluation metrics 

# Amazon Delivery Time Prediction System

# Amazon_delivery_time_prediction & Risk Scoring App


## 📖 Project Overview
This project is an end-to-end Machine Learning pipeline designed to predict the probability of a customer leaving a service (Churn). Unlike standard classification which outputs a binary "Yes/No," this system calculates a **Risk Score (0-100%)**, allowing businesses to prioritize intervention for high-risk customers.

The project follows a standard ML lifecycle: **Data Ingestion → Preprocessing → Model Training → Evaluation → Deployment**.

## 📊 Workflow Architecture
*As visualized in the project flowchart:*

1.  **Data Ingestion:** Loads the Amazon_delivery_time_prediction dataset.
2.  **Preprocessing:** Handles missing values, performs One-Hot Encoding for categorical data, and scales numerical features using Scikit-Learn Pipelines.
3.  **Modeling:**
    * **Logistic Regression:** Used as a baseline for interpretability.
    * **Random Forest Classifier:** Used as the final production model for better handling of non-linear data and interactions.
4.  **Deployment:** A web-based user interface built with **Streamlit** that takes user inputs and displays the churn risk in real-time.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Logistic Regression, Random Forest, Pipeline, ColumnTransformer)
* **Web Interface:** Streamlit

## 📂 Project Structure
```text
├── data/
│   └── Amazon_delivery_time.csv  # Raw Dataset
├── models/
│   └── Amazon_delivery_time_prediction_model.pkl                       # Saved trained model pipeline
├── train_model.py                            # Script to preprocess, train, and save model
├── app.py                                    # Streamlit frontend application
├── requirements.txt                          # List of dependencies
└── README.md                                 # Project documentation
