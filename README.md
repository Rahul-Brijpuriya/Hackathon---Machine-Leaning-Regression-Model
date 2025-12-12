# Amazon Delivery Time Prediction System

## 📖 Project Overview
This project is an end-to-end Machine Learning pipeline designed to predict the **estimated time of delivery** for Amazon orders based on various logistical factors. Unlike simple estimation methods, this system utilizes advanced regression models to analyze historical data—such as delivery person age, ratings, location coordinates, traffic density, and weather conditions—to forecast precise delivery times.

The project follows a standard ML lifecycle: **Data Ingestion → Preprocessing → Model Training → Evaluation → Deployment**.

## 📊 Workflow Architecture
1. **Data Ingestion:** Loads the `Amazon_delivery_time.csv` dataset containing delivery logs.
2. **Preprocessing:** - **Handling Missing Values:** Imputation strategies for null values.
   - **Feature Engineering:** Calculating the distance between the restaurant and delivery location using the Haversine formula.
   - **Encoding:** One-Hot Encoding for categorical data (e.g., Weather, Traffic).
   - **Scaling:** Standardizing numerical features using Scikit-Learn Pipelines.
3. **Modeling:**
   - **Linear Regression:** Baseline model to establish linear relationships.
   - **Ridge Regression:** Regularized linear model to handle multicollinearity.
   - **Random Forest Regressor:** Bagging ensemble method to handle non-linear data and reduce variance.
   - **XGBoost Regressor:** Boosting ensemble method that provided the highest accuracy.
4. **Deployment:** A web-based user interface built with **Streamlit** that takes delivery parameters and predicts the time in minutes.

## 📈 Model Evaluation
The models were trained and evaluated on the industrial dataset. Below is the performance comparison across all trained models, sorted by performance:

| Metric | Linear Regression | Ridge Regression | Random Forest | XGBoost Regressor |
| :--- | :--- | :--- | :--- | :--- |
| **RMSE** (Root Mean Squared Error) | 33.3036 | 33.3042 | 23.1124 | **22.1654** |
| **MAE** (Mean Absolute Error) | 26.3118 | 26.3116 | 17.6397 | **17.1974** |
| **R² Score** | 0.5782 | 0.5782 | 0.7969 | **0.8132** |
| **Adjusted R²** | 0.5779 | 0.5778 | 0.7967 | **0.8130** |
| **MAPE** (Mean Absolute Percentage Error) | 27.96% | 27.96% | 16.16% | **15.94%** |

> **Key Observation:** > * **Tree-based models (XGBoost & Random Forest) significantly outperform** the linear models, explaining ~80-81% of the variance compared to ~58% for linear models. 
> * **XGBoost** is the best-performing model with the lowest RMSE (22.16) and highest R² (0.8132).

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Linear, Ridge, Random Forest), XGBoost
* **Web Interface:** Streamlit
* **Utils:** Pickle (for model serialization)

## 📂 Project Structure
```text
├── data/
│   └── Amazon_delivery_time.csv      # Raw Dataset
├── models/
│   └── final_model.pkl               # Saved Best Model (XGBoost)
├── src/
│   ├── logger.py                     # Logging configuration
│   ├── exception.py                  # Custom exception handling
│   └── utils.py                      # Utility functions
├── train_model.py                    # Script to preprocess, train, and save model
├── app.py                            # Streamlit frontend application
├── requirements.txt                  # List of dependencies
└── README.md                         # Project documentation
