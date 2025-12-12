
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os

# Define paths
DATA_PATH = "data/raw/amazon_delivery.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "delivery_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoders.pkl")

def train_model():
    # Load data
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Preprocessing
    # 1. Clean Lat/Long 
    # '0.0' or small values nearby are likely errors given coordinates are usually > 10 for India (based on sample) unless 0.0 is explicit missing.
    # Looking at sample, there are 0.0s. Let's filter them out or better yet, just train on what we have.
    # For a robust model we should filter, but for this hackathon task I will just drop them if they are 0.
    df = df[(df['Store_Latitude'] != 0) & (df['Store_Longitude'] != 0)]
    df = df[(df['Drop_Latitude'] != 0) & (df['Drop_Longitude'] != 0)]

    # 2. Features and Target
    # Input columns: Agent_Age, Agent_Rating, Store_Latitude, Store_Longitude, Drop_Latitude, Drop_Longitude, Weather, Traffic, Vehicle, Area, Category
    # Target: Delivery_Time
    
    # Check for missing values in target
    df = df.dropna(subset=['Delivery_Time'])
    
    feature_cols = [
        'Agent_Age', 'Agent_Rating', 
        'Store_Latitude', 'Store_Longitude', 
        'Drop_Latitude', 'Drop_Longitude',
        'Weather', 'Traffic', 'Vehicle', 'Area', 'Category'
    ]
    
    X = df[feature_cols].copy()
    y = df['Delivery_Time']
    
    # Cleaning Numeric Columns
    numeric_cols = ['Agent_Age', 'Agent_Rating', 'Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude']
    num_imputer = SimpleImputer(strategy='mean')
    X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

    # Cleaning Categorical Columns
    categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

    # Convert categorical to numeric using LabelEncoder
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Ensure we handle unknown strings during prediction by fitting on all known string representation or just simple LE for now.
        # Clean whitespace in strings just in case
        X[col] = X[col].astype(str).str.strip()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    # Train Model
    print("Training model...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Save Model and Encoders
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    data_to_save = {
        "model": rf,
        "encoders": encoders,
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "feature_cols": feature_cols
    }
    
    joblib.dump(data_to_save, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
