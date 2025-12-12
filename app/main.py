
from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import pandas as pd
from typing import Optional, List, Dict, Any
import joblib
import os
import numpy as np
import io
from typing import Union

# Base directory (this file's folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use absolute paths so uvicorn's working dir doesn't break finds
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "processed", "amazon_delivery.db"))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "delivery_model.pkl"))

# Optional: mysql connector
try:
    import pymysql
except Exception:
    pymysql = None

app = FastAPI(title="Amazon Delivery System API", version="1.0")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "data/processed/amazon_delivery.db"
MODEL_PATH = "../models/delivery_model.pkl"

# Global model holder
model_data = {}

# ---------- Metrics Helpers ----------
def mse_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0: return float("nan")
    return float(np.mean((y_true[mask] - y_pred[mask]) ** 2))

def rmse_np(y_true, y_pred):
    return float(np.sqrt(mse_np(y_true, y_pred)))

def mae_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0: return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

def r2_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0: return float("nan")
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]
    ss_res = np.sum((y_true_f - y_pred_f) ** 2)
    ss_tot = np.sum((y_true_f - np.mean(y_true_f)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")

def mape_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    if mask.sum() == 0: return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def try_load_model(path: str):
    global model_data
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    model_data = joblib.load(path)
    return model_data

def read_uploadfile_to_df(uploaded_file: UploadFile, max_mb=10) -> pd.DataFrame:
    MAX_UPLOAD_BYTES = max_mb * 1024 * 1024
    content = uploaded_file.file.read()
    uploaded_file.file.seek(0)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max allowed is {max_mb} MB.")
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        df = pd.read_csv(io.BytesIO(content), encoding="latin1")
    return df

@app.on_event("startup")
def load_model():
    global model_data
    try:
        if os.path.exists(MODEL_PATH):
            model_data = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}.")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}. Prediction endpoint will fail.")
    except Exception as e:
        print(f"Error loading model at startup: {e}")

def get_db_connection():
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": bool(model_data)}

# Input Schema
class DeliveryInput(BaseModel):
    Agent_Age: float
    Agent_Rating: float
    Store_Latitude: float
    Store_Longitude: float
    Drop_Latitude: float
    Drop_Longitude: float
    Weather: str
    Traffic: str
    Vehicle: str
    Area: str
    Category: str

@app.post("/load_model")
def load_model_endpoint(payload: Dict[str, str]):
    model_path = payload.get("model_path")
    if not model_path:
        raise HTTPException(status_code=400, detail="model_path missing")
    try:
        # Resolve relative model paths against repository root (one level above `app`)
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(os.path.join(BASE_DIR, "..", model_path))
        try_load_model(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "model_path": model_path}

@app.post("/predict")
def predict_delivery_time(payload: Dict[str, Any]):
    # Supports both Pydantic schema via automatic conversion or direct dict
    if not model_data:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        model = model_data.get("model")
        encoders = model_data.get("encoders")
        
        # Accept payload as dict
        input_dict = payload
        df = pd.DataFrame([input_dict])
        
        # Preprocess
        categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
        for col in categorical_cols:
            if col in df.columns:
                val = str(df.loc[0, col]).strip()
                le = encoders.get(col)
                if le:
                    if val in le.classes_:
                        df.loc[0, col] = le.transform([val])[0]
                    else:
                        df.loc[0, col] = 0
                else:
                     df.loc[0, col] = 0 

        feature_cols = [
            'Agent_Age', 'Agent_Rating', 
            'Store_Latitude', 'Store_Longitude', 
            'Drop_Latitude', 'Drop_Longitude',
            'Weather', 'Traffic', 'Vehicle', 'Area', 'Category'
        ]
        
        # Filter columns that exist
        cols_to_use = [c for c in feature_cols if c in df.columns]
        X = df[cols_to_use] # Note: Model expects specific columns, ensure inputs match training
        
        prediction = model.predict(X)[0]
        
        return {"predicted_delivery_time": float(prediction)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_metrics")
async def upload_metrics_endpoint(file: UploadFile = File(...), target_col: Optional[str] = Form(None)):
    if not model_data:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    df = read_uploadfile_to_df(file)
    
    # Target column resolution
    y_col = target_col if target_col and target_col in df.columns else "Delivery_Time"
    if y_col not in df.columns:
        # Try finding a likely target or use last
        y_col = df.columns[-1]
    
    try:
        y_true = df[y_col].values
        X = df.drop(columns=[y_col])
        
        # Preprocess X similar to single predict
        model = model_data.get("model")
        encoders = model_data.get("encoders")
        
        # Basic preprocessing for batch (simplified)
        categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
        for col in categorical_cols:
            if col in X.columns:
                le = encoders.get(col)
                if le:
                    # Apply label encoding safely
                    X[col] = X[col].astype(str).apply(lambda x: le.transform([x.strip()])[0] if x.strip() in le.classes_ else 0)
        
        # Ensure only feature columns are passed if possible, or trust model to ignore extras if it's robust (RF is not usually)
        # For this hackathon, let's assume input CSV matches feature specs
        feature_cols = model_data.get("feature_cols", [])
        if feature_cols:
             X = X[feature_cols]

        preds = model.predict(X)
        
        metrics = {
            "MSE": mse_np(y_true, preds),
            "RMSE": rmse_np(y_true, preds),
            "MAE": mae_np(y_true, preds),
            "R2": r2_np(y_true, preds),
            "MAPE (%)": mape_np(y_true, preds)
        }
        
        return {"metrics": metrics}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

@app.post("/upload_predict")
async def upload_predict_endpoint(file: UploadFile = File(...)):
    if not model_data:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    df = read_uploadfile_to_df(file)
    model = model_data.get("model")
    encoders = model_data.get("encoders")
    
    X = df.copy()
    # Preprocess
    categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
    for col in categorical_cols:
        if col in X.columns:
            le = encoders.get(col)
            if le:
                X[col] = X[col].astype(str).apply(lambda x: le.transform([x.strip()])[0] if x.strip() in le.classes_ else 0)
    
    feature_cols = model_data.get("feature_cols", [])
    if feature_cols:
            # Handle missing columns by adding 0
            for c in feature_cols:
                if c not in X.columns:
                    X[c] = 0
            X = X[feature_cols]
            
    try:
        preds = model.predict(X)
        out_df = df.copy()
        out_df["Predicted_Delivery_Time"] = preds
        
        stream = io.StringIO()
        out_df.to_csv(stream, index=False)
        stream.seek(0)
        
        return StreamingResponse(
            io.BytesIO(stream.getvalue().encode("utf-8")), 
            media_type="text/csv", 
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mysql")
def mysql_endpoint(payload: Dict[str, Any]):
    if pymysql is None:
        raise HTTPException(status_code=500, detail="pymysql not installed")
    
    host = payload.get("host")
    port = int(payload.get("port", 3306))
    user = payload.get("username")
    password = payload.get("password")
    database = payload.get("database")
    table = payload.get("table")
    
    try:
        conn = pymysql.connect(host=host, port=port, user=user, password=password, db=database, cursorclass=pymysql.cursors.DictCursor)
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM `{table}` LIMIT 100")
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/deliveries")
def get_deliveries(
    page: int = 1,
    limit: int = 50,
    area: Optional[str] = None,
    category: Optional[str] = None
):
    conn = get_db_connection()
    if not conn:
         return [] # DB not ready

    offset = (page - 1) * limit
    
    query = "SELECT * FROM deliveries WHERE 1=1"
    params = []
    
    if area:
        query += " AND area = ?"
        params.append(area)
    if category:
        query += " AND category = ?"
        params.append(category)
        
    query += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    rows = conn.execute(query, params).fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

@app.get("/metrics")
def get_metrics():
    conn = get_db_connection()
    if not conn:
        return {}

    # Average delivery time
    avg_time = conn.execute("SELECT AVG(delivery_time) FROM deliveries").fetchone()[0]
    
    # Counts by Weather
    weather_counts = conn.execute("SELECT weather, COUNT(*) as count FROM deliveries GROUP BY weather").fetchall()
    
    # Counts by Traffic
    traffic_counts = conn.execute("SELECT traffic, COUNT(*) as count FROM deliveries GROUP BY traffic").fetchall()
    
    conn.close()
    
    return {
        "average_delivery_time": avg_time,
        "orders_by_weather": {row['weather']: row['count'] for row in weather_counts},
        "orders_by_traffic": {row['traffic']: row['count'] for row in traffic_counts}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
