
import pandas as pd
import sqlite3
import os

def clean_data(input_path):
    print(f"Reading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Strip whitespace from string columns
    str_cols = ['Order_ID', 'Order_Date', 'Order_Time', 'Pickup_Time', 'Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    # Convert numeric columns explicitly, coercing errors
    num_cols = ['Agent_Age', 'Agent_Rating', 'Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude', 'Delivery_Time']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN in critical columns or 0.0 coordinates (likely errors for Indian context if out of bounds, but for now just exact 0.0)
    # Note: 0.0, 0.0 is in the Atlantic Ocean, definitely not an Amazon store/drop in India.
    df = df[df['Store_Latitude'] != 0.0]
    df = df[df['Store_Longitude'] != 0.0]
    df = df[df['Drop_Latitude'] != 0.0]
    df = df[df['Drop_Longitude'] != 0.0]
    
    df.dropna(inplace=True)
    
    # Rename columns to match snake_case convention for DB
    df.rename(columns={
        'Order_ID': 'id',
        'Agent_Age': 'agent_age',
        'Agent_Rating': 'agent_rating',
        'Store_Latitude': 'store_lat',
        'Store_Longitude': 'store_long',
        'Drop_Latitude': 'drop_lat',
        'Drop_Longitude': 'drop_long',
        'Order_Date': 'order_date',
        'Order_Time': 'order_time',
        'Pickup_Time': 'pickup_time',
        'Weather': 'weather',
        'Traffic': 'traffic',
        'Vehicle': 'vehicle',
        'Area': 'area',
        'Delivery_Time': 'delivery_time',
        'Category': 'category'
    }, inplace=True)
    
    print(f"Data cleaned. Rows: {len(df)}")
    return df

def ingest_to_db(df, db_path):
    print(f"Ingesting to {db_path}...")
    conn = sqlite3.connect(db_path)
    df.to_sql('deliveries', conn, if_exists='replace', index=False)
    conn.close()
    print("Ingestion complete.")

if __name__ == "__main__":
    raw_path = "data/raw/amazon_delivery.csv"
    # Project root is assumed to be Amazon-Delivery-System based on where this script usually runs
    # Adjust paths if running from src directory
    if not os.path.exists(raw_path):
        # Try going up one level
        raw_path = "../data/raw/amazon_delivery.csv"
        db_path = "../data/processed/amazon_delivery.db"
    else:
        db_path = "data/processed/amazon_delivery.db"
        
    if os.path.exists(raw_path):
        df = clean_data(raw_path)
        ingest_to_db(df, db_path)
    else:
        print(f"Error: {raw_path} not found.")
