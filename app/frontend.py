
import streamlit as st
import pandas as pd
import io
import requests
import os
import plotly.express as px

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Amazon Delivery Dashboard", layout="wide")
st.title("Amazon Delivery System Dashboard")

# Helper functions
def list_models(models_dir="../models"):
    if not os.path.exists(models_dir):
        return []
    return [f for f in os.listdir(models_dir) if f.endswith((".pkl", ".joblib"))]

def is_csv_size_allowed(uploaded_file, max_mb: int = 10) -> bool:
    if uploaded_file is None:
        return False
    max_bytes = int(max_mb * 1024 * 1024)
    if hasattr(uploaded_file, "size"):
        try:
            return int(uploaded_file.size) <= max_bytes
        except Exception:
            pass
    # Fallback: try to get bytes and check length
    try:
        data = uploaded_file.getvalue()
        return len(data) <= max_bytes
    except Exception:
        return False

def upload_csv(action, timeout=60):
    upload_test = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"])
    if not upload_test:
        return None
    
    if not is_csv_size_allowed(upload_test, max_mb=10):
        st.error(f"CSV exceeds maximum allowed size of {10} MB")
        return None

    st.info(f"Selected file: {upload_test.name} — {upload_test.size} bytes")
    files = {
        "file": (upload_test.name, upload_test.getvalue(), "text/csv")
    }

    if action == "Evaluate on test set":
        endpoint = f"{API_URL}/upload_metrics"
    elif action == "Predict batch":
        endpoint = f"{API_URL}/upload_predict"
    else:
        return None

    with st.spinner(f"Processing with {endpoint} ..."):
        try:
            resp = requests.post(endpoint, files=files, timeout=timeout)
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {e}")
            return None

    if resp.status_code != 200:
        st.error(f"Failed (HTTP {resp.status_code}): {resp.text}")
        return None

    if action == "Evaluate on test set":
        data = resp.json()
        if "metrics" in data:
            metrics = data["metrics"]
            metrics_df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
            st.subheader("Evaluation Metrics")
            st.table(metrics_df.set_index("metric"))
        else:
            st.json(data)
        return data
        
    elif action == "Predict batch":
        try:
            pred_df = pd.read_csv(io.StringIO(resp.content.decode("utf-8")))
            st.subheader("Batch Predictions")
            st.dataframe(pred_df.head(200))
            csv = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
            return pred_df
        except Exception as e:
            st.error(f"Failed to parse response: {e}")
            return None

def manual_input_form():
    st.subheader("Single Order Prediction")
    col1, col2 = st.columns(2)

    with col1:
        agent_age = st.number_input("Agent Age", min_value=18, max_value=80, value=30)
        agent_rating = st.number_input("Agent Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
        store_lat = st.number_input("Store Latitude", format="%.6f", value=22.745049)
        store_long = st.number_input("Store Longitude", format="%.6f", value=75.892471)
        drop_lat = st.number_input("Drop Latitude", format="%.6f", value=22.765049)
        drop_long = st.number_input("Drop Longitude", format="%.6f", value=75.912471)

    with col2:
        weather = st.selectbox("Weather", ["Sunny", "Stormy", "Sandstorms", "Windy", "Fog", "Cloudy"])
        traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"])
        vehicle = st.selectbox("Vehicle", ["motorcycle", "scooter", "van", "bicycle"])
        area = st.selectbox("Area", ["Urban", "Metropolitian", "Semi-Urban", "Other"])
        category = st.selectbox("Category", ["Clothing", "Electronics", "Sports", "Cosmetics", "Toys", "Snacks", "Shoes", "Jewelry", "Apparel", "Grocery", "Outdoors", "Kitchen", "Books", "Pet Supplies", "Skincare", "Home"])

    if st.button("Predict Delivery Time", type="primary"):
        payload = {
            "Agent_Age": agent_age,
            "Agent_Rating": agent_rating,
            "Store_Latitude": store_lat,
            "Store_Longitude": store_long,
            "Drop_Latitude": drop_lat,
            "Drop_Longitude": drop_long,
            "Weather": weather,
            "Traffic": traffic,
            "Vehicle": vehicle,
            "Area": area,
            "Category": category
        }
        
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                time = result['predicted_delivery_time']
                st.success(f"Predicted Delivery Time: **{time:.2f} minutes**")
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Connection error: {str(e)}")

def mysql_input(action):
    st.markdown("### MySQL Connection")
    c1, c2 = st.columns(2)
    with c1:
        mysql_host = st.text_input("Host", "localhost")
        mysql_user = st.text_input("Username", "root")
        mysql_password = st.text_input("Password", "", type="password")
    with c2:
        mysql_port = st.text_input("Port", "3306")
        mysql_db = st.text_input("Database", "amazon_delivery")
        mysql_table = st.text_input("Table Name", "deliveries_test")
        
    if st.button("Fetch & Process from MySQL"):
        mysql_data = {
            "host": mysql_host,
            "port": mysql_port,
            "database": mysql_db,
            "username": mysql_user,
            "password": mysql_password,
            "table": mysql_table
        }
        try:
            resp = requests.post(f"{API_URL}/mysql", json=mysql_data)
            if resp.status_code == 200:
                data = resp.json()
                df = pd.DataFrame(data)
                st.write(f"Fetched {len(df)} rows.")
                st.dataframe(df.head())
                if action == "Predict batch":
                    st.info("Simulation: Predictions would be generated for these rows.")
            else:
                st.error(f"MySQL Error: {resp.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

# Sidebar Navigation
page = st.sidebar.selectbox("Navigation", ["Prediction Dashboard", "Data Explorer", "System Overview"])

if page == "Prediction Dashboard":
    st.header("Prediction Dashboard")
    
    # Model Selection
    model_list = list_models("../models")
    if model_list:
        selected_model = st.selectbox("Select Model", model_list)
        if st.button("Load Model"):
            requests.post(f"{API_URL}/load_model", json={"model_path": f"models/{selected_model}"})
            st.success(f"Model {selected_model} loaded!")
    else:
        st.warning("No models found in 'models/' directory.")

    input_method = st.radio("Input Method", ["Manual Input", "Upload CSV", "MySQL Database"], horizontal=True)
    
    if input_method == "Manual Input":
        manual_input_form()
    elif input_method == "Upload CSV":
        action = st.selectbox("Action", ["Predict batch", "Evaluate on test set"])
        upload_csv(action)
    elif input_method == "MySQL Database":
        action = st.selectbox("Action", ["Predict batch", "Evaluate on test set"])
        mysql_input(action)

elif page == "Data Explorer":
    st.header("Delivery Data Explorer")
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        area_filter = st.selectbox("Filter by Area", ["All", "Urban", "Metropolitian", "Semi-Urban", "Other"])
    with col2:
        categories = ["All", "Clothing", "Electronics", "Sports", "Cosmetics", "Toys", "Snacks", "Jewelry", "Shoes", "Home", "Kitchen", "Books", "Grocery", "Outdoors", "Pet Supplies", "Skincare", "Apparel"]
        category_filter = st.selectbox("Filter by Category", categories)
        
    limit = st.slider("Rows to fetch", 10, 500, 50)
    
    params = {"limit": limit}
    if area_filter != "All":
        params["area"] = area_filter.strip()
    if category_filter != "All":
        params["category"] = category_filter
        
    if st.button("Fetch Data"):
        try:
            response = requests.get(f"{API_URL}/deliveries", params=params)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                st.dataframe(df)
            else:
                st.error("Failed to fetch data.")
        except Exception as e:
            st.error(f"Error: {e}")

elif page == "System Overview":
    st.header("System Overview")
    try:
        response = requests.get(f"{API_URL}/metrics")
        if response.status_code == 200:
            metrics = response.json()
            if metrics:
                col1, col2 = st.columns(2)
                with col1:
                    val = metrics.get('average_delivery_time')
                    st.metric("Avg Delivery Time", f"{val:.2f} mins" if (val is not None) else "N/A")
                with col2:
                    weather_data = metrics.get('orders_by_weather', {}) or {}
                    total_orders = sum(weather_data.values()) if isinstance(weather_data, dict) else 0
                    st.metric("Total Orders Processed", total_orders)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Orders by Weather")
                    st.bar_chart(weather_data)
                with c2:
                    st.subheader("Orders by Traffic")
                    st.bar_chart(metrics.get('orders_by_traffic', {}))
            else:
                st.warning("No metrics data available.")
        else:
            st.error("Backend error.")
    except Exception as e:
        st.error(f"Connection error: {e}")

st.markdown("---")
st.caption("Powered by Amazon Delivery System AI")
