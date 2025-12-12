# Hackathon---Machine-Leaning-Regression-Model
Choose the industrial dataset , apply Ingestion Layer , Data preprocessing  and train different ML model with the evaluation metrics 

# Amazon Delivery Time Prediction System

## 📌 Overview
This project is an end-to-end Machine Learning system designed to predict the estimated delivery time for Amazon orders. It utilizes the Amazon Delivery Dataset to build a regression model that factors in agent attributes, weather conditions, traffic density, and geospatial data.

The system includes a reproducible training pipeline, a REST API for inference (FastAPI), and an interactive user dashboard (Streamlit).

## 📂 Project Structure
```text
Amazon-Delivery-System/
├── data/
│   ├── raw/                  # Original CSV (downloaded from Kaggle)
│   └── processed/            # Cleaned data ready for training
├── notebooks/                # Jupyter notebooks for EDA and experiments
├── src/
│   ├── preprocessing.py      # Cleaning, Haversine distance, and encoding functions
│   ├── train.py              # Main training script (saves model.pkl)
│   └── inference.py          # Prediction logic for the API
├── app/
│   ├── main.py               # FastAPI backend
│   └── streamlit_app.py      # UI frontend
├── Dockerfile                # Container configuration
└── requirements.txt          # Python dependencies
