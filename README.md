# Hackathon---Machine-Leaning-Regression-Model
Choose the industrial dataset , apply Ingestion Layer , Data preprocessing  and train different ML model with the evaluation metrics 


Amazon-Delivery-System/
├── data/
│   ├── raw/                  # Original CSV
│   └── processed/            # Cleaned data for training
├── notebooks/                # Jupyter notebooks for EDA and experiments
├── src/
│   ├── preprocessing.py      # Cleaning and Feature Engineering functions
│   ├── train.py              # Training script (saves model.pkl)
│   └── inference.py          # Prediction logic
├── app/
│   ├── main.py               # FastAPI backend
│   └── streamlit_app.py      # UI frontend
├── Dockerfile
└── requirements.txt
