**Commodity Price Index Forecasting**
This project provides a robust and advanced solution for forecasting commodity price indices. It uses a modern machine learning workflow, combining powerful time-series models with a microservice architecture for easy deployment.

**Key Features**
**Advanced Forecasting Models**: Utilizes state-of-the-art models including XGBoost, LightGBM, and CatBoost for accurate price index prediction.

**Ensemble Modeling**: Combines the predictions from multiple models using a Voting Regressor to produce a more stable and reliable forecast.

**Production-Ready API**: The core forecasting model is exposed as a lightweight and efficient API built with Flask. This allows the model to be easily integrated into any application.

**Interactive Web Application**: A user-friendly front-end built with Streamlit allows users to select a date and get an instant price forecast by communicating with the Flask API.

**Robust Data Pipeline**: Handles messy, real-world CSV data by dynamically cleaning and extracting the correct price indices.

**Model Persistence**: Trained models are saved using joblib to avoid re-training, making the deployment fast and efficient.
