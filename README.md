# Credit Card Fraud Detection Using XGBoost and Flask

## Overview
This project aims to detect fraudulent credit card transactions using machine learning. The model is built using **XGBoost**, a highly efficient and scalable algorithm, and deployed via a **Flask** web application.

## Features
- **Machine Learning Model**: Uses XGBoost for accurate fraud detection.
- **Web API Deployment**: Flask is used to provide a REST API for making predictions.
- **Scalable & Efficient**: Handles transaction data efficiently to detect fraud.

## Project Structure
```plaintext
├── app.py              # Flask application code
├── credit_card_model.joblib  # Trained XGBoost model file
├── requirements.txt    # Python dependencies
├── README.md           # Project description
└── dataset.csv         # (Optional) Dataset used for training
