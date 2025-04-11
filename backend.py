"""
Backend for the Keystroke Dynamics Demo project.

This FastAPI backend provides endpoints to analyze keystroke data and simulate authentication predictions
using dummy machine learning models (LSTM, SVM, and Random Forest). Each model's prediction is simulated based on
simple conditional logic, and a majority vote is used to derive a final prediction.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI(title="Keystroke Dynamics Backend", description="API for analyzing keystroke data and predicting authentication using simulated ML models.", version="1.0")

class Keystroke(BaseModel):
    key: str
    interval: int  # Time between key presses in milliseconds

class KeystrokeData(BaseModel):
    keystrokes: List[Keystroke]

def extract_features(data: KeystrokeData):
    """
    Extract features from keystroke data.
    
    Returns:
        total_keys (int): Total number of keystrokes.
        avg_interval (float): Average interval between keystrokes.
    """
    total_keys = len(data.keystrokes)
    avg_interval = sum(ks.interval for ks in data.keystrokes) / total_keys if total_keys > 0 else 0
    return total_keys, avg_interval

def simulate_ml_predictions(total_keys: int, avg_interval: float):
    """
    Simulate predictions for keystroke data using dummy ML models.
    
    Returns:
        lstm_prediction (str): Simulated prediction from LSTM model.
        svm_prediction (str): Simulated prediction from SVM model.
        rf_prediction (str): Simulated prediction from Random Forest model.
    """
    lstm_prediction = "Authenticated" if avg_interval < 200 else "Not Authenticated"
    svm_prediction = "Authenticated" if total_keys > 5 else "Not Authenticated"
    rf_prediction = "Authenticated" if avg_interval < 250 else "Not Authenticated"
    return lstm_prediction, svm_prediction, rf_prediction

def majority_vote(predictions: List[str]) -> str:
    """
    Determine final prediction using majority voting.
    
    Args:
        predictions (List[str]): List of predictions from different models.
        
    Returns:
        str: Final prediction ("Authenticated" or "Not Authenticated").
    """
    auth_count = sum(1 for pred in predictions if pred == "Authenticated")
    return "Authenticated" if auth_count >= 2 else "Not Authenticated"

@app.post("/analyze", summary="Analyze keystroke data")
async def analyze_keystrokes(data: KeystrokeData):
    """
    Analyze keystroke data by extracting features.
    
    Returns:
        JSON response with total keystrokes and average interval.
    """
    total_keys, avg_interval = extract_features(data)
    return {
        "message": "Data analyzed successfully",
        "total_keys": total_keys,
        "average_interval": avg_interval
    }

@app.post("/predict", summary="Predict authentication status based on keystroke data")
async def predict_keystrokes(data: KeystrokeData):
    """
    Predict authentication result based on keystroke data.

    Simulates predictions from three dummy models and uses majority voting to determine the final result.
    
    Returns:
        JSON response with individual model predictions, final prediction, and extracted features.
    """
    total_keys, avg_interval = extract_features(data)
    lstm_pred, svm_pred, rf_pred = simulate_ml_predictions(total_keys, avg_interval)
    final_prediction = majority_vote([lstm_pred, svm_pred, rf_pred])
    return {
        "predictions": {
            "LSTM": lstm_pred,
            "SVM": svm_pred,
            "RandomForest": rf_pred
        },
        "final_prediction": final_prediction,
        "total_keys": total_keys,
        "average_interval": avg_interval
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)