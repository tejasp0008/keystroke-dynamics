from fastapi import FastAPI, HTTPException
# Remove 'HTMLResponse' as it's not strictly needed if StaticFiles serves HTML
# from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles # Essential import for serving frontend
from pydantic import BaseModel, Field
from typing import List, Dict, Union
import uvicorn
import joblib
import os
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np

# --- Configuration ---
MODELS_DIR = 'models'
# This TARGET_SUBJECT_ID MUST match the one used during training in training_script.py
TARGET_SUBJECT_ID = 's002'
FRONTEND_DIR = 'frontend' # Path to your frontend files

app = FastAPI(
    title="Keystroke Dynamics Backend",
    description="API for analyzing keystroke data and predicting authentication using trained ML models.",
    version="2.0"
)

# --- Serve Static Frontend Files ---
# IMPORTANT: This must be mounted *before* any other routes that might catch "/"
# This single line handles serving index.html at "/" and all other files
# (style.css, script.js, favicon.ico, etc.) from the 'frontend' directory at their root paths.
app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend_app")

# Remove the explicit @app.get("/") route you added, as it's now correctly handled
# by StaticFiles with html=True.
# You also don't need 'FileResponse' import if using this method.
# @app.get("/", response_class=HTMLResponse)
# async def serve_frontend():
#     return FileResponse("frontend/index.html")

# --- Pydantic Models (No changes needed here, keeping as is) ---
class Keystroke(BaseModel):
    """
    Represents a single extracted timing feature sent from the frontend.
    The 'interval' holds the pre-calculated H, DD, or UD value.
    """
    key: str # e.g., "H.period", "DD.period.t", "UD.period.t"
    interval: int  # Time in milliseconds

class KeystrokeData(BaseModel):
    """
    The full payload received from the frontend's /predict endpoint.
    """
    keystrokes: List[Keystroke]
    subject_id: str = Field(..., description="ID of the subject (e.g., 's002') who is claiming to type.")


# --- Global variables for models and scaler (No changes needed here, keeping as is) ---
models = {}
scaler = None
timing_feature_names = [] # To store the ordered list of feature names for preprocessing

# --- Startup Event Handler (No changes needed here, keeping as is) ---
@app.on_event("startup")
async def load_ml_models():
    """
    Load pre-trained ML models and scaler when the FastAPI application starts.
    Also, define the expected feature order.
    """
    global models, scaler, timing_feature_names

    print(f"Loading models from {MODELS_DIR}...")
    try:
        # Define the expected order of features based on the password '.tie5Roanl'
        # This MUST match how features are calculated in your frontend's script.js
        # and how the training data was structured in training_script.py.
        password_keys = ['.', 't', 'i', 'e', '5', 'R', 'o', 'a', 'n', 'l']
        
        # 1. Hold times (H.key) - 10 features
        for key in password_keys:
            timing_feature_names.append(f"H.{key}")
        
        # 2. Keydown-keydown times (DD.key1.key2) - 9 features
        for i in range(len(password_keys) - 1):
            key1 = password_keys[i]
            key2 = password_keys[i+1]
            timing_feature_names.append(f"DD.{key1}.{key2}")
        
        # 3. Keyup-keydown times (UD.key1.key2) - 9 features
        for i in range(len(password_keys) - 1):
            key1 = password_keys[i]
            key2 = password_keys[i+1]
            timing_feature_names.append(f"UD.{key1}.H{key2}")
        
        # Total expected features: 10 + 9 + 9 = 28
        print(f"Expected {len(timing_feature_names)} features: {timing_feature_names}")


        # Load Scaler
        scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("StandardScaler loaded.")
        else:
            raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please run training_script.py first.")

        # Load Random Forest
        rf_model_path = os.path.join(MODELS_DIR, 'rf_model.joblib')
        if os.path.exists(rf_model_path):
            models['RandomForest'] = joblib.load(rf_model_path)
            print("Random Forest model loaded.")
        else:
            raise FileNotFoundError(f"Random Forest model not found at {rf_model_path}. Please run training_script.py first.")

        # Load SVM
        svm_model_path = os.path.join(MODELS_DIR, 'svm_model.joblib')
        if os.path.exists(svm_model_path):
            models['SVM'] = joblib.load(svm_model_path)
            print("SVM model loaded.")
        else:
            raise FileNotFoundError(f"SVM model not found at {svm_model_path}. Please run training_script.py first.")

        # Load LSTM
        lstm_model_path = os.path.join(MODELS_DIR, 'lstm_model.h5')
        if os.path.exists(lstm_model_path):
            models['LSTM'] = load_model(lstm_model_path)
            print("LSTM model loaded.")
        else:
            raise FileNotFoundError(f"LSTM model not found at {lstm_model_path}. Please run training_script.py first.")

        # Verify the number of features matches the models' input
        # We need to make sure the loaded models expect the same number of features
        # as we are deriving.
        if models['RandomForest'] is not None and hasattr(models['RandomForest'], 'n_features_in_') and models['RandomForest'].n_features_in_ != len(timing_feature_names):
            print(f"WARNING: RF model expected {models['RandomForest'].n_features_in_} features, but {len(timing_feature_names)} derived. Mismatch might cause errors.")
        if models['SVM'] is not None and hasattr(models['SVM'], 'n_features_in_') and models['SVM'].n_features_in_ != len(timing_feature_names):
            print(f"WARNING: SVM model expected {models['SVM'].n_features_in_} features, but {len(timing_feature_names)} derived. Mismatch might cause errors.")
        if models['LSTM'] is not None and models['LSTM'].input_shape[-1] != len(timing_feature_names):
             print(f"WARNING: LSTM model expected {models['LSTM'].input_shape[-1]} features, but {len(timing_feature_names)} derived. Mismatch might cause errors.")
        
        print("All models and scaler loaded successfully!")

    except FileNotFoundError as fnfe:
        print(f"Configuration Error: {fnfe}")
        raise RuntimeError(f"Missing model files. Ensure training_script.py was run: {fnfe}")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise RuntimeError(f"Failed to load ML models: {e}")

# --- Helper Functions (No changes needed here, keeping as is) ---
def convert_keystrokes_to_features_df(data: KeystrokeData, expected_feature_names: List[str]) -> pd.DataFrame:
    """
    Converts the incoming KeystrokeData list into a pandas DataFrame
    with features in the expected order, ready for scaling and prediction.
    """
    if len(data.keystrokes) != len(expected_feature_names):
        raise ValueError(
            f"Expected {len(expected_feature_names)} keystroke feature entries "
            f"from frontend, but received {len(data.keystrokes)}. "
            f"Ensure frontend feature extraction matches backend's expectation."
        )
    
    # Create a dictionary to map feature names to their values
    # This ensures correct ordering even if the frontend sends them slightly out of order
    # (though frontend script.js now sends them in the correct order).
    feature_map = {ks.key: ks.interval for ks in data.keystrokes}
    
    # Extract values in the predefined order
    feature_values_ms = [feature_map.get(name, 0) for name in expected_feature_names]
    
    # Convert milliseconds to seconds if your training data was in seconds (as per the dataset description)
    feature_values_seconds = [val / 1000.0 for val in feature_values_ms]
    
    # Create a DataFrame for a single sample (row)
    feature_df = pd.DataFrame([feature_values_seconds], columns=expected_feature_names)
    
    return feature_df

def majority_vote_prediction(predictions: Dict[str, int]) -> str:
    """
    Determine final prediction using majority voting from binary predictions (0 or 1).
    1 means "Authenticated" (i.e., matches TARGET_SUBJECT_ID's pattern).
    0 means "Not Authenticated" (i.e., does NOT match TARGET_SUBJECT_ID's pattern).
    """
    auth_count = sum(1 for pred_val in predictions.values() if pred_val == 1) # Count models predicting "1" (Authenticated for Target)
    return "Authenticated" if auth_count >= 2 else "Not Authenticated"

# --- API Endpoints (No changes needed for /predict itself, keeping as is) ---

@app.post("/predict", summary="Predict authentication status using trained ML models")
async def predict_keystrokes(data: KeystrokeData):
    """
    Predict authentication result based on keystroke data using pre-trained
    LSTM, SVM, and Random Forest models. The final prediction is determined
    by a majority vote.

    The input `keystrokes` must be a list of `Keystroke` objects where the `interval`
    value for each `Keystroke` object corresponds to a pre-extracted timing feature
    (H, DD, UD). The order of these entries is crucial and must match the order
    used during model training (defined in `timing_feature_names` on startup).
    """
    if not models or not scaler or not timing_feature_names:
        raise HTTPException(status_code=500, detail="ML models, scaler, or feature names not loaded. Server is not ready.")
    
    try:
        # Convert input keystrokes to a feature vector (DataFrame)
        input_features_df = convert_keystrokes_to_features_df(data, timing_feature_names)
        
        # Scale the input features using the loaded scaler
        input_features_scaled = scaler.transform(input_features_df)

        # Reshape for LSTM: (1 sample, 1 timestep, num_features)
        input_features_lstm = input_features_scaled.reshape(1, 1, input_features_scaled.shape[1]) 

        # Get predictions from each model
        model_raw_predictions = {} # Store 0 or 1
        model_prediction_labels = {} # Store "Authenticated" or "Not Authenticated"

        # Random Forest Prediction
        # predict_proba returns [[prob_class_0, prob_class_1]]
        rf_pred_proba = models['RandomForest'].predict_proba(input_features_scaled)[0][1] # Probability of being TARGET_SUBJECT
        model_raw_predictions['RandomForest'] = 1 if rf_pred_proba >= 0.5 else 0 
        model_prediction_labels['RandomForest'] = "Authenticated" if model_raw_predictions['RandomForest'] == 1 else "Not Authenticated"

        # SVM Prediction
        svm_pred_proba = models['SVM'].predict_proba(input_features_scaled)[0][1] # Probability of being TARGET_SUBJECT
        model_raw_predictions['SVM'] = 1 if svm_pred_proba >= 0.5 else 0
        model_prediction_labels['SVM'] = "Authenticated" if model_raw_predictions['SVM'] == 1 else "Not Authenticated"

        # LSTM Prediction
        lstm_pred_proba = models['LSTM'].predict(input_features_lstm, verbose=0)[0][0] # Probability of being TARGET_SUBJECT
        model_raw_predictions['LSTM'] = 1 if lstm_pred_proba >= 0.5 else 0
        model_prediction_labels['LSTM'] = "Authenticated" if model_raw_predictions['LSTM'] == 1 else "Not Authenticated"
        
        # Determine final prediction using majority voting (on the 0/1 raw predictions)
        final_prediction_for_target_subject = majority_vote_prediction(model_raw_predictions)
        
        system_authentication_status_for_claimant = "Not Authenticated" # Default to Not Authenticated

        if data.subject_id == TARGET_SUBJECT_ID:
            if final_prediction_for_target_subject == "Authenticated":
                system_authentication_status_for_claimant = "Authenticated"
                justification = f"Claimant '{data.subject_id}' matches the target subject's biometric pattern."
            else:
                justification = f"Claimant '{data.subject_id}' does NOT match the target subject's biometric pattern."
        else:
            # For this binary classification demo:
            if final_prediction_for_target_subject == "Authenticated":
                # This means the provided keystrokes match the TARGET_SUBJECT_ID,
                # even if the claimant is someone else. This is an "impostor" scenario.
                system_authentication_status_for_claimant = "Not Authenticated" # Impostor trying to be Target
                justification = f"Claimant '{data.subject_id}' does NOT match claimant's ID, but their typing strongly resembles the target subject '{TARGET_SUBJECT_ID}'. High alert."
            else:
                # Claimant is not the target, and their typing doesn't match the target. Expected.
                system_authentication_status_for_claimant = "Not Authenticated"
                justification = f"Claimant '{data.subject_id}' does not match the target subject '{TARGET_SUBJECT_ID}'s biometric pattern, which is expected. System is designed to authenticate '{TARGET_SUBJECT_ID}'."

        return {
            "predictions": model_prediction_labels, # Individual model labels
            "final_prediction_for_target_subject": final_prediction_for_target_subject, # Overall verdict for target
            "claimant_subject_id": data.subject_id,
            "system_authentication_status_for_claimant": system_authentication_status_for_claimant,
            "justification": justification
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Data processing error: {ve}")
    except Exception as e:
        print(f"Prediction API error: {e}", exc_info=True) # Print traceback for debugging
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)