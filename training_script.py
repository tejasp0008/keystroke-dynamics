import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# --- Configuration ---
DATA_DIR = 'data'
DATA_FILE = 'keystroke_data.csv'
MODELS_DIR = 'models'
TARGET_PASSWORD = ".tie5Roanl"
# This is the subject ID that the system will try to authenticate against.
# All other subjects will be treated as 'impostors' during training for this binary classification task.
TARGET_SUBJECT_ID = 's002'

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Feature Extraction Function ---
def extract_keystroke_features(df_row, password_chars):
    """
    Extracts Hold (H), Keydown-Keydown (DD), and Keyup-Keydown (UD) features
    from a single row of raw keystroke data.

    Args:
        df_row (pd.Series): A row from the DataFrame containing raw KD/KU times.
        password_chars (list): List of characters in the target password.

    Returns:
        dict: A dictionary of extracted features.
    """
    features = {}

    # 1. Hold Times (H)
    for char in password_chars:
        kd_col = f"{char}_KD"
        ku_col = f"{char}_KU"
        if kd_col in df_row and ku_col in df_row:
            hold_time = df_row[ku_col] - df_row[kd_col]
            features[f"H.{char}"] = hold_time
        else:
            features[f"H.{char}"] = 0 # Default to 0 if data missing, though ideally data is complete

    # 2. Keydown-Keydown (DD) - Digraphs
    for i in range(len(password_chars) - 1):
        char1 = password_chars[i]
        char2 = password_chars[i+1]
        kd1_col = f"{char1}_KD"
        kd2_col = f"{char2}_KD"
        if kd1_col in df_row and kd2_col in df_row:
            dd_time = df_row[kd2_col] - df_row[kd1_col]
            features[f"DD.{char1}.{char2}"] = dd_time
        else:
            features[f"DD.{char1}.{char2}"] = 0

    # 3. Keyup-Keydown (UD) - Digraphs
    for i in range(len(password_chars) - 1):
        char1 = password_chars[i]
        char2 = password_chars[i+1]
        ku1_col = f"{char1}_KU"
        kd2_col = f"{char2}_KD"
        if ku1_col in df_row and kd2_col in df_row:
            ud_time = df_row[kd2_col] - df_row[ku1_col]
            features[f"UD.{char1}.{char2}"] = ud_time
        else:
            features[f"UD.{char1}.H{char2}"] = 0 # This should be UD.{char1}.{char2} if data missing


    return features

# --- Data Loading and Preprocessing ---
def preprocess_data(file_path, target_subject_id, password_chars):
    """
    Loads data, extracts features, and creates labels for classification.
    Labels: 1 for target_subject_id, 0 for others.
    """
    print(f"Loading data from {file_path}...")
    try:
        # Assuming the first column is 'subject' and subsequent columns are timings.
        # Assuming space-separated values as often found in keystroke datasets.
        # Adjust sep=',' if your CSV uses commas.
        raw_df = pd.read_csv(file_path) # Changed to default comma sep, adjust if needed
    except FileNotFoundError:
        print(f"Error: Dataset not found at {file_path}")
        exit()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit()

    # Rename period columns if they are not '.KD' and '.KU' directly
    # This is a common issue with CSV headers.
    if 'period_KD' in raw_df.columns:
        raw_df.rename(columns={'period_KD': '._KD', 'period_KU': '._KU'}, inplace=True)
    if 'R_KD' in raw_df.columns and 'R_KU' in raw_df.columns:
         raw_df.rename(columns={'R_KD': 'R_KD', 'R_KU': 'R_KU'}, inplace=True) # Ensure 'R' is handled as 'R' not 'r' if data is case-sensitive

    # Extract features for each row
    processed_features = raw_df.apply(lambda row: extract_keystroke_features(row, password_chars), axis=1)
    
    # Convert list of dictionaries to DataFrame
    feature_df = pd.DataFrame(processed_features.tolist())

    # Add 'subject' column back to the feature DataFrame
    feature_df['subject'] = raw_df['subject'] # Ensure 'subject' column exists in raw_df

    # Create labels: 1 for target subject, 0 for others
    feature_df['label'] = (feature_df['subject'] == target_subject_id).astype(int)

    # Convert features from milliseconds to seconds (if they are not already)
    # This aligns with standard practices for features of this type and helps scaling
    timing_feature_columns = [col for col in feature_df.columns if col not in ['subject', 'label']]
    feature_df[timing_feature_columns] = feature_df[timing_feature_columns] / 1000.0

    print(f"Data preprocessed. Total samples: {len(feature_df)}")
    print(f"Features extracted: {timing_feature_columns[:5]}... ({len(timing_feature_columns)} total)")
    print(f"Samples for target subject ({target_subject_id}): {feature_df['label'].sum()}")
    print(f"Samples for other subjects: {len(feature_df) - feature_df['label'].sum()}")
    
    # Return features (X) and labels (y)
    X = feature_df[timing_feature_columns]
    y = feature_df['label']

    return X, y, timing_feature_columns

# --- Main Training Script ---
if __name__ == "__main__":
    password_chars = list(TARGET_PASSWORD) # ['.', 't', 'i', 'e', '5', 'R', 'o', 'a', 'n', 'l']
    
    # Load and preprocess data
    data_file_path = os.path.join(DATA_DIR, DATA_FILE)
    X, y, timing_feature_columns = preprocess_data(data_file_path, TARGET_SUBJECT_ID, password_chars)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    print("StandardScaler fitted and saved.")

    # --- Train Random Forest Classifier ---
    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_scaled, y_train)
    rf_predictions = rf_model.predict(X_test_scaled)
    print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
    print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))
    joblib.dump(rf_model, os.path.join(MODELS_DIR, 'rf_model.joblib'))
    print("Random Forest model saved.")

    # --- Train Support Vector Machine (SVM) ---
    print("\nTraining Support Vector Machine (SVM)...")
    svm_model = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
    svm_model.fit(X_train_scaled, y_train)
    svm_predictions = svm_model.predict(X_test_scaled)
    print("SVM Accuracy:", accuracy_score(y_test, svm_predictions))
    print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))
    joblib.dump(svm_model, os.path.join(MODELS_DIR, 'svm_model.joblib'))
    print("SVM model saved.")

    # --- Train Long Short-Term Memory (LSTM) Network ---
    print("\nTraining LSTM Network...")
    
    # Reshape data for LSTM: (samples, timesteps, features)
    # For this type of data, 1 timestep is common, where each feature is a different keystroke timing.
    X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

    lstm_model = Sequential([
        LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1, activation='sigmoid') # Binary classification
    ])

    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = lstm_model.fit(
        X_train_lstm, y_train,
        epochs=100, # Increased epochs, but early stopping will manage
        batch_size=32,
        validation_split=0.2, # Use part of training data for validation
        callbacks=[early_stopping],
        verbose=1
    )

    lstm_predictions_proba = lstm_model.predict(X_test_lstm)
    lstm_predictions = (lstm_predictions_proba > 0.5).astype(int)
    print("LSTM Accuracy:", accuracy_score(y_test, lstm_predictions))
    print("LSTM Classification Report:\n", classification_report(y_test, lstm_predictions))
    lstm_model.save(os.path.join(MODELS_DIR, 'lstm_model.h5'))
    print("LSTM model saved.")

    print("\nTraining complete. All models and scaler saved in the 'models' directory.")