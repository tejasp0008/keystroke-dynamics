# 🔐 Keystroke Dynamics Biometric Authentication Demo

This project demonstrates a **proof-of-concept for biometric authentication using keystroke dynamics**. It captures a user's typing pattern for a fixed password and uses machine learning models to authenticate against a predefined user.

---

## 🚀 Overview

The system comprises:
- A **FastAPI** backend for data processing and ML predictions
- A **lightweight HTML/CSS/JavaScript** frontend for interaction

All incoming typing patterns are compared to a fixed target user's learned behavior (`Subject ID: s002`) using pre-trained models.

---

## ✨ Features

- 🎯 **Keystroke Capture**: Records keydown and keyup events precisely
- 🔍 **Feature Extraction**: Computes Hold time, Down-Down time, and Up-Down time
- 🤖 **ML Predictions**:
  - LSTM Neural Network
  - Random Forest
  - Support Vector Machine
- 🗳️ **Majority Voting**: Combines model outputs for final verdict
- 🧑‍💻 **Frontend UI**: Easy-to-use typing interface with real-time results

---

## 🛠️ Technology Stack

| Component      | Tech |
|----------------|------|
| **Backend**    | FastAPI, Uvicorn, Python 3 |
| **ML**         | TensorFlow/Keras, Scikit-learn, Pandas, NumPy |
| **Frontend**   | HTML5, CSS3, JavaScript |

---

## 📁 Project Structure

```
keystroke-dynamics/
├── backend.py                 # FastAPI app
├── training_script.py         # Model training script
├── models/                    # Trained models
│   ├── lstm_model.h5
│   ├── rf_model.joblib
│   ├── svm_model.joblib
│   └── scaler.joblib
├── data/
│   └── keystroke_data.csv     # Dataset for training
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── requirements.txt
├── README.md
└── venv/                      # Python virtual environment
```

---

## ⚙️ Setup & Installation

### ✅ Prerequisites

- Python 3.8+
- `pip`

---

### 1️⃣ Clone the Repository

```bash
git clone <repository_url>
cd keystroke-dynamics
```

---

### 2️⃣ Create & Activate Virtual Environment

```bash
python -m venv venv
```

Activate:

- **Windows**: `.\venv\Scripts\activate`
- **macOS/Linux**: `source venv/bin/activate`

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install fastapi uvicorn pandas scikit-learn tensorflow keras joblib numpy
```

---

### 4️⃣ Train ML Models

Ensure `data/keystroke_data.csv` exists.

```bash
python training_script.py
```

This will generate:
- `lstm_model.h5`
- `rf_model.joblib`
- `svm_model.joblib`
- `scaler.joblib`

in the `models/` directory.

---

## ▶️ Running the Application

### 🔥 Start Backend

```bash
python backend.py
```

Visit: [http://localhost:8000](http://localhost:8000)

---

### 💻 Access Frontend

Your browser will open the interactive interface served from `index.html`.

---

## 🧪 How to Use

1. **Type the password**: `.tie5Roanl`
2. **Set Subject ID**: Default is `s002`
3. **Press Enter**: After typing the full password
4. **See Results**: Predictions from LSTM, SVM, RF + final decision
5. **Reset Demo**: Try again with different rhythm or ID (e.g., `s003`)

---

## 📜 License

This project is licensed under the **MIT License**.
