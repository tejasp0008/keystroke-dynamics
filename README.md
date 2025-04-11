# Keystroke Dynamics Demo

This project demonstrates a keystroke dynamics system that captures typing patterns via a web interface and uses a FastAPI backend to analyze and simulate authentication predictions. The objective is to explore behavioral biometrics for user authentication using dummy machine learning models.

## Features

- **Keystroke Data Capture:**  
  An HTML front-end captures keystroke timings, including key presses and the intervals between them.

- **Data Analysis:**  
  The backend provides an `/analyze` endpoint to extract features from the keystroke data, such as the total number of keystrokes and the average interval between them.

- **Simulated Authentication Prediction:**  
  A `/predict` endpoint simulates predictions using dummy logic for three models:
  - **LSTM Model:** Authenticates if the average interval is less than 200 ms.
  - **SVM Model:** Authenticates if the total number of keystrokes is more than 5.
  - **Random Forest Model:** Authenticates if the average interval is less than 250 ms.  
  A majority vote among these predictions determines the final authentication result.

## Project Structure

```
keystroke-dynamics-demo/
├── index.html       # Front-end to capture keystroke data
├── backend.py       # FastAPI backend with analysis and prediction endpoints
└── README.md        # Project documentation
```

## How to Run

1. **Install Dependencies:**  
   Ensure you have Python installed, then install the required packages:
   ```
   pip install fastapi uvicorn pydantic
   ```

2. **Start the Backend Server:**  
   Navigate to the project directory and run:
   ```
   uvicorn backend:app --reload
   ```
   This will start the FastAPI server on port 8000 with automatic reload on code changes.

3. **Open the Front-End:**  
   Open `index.html` in your web browser. Use the provided buttons to submit keystroke data for analysis and prediction, and view the results via alerts.

## API Endpoints

- **POST /analyze**  
  - **Description:** Analyze keystroke data to calculate total keystrokes and average interval.
  - **Request JSON Format:**
    ```json
    {
      "keystrokes": [
        {"key": "a", "interval": 120},
        {"key": "b", "interval": 150}
      ]
    }
    ```
  - **Response Example:**
    ```json
    {
      "message": "Data analyzed successfully",
      "total_keys": 10,
      "average_interval": 180.0
    }
    ```

- **POST /predict**  
  - **Description:** Predict authentication status using simulated ML predictions.
  - **Request JSON Format:** Same as `/analyze`
  - **Response Example:**
    ```json
    {
      "predictions": {
        "LSTM": "Authenticated",
        "SVM": "Not Authenticated",
        "RandomForest": "Authenticated"
      },
      "final_prediction": "Authenticated",
      "total_keys": 10,
      "average_interval": 180.0
    }
    ```

## Future Enhancements

- Replace dummy logic with actual machine learning models (e.g., LSTM, SVM, Random Forest) trained on keystroke dynamics data.
- Enhance feature extraction and preprocessing techniques.
- Improve the user interface with real-time visualization of typing patterns.
- Implement security features for data encryption and secure transmission.

## License

This project is licensed under the MIT License.