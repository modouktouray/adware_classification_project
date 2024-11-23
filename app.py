from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback
import os

app = Flask(__name__)

# Load the saved model
MODEL_PATH = "model.joblib"
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the ML Model API! Upload a CSV file to make predictions."})

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    """
    Endpoint to make predictions from a CSV file.
    Accepts a file upload via POST request.
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Cannot make predictions."}), 500

    try:
        # Check if a file is included in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request."}), 400

        # Retrieve the file from the request
        file = request.files['file']

        # Check if the file is valid
        if file.filename == '':
            return jsonify({"error": "No file selected for uploading."}), 400

        # Read the CSV file into a Pandas DataFrame
        data = pd.read_csv(file)

        # Ensure the DataFrame contains the required column format
        required_columns = ['Source Port', 'Flow Duration', 'Flow IAT Max', 'Fwd Packets/s', 'Flow Packets/s', 'Flow IAT Mean']
        for column in required_columns:
            if column not in data.columns:
                return jsonify({"error": f"Missing required column: {column}"}), 400

       
        features = data[required_columns]

        # Make predictions
        predictions = model.predict(features)

        # Add predictions to the original DataFrame
        data['Predictions'] = predictions

        # Convert the result to JSON
        result = data.to_dict(orient='records')
        return jsonify({"predictions": result})

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    # Ensure the app runs on port 5000
    app.run(debug=True)
