from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
import traceback
import os

app = Flask(__name__)

# Load model
MODEL_PATH = "model.joblib"
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('upload.html')

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
        required_columns = ['Timestamp', 'Flow ID', 'Fwd Packets/s', 'Flow Packets/s', 'Flow Duration', 'Flow IAT Mean', 'Flow IAT Max', 'Destination IP']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return jsonify({"error": f"Missing required columns: {missing_columns}"}), 400

        # Extract features
        features = data[required_columns].copy()

        # Convert 'Timestamp' to seconds since epoch
        if 'Timestamp' in features.columns:
            features['Timestamp'] = pd.to_datetime(features['Timestamp'], errors='coerce').astype('int64') // 10**9

        # Handle categorical columns with LabelEncoder
        label_encoder = LabelEncoder()
        categorical_columns = features.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            features[col] = label_encoder.fit_transform(features[col].astype(str))  # Ensure all values are string for consistency

        # Make predictions
        predictions = model.predict(features)

        # Add predictions as a column
        data['Predictions'] = predictions

        # Reorder columns to ensure 'Predictions' is at the end
        reordered_columns = [col for col in data.columns if col != 'Predictions'] + ['Predictions']
        data = data[reordered_columns]

        # Convert the result to JSON
        result = data.to_dict(orient='records')
        return jsonify({"predictions": result})

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(debug=True)
