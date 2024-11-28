import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import traceback

# Load the model
MODEL_PATH = "model.joblib"
try:
    model = joblib.load(MODEL_PATH)
    st.success(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Streamlit app
st.title("Adware Prediction App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file for predictions", type=["csv"])

if uploaded_file is not None:
    if model is None:
        st.error("Model not loaded. Cannot make predictions.")
    else:
        try:
            # Read the uploaded CSV file
            data = pd.read_csv(uploaded_file)

            # Required columns
            required_columns = ['Timestamp', 'Flow ID', 'Fwd Packets/s', 'Flow Packets/s',
                                'Flow Duration', 'Flow IAT Mean', 'Flow IAT Max', 'Destination IP']

            # Check for missing columns
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
            else:
                # Extract features
                features = data[required_columns].copy()

                # Convert 'Timestamp' to seconds since epoch
                if 'Timestamp' in features.columns:
                    features['Timestamp'] = pd.to_datetime(features['Timestamp'], errors='coerce').astype('int64') // 10**9

                # Handle categorical columns with LabelEncoder
                label_encoder = LabelEncoder()
                categorical_columns = features.select_dtypes(include=['object']).columns
                for col in categorical_columns:
                    features[col] = label_encoder.fit_transform(features[col].astype(str))

                # Make predictions
                predictions = model.predict(features)

                # Add predictions to the DataFrame
                data['Predictions'] = predictions

                # Reorder columns
                reordered_columns = [col for col in data.columns if col != 'Predictions'] + ['Predictions']
                data = data[reordered_columns]

                # Display results
                st.write("Predictions:")
                st.dataframe(data)

                # Option to download results as CSV
                csv_result = data.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download Predictions as CSV",
                                   data=csv_result,
                                   file_name='predictions.csv',
                                   mime='text/csv')

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.text(traceback.format_exc())
