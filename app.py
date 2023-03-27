import os
import pathlib
import pickle
import pandas as pd
from flask import Flask, request, jsonify, flash
from werkzeug.utils import secure_filename
from Monthly_Revenue_Forecast import (
    read_remote_data, preprocess_data, predict_next_month, ALLOWED_EXTENSIONS, allowed_file, create_and_save_arima_model
)

# Initialize Flask app
app = Flask(__name__)

# Load the ARIMA model from file
arima_results_path = 'arima_results.pkl'
if not pathlib.Path(arima_results_path).exists():
    raise Exception("Please create your arima_results.pkl model by running python monthly_revenue_forecast.py first.")

with open(arima_results_path, 'rb') as f:
    model = pickle.load(f)

# Remote data URL
url = "https://github.com/aurimas13/Revenue-Forecast/blob/main/Dataset/Barbora%20Homework.xlsx?raw=true"

# Define the upload folder
UPLOAD_FOLDER = 'Dataset'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cache the data by reading it once
revenue_data, weather_data = read_remote_data(url)

# Define the forecast route
@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    global revenue_data, weather_data

    # Handle GET request to update cached data
    if request.method == 'GET':
        revenue_data, weather_data = read_remote_data(url)
    
    # Handle POST request to upload a file
    elif request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return jsonify({'error': 'No file uploaded'})

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return jsonify({'error': 'No selected file'})

        # Check if the uploaded file has a valid format
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format'})

        # Save the uploaded file securely
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        input_data_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

       # Read the uploaded revenue and weather data
        revenue_input = pd.read_excel(input_data_path, sheet_name='November_Revenue', parse_dates=['Date'], engine='openpyxl')
        weather_input = pd.read_excel(input_data_path, sheet_name='November_Weather', parse_dates=['dt'], engine='openpyxl')

        # Update cached data
        revenue_data = revenue_data.append(revenue_input)
        weather_data = weather_data.append(weather_input)

    # Preprocess the data
    preprocessed_data = preprocess_data(revenue_data, weather_data)

    # Make a prediction
    prediction = predict_next_month(model, preprocessed_data)

    # Return the prediction as JSON
    return jsonify({'prediction': int(prediction)})

# Start the Flask app
if __name__ == '__main__':
    app.run()
