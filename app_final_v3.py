import pickle
import pandas as pd
from flask import Flask, request, jsonify, flash
from Monthly_Revenue_Forecast import preprocess_data, predict_next_month, read_remote_data
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['DEBUG'] = True

with open('arima_results.pkl', 'rb') as f:
    model = pickle.load(f)

url = "https://github.com/aurimas13/Revenue-Forecast/blob/main/Dataset/Barbora%20Homework.xlsx?raw=true"

UPLOAD_FOLDER = 'Dataset'
ALLOWED_EXTENSIONS = {'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Cache the data by reading it once
revenue_data, weather_data = read_remote_data(url)

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    global revenue_data, weather_data

    if request.method == 'GET':
        # Update cached data with the latest data
        revenue_data, weather_data = read_remote_data(url)

    elif request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'})
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format'})

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        revenue_input = pd.read_excel(file_path, sheet_name='November_Revenue', parse_dates=['Date'], engine='openpyxl')
        weather_input = pd.read_excel(file_path, sheet_name='November_Weather', parse_dates=['dt'], engine='openpyxl')

        # Update cached data
        revenue_data = revenue_data.append(revenue_input)
        weather_data = weather_data.append(weather_input)

    # Preprocess the data
    preprocessed_data = preprocess_data(revenue_data, weather_data)

    # Make a prediction
    prediction = predict_next_month(model, preprocessed_data)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
